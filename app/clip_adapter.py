from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _find_backend() -> Tuple[Optional[str], Optional[str]]:
    try:
        import open_clip  # type: ignore

        return "open_clip", open_clip.__name__
    except Exception:
        pass

    try:
        from transformers import CLIPModel  # type: ignore # noqa: F401

        return "transformers", "transformers"
    except Exception:
        pass

    return None, None


@dataclass
class ClipStatus:
    available: bool
    backend: Optional[str]
    package: Optional[str]
    message: str
    install_hints: List[str]


class ClipAdapter:
    def __init__(self, model_id: str = "ViT-B-32", pretrained: str = "openai") -> None:
        self.model_id = model_id
        self.pretrained = pretrained
        self.device = torch.device("cpu")
        self.backend, self.package = _find_backend()
        self.model = None
        self.preprocess = None
        self.processor = None
        self.tokenizer = None
        self.input_size = 224

    @staticmethod
    def status() -> ClipStatus:
        backend, package = _find_backend()
        if backend:
            return ClipStatus(
                available=True,
                backend=backend,
                package=package,
                message="CLIP backend detected.",
                install_hints=[],
            )
        return ClipStatus(
            available=False,
            backend=None,
            package=None,
            message="No CLIP backend detected. Install open_clip_torch or transformers before enabling CLIP attacks.",
            install_hints=[
                "pip install open_clip_torch",
                "pip install transformers",
            ],
        )

    def is_available(self) -> bool:
        return self.backend is not None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.backend:
            raise RuntimeError(self.status().message)

        if self.backend == "open_clip":
            import open_clip  # type: ignore

            model, _, preprocess = open_clip.create_model_and_transforms(self.model_id, pretrained=self.pretrained)
            self.model = model.eval().to(self.device)
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(self.model_id)
            image_size = getattr(getattr(model, "visual", None), "image_size", 224)
            if isinstance(image_size, tuple):
                self.input_size = int(image_size[0])
            else:
                self.input_size = int(image_size)
            return

        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        hf_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(hf_id).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(hf_id)
        self.input_size = 224

    def available_model_names(self) -> List[str]:
        if not self.is_available():
            return []
        return ["clip_vit_b32"]

    def default_prompts(self, labels: List[str]) -> List[str]:
        return [f"a photo of a {label}" for label in labels]

    def _attack_preprocess(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
        return transform(image).unsqueeze(0).to(self.device)

    def _deprocess(self, tensor: torch.Tensor) -> Image.Image:
        image = tensor.detach().cpu().clone().squeeze(0)
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        image = image * std + mean
        image = image.clamp(0, 1)
        array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

    def _tokenize(self, prompts: List[str]):
        if self.backend == "open_clip":
            return self.tokenizer(prompts)
        return self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)

    def _forward(self, image_tensor: torch.Tensor, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.load()
        if self.backend == "open_clip":
            text_tokens = self._tokenize(prompts).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            logits = image_features @ text_features.T
            return logits, logits.softmax(dim=-1)

        tokenized = self._tokenize(prompts)
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        image_features = self.model.get_image_features(pixel_values=image_tensor)
        text_features = self.model.get_text_features(**tokenized)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = image_features @ text_features.T
        return logits, logits.softmax(dim=-1)

    def predict_image(self, image: Image.Image, prompts: List[str]) -> Dict:
        tensor = self._attack_preprocess(image.convert("RGB"))
        logits, probs = self._forward(tensor, prompts)
        confidence, index = probs.max(dim=-1)
        return {
            "index": int(index.item()),
            "confidence": float(confidence.item()),
            "probabilities": [round(float(v), 4) for v in probs.squeeze(0).tolist()],
            "logits": [round(float(v), 4) for v in logits.squeeze(0).tolist()],
        }

    def predict_from_path(self, image_path: str | Path, prompts: List[str]) -> Dict:
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image, prompts)

    def fgsm_attack(
        self,
        image_path: str | Path,
        prompts: List[str],
        source_index: int,
        target_index: int,
        epsilon: float,
    ) -> Tuple[Image.Image, Dict]:
        image = Image.open(image_path).convert("RGB")
        adv = self._attack_preprocess(image).detach().clone().requires_grad_(True)
        logits, probs = self._forward(adv, prompts)
        loss = logits[0, source_index] - logits[0, target_index]
        self.model.zero_grad()
        loss.backward()
        adv = adv - epsilon * adv.grad.sign()
        logits_adv, probs_adv = self._forward(adv.detach(), prompts)
        return self._deprocess(adv.detach()), {
            "source_logit": float(logits[0, source_index].item()),
            "target_logit": float(logits[0, target_index].item()),
            "adversarial_source_logit": float(logits_adv[0, source_index].item()),
            "adversarial_target_logit": float(logits_adv[0, target_index].item()),
            "adversarial_probabilities": [round(float(v), 4) for v in probs_adv.squeeze(0).tolist()],
        }

    def pgd_attack(
        self,
        image_path: str | Path,
        prompts: List[str],
        source_index: int,
        target_index: int,
        epsilon: float,
        alpha: float,
        steps: int,
    ) -> Tuple[Image.Image, Dict]:
        image = Image.open(image_path).convert("RGB")
        original = self._attack_preprocess(image).detach()
        adv = original.clone()
        for _ in range(steps):
            adv.requires_grad_(True)
            logits, _ = self._forward(adv, prompts)
            loss = logits[0, source_index] - logits[0, target_index]
            self.model.zero_grad()
            loss.backward()
            adv = adv.detach() - alpha * adv.grad.sign()
            delta = torch.clamp(adv - original, min=-epsilon, max=epsilon)
            adv = original + delta
        logits_adv, probs_adv = self._forward(adv.detach(), prompts)
        return self._deprocess(adv.detach()), {
            "adversarial_source_logit": float(logits_adv[0, source_index].item()),
            "adversarial_target_logit": float(logits_adv[0, target_index].item()),
            "adversarial_probabilities": [round(float(v), 4) for v in probs_adv.squeeze(0).tolist()],
        }
