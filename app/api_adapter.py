from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass
class ApiStatus:
    provider: str
    available: bool
    message: str


class VisionApiAdapter:
    def __init__(self, class_names: List[str], model_registry, image_size: int, device: torch.device) -> None:
        self.class_names = class_names
        self.registry = model_registry
        self.image_size = image_size
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def statuses(self) -> List[Dict]:
        return [
            ApiStatus(provider="mock", available=True, message="Local mock multimodal API is available.").__dict__,
            ApiStatus(
                provider="openai_compatible",
                available=True,
                message="Requires api_base_url, api_key and a chat-completions compatible multimodal model.",
            ).__dict__,
        ]

    def _parse_forced_label(self, prompt: str) -> Optional[str]:
        lowered = prompt.lower()
        for label in self.class_names:
            if re.search(rf"(ignore|answer|respond|output).{{0,30}}\b{re.escape(label)}\b", lowered):
                return label
        return None

    def _predict_with_local_model(self, image: Image.Image) -> Dict:
        model = self.registry.get("resnet18_demo")
        model.eval()
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().squeeze(0)
        index = int(torch.argmax(probs).item())
        return {
            "label": self.class_names[index],
            "confidence": float(probs[index].item()),
            "probabilities": [round(float(v), 4) for v in probs.tolist()],
        }

    def classify_mock(self, image: Image.Image, prompt: str, candidate_labels: List[str]) -> Dict:
        forced = self._parse_forced_label(prompt)
        base = self._predict_with_local_model(image)
        if forced and forced in candidate_labels:
            confidence = max(base["confidence"], 0.95)
            answer = forced
            mode = "prompt_injected"
        else:
            answer = base["label"]
            confidence = base["confidence"]
            mode = "vision_reasoning"
        return {
            "provider": "mock",
            "answer": answer,
            "confidence": round(confidence, 4),
            "probabilities": base["probabilities"],
            "mode": mode,
            "raw_text": f"The image is classified as {answer}.",
        }

    def classify_openai_compatible(
        self,
        image: Image.Image,
        prompt: str,
        candidate_labels: List[str],
        api_base_url: str,
        api_key: str,
        api_model: str,
    ) -> Dict:
        if not api_base_url or not api_key or not api_model:
            raise ValueError("api_base_url, api_key and api_model are required for openai_compatible provider")

        content = [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n"
                    f"Candidate labels: {', '.join(candidate_labels)}.\n"
                    "Return JSON with keys label and reason."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image)},
            },
        ]
        payload = {
            "model": api_model,
            "messages": [{"role": "user", "content": content}],
        }
        response = requests.post(
            api_base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if not response.ok:
            detail = response.text
            try:
                error_payload = response.json()
                detail = error_payload.get("error", {}).get("message", detail)
            except Exception:
                pass
            raise ValueError(f"Remote API request failed: {response.status_code} {detail}")
        result = response.json()
        raw_text = result["choices"][0]["message"]["content"]
        label = None
        if isinstance(raw_text, str):
            try:
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                parsed = json.loads(cleaned)
                label = parsed.get("label")
            except Exception:
                lowered = raw_text.lower()
                for candidate in candidate_labels:
                    if candidate in lowered:
                        label = candidate
                        break
        if label not in candidate_labels:
            raise ValueError("Could not parse label from API response")
        return {
            "provider": "openai_compatible",
            "answer": label,
            "confidence": None,
            "probabilities": [],
            "mode": "remote_api",
            "raw_text": raw_text,
        }

    def classify(
        self,
        image: Image.Image,
        prompt: str,
        candidate_labels: List[str],
        provider: str,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_model: Optional[str] = None,
    ) -> Dict:
        if provider == "mock":
            return self.classify_mock(image, prompt, candidate_labels)
        if provider == "openai_compatible":
            return self.classify_openai_compatible(
                image,
                prompt,
                candidate_labels,
                api_base_url=api_base_url or "",
                api_key=api_key or "",
                api_model=api_model or "",
            )
        raise ValueError(f"Unsupported provider: {provider}")
