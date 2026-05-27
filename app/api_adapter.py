from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
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
            ApiStatus(
                provider="mock",
                available=True,
                message="Local mock provider for offline pipeline verification.",
            ).__dict__,
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
            matched = next((label for label in candidate_labels if label.lower() == base["label"].lower()), None)
            answer = matched or (candidate_labels[0] if candidate_labels else base["label"])
            confidence = base["confidence"] if matched else max(base["confidence"], 0.55)
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
        messages: List[Dict],
        candidate_labels: List[str],
        api_base_url: str,
        api_key: str,
        api_model: str,
    ) -> Dict:
        if not api_base_url or not api_key or not api_model:
            raise ValueError("api_base_url, api_key and api_model are required for openai_compatible provider")
        payload = {
            "model": api_model,
            "messages": messages,
        }
        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    api_base_url.rstrip("/") + "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise ValueError(f"Remote API request failed: {exc}") from exc

            if response.ok:
                break

            detail = response.text
            try:
                error_payload = response.json()
                detail = error_payload.get("error", {}).get("message", detail)
            except Exception:
                pass

            retryable = response.status_code in {429, 500, 502, 503, 504}
            if retryable and attempt < 2:
                last_error = ValueError(f"{response.status_code} {detail}")
                time.sleep(2**attempt)
                continue
            raise ValueError(f"Remote API request failed: {response.status_code} {detail}")

        if response is None:
            raise ValueError(f"Remote API request failed: {last_error}")
        result = response.json()
        raw_text = result["choices"][0]["message"]["content"]
        if isinstance(raw_text, list):
            raw_text = "\n".join(
                item.get("text", "")
                for item in raw_text
                if isinstance(item, dict)
            )
        label = None
        options = {f"c{idx}": candidate for idx, candidate in enumerate(candidate_labels)}
        if isinstance(raw_text, str):
            try:
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                parsed = json.loads(cleaned)
                label = parsed.get("label")
                choice_id = parsed.get("choice_id")
                if choice_id in options:
                    label = options[choice_id]
            except Exception:
                lowered = raw_text.lower()
                for candidate in candidate_labels:
                    if candidate.lower() in lowered:
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

    def _flatten_message_text(self, value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            return str(value.get("text", ""))
        return ""

    def classify(
        self,
        image: Image.Image,
        candidate_labels: List[str],
        provider: str,
        prompt: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[List[Dict]] = None,
    ) -> Dict:
        if provider == "mock":
            prompt_parts: List[str] = []
            if system_prompt:
                prompt_parts.append(system_prompt)
            if prompt:
                prompt_parts.append(prompt)
            for message in extra_messages or []:
                prompt_parts.append(self._flatten_message_text(message.get("content")))
            return self.classify_mock(image, "\n".join(part for part in prompt_parts if part), candidate_labels)

        if provider == "openai_compatible":
            base_prompt = prompt or ""
            options_text = "\n".join(f"c{idx}: {candidate}" for idx, candidate in enumerate(candidate_labels))
            content = [
                {
                    "type": "text",
                    "text": (
                        f"{base_prompt}\n"
                        "Choose exactly one option from the candidate list below.\n"
                        f"{options_text}\n"
                        "Return strict JSON with keys choice_id, label, and reason."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image)},
                },
            ]
            messages: List[Dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if extra_messages:
                messages.extend(extra_messages)
            messages.append({"role": "user", "content": content})
            return self.classify_openai_compatible(
                messages,
                candidate_labels,
                api_base_url=api_base_url or "",
                api_key=api_key or "",
                api_model=api_model or "",
            )
        raise ValueError(f"Unsupported provider: {provider}")
