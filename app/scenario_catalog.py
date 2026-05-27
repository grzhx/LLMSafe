from __future__ import annotations

from typing import Dict, List


SCENARIOS: List[Dict] = [
    {
        "scenario_id": "clip_zero_shot_classification",
        "name": "CLIP Zero-Shot Classification",
        "task_type": "classification",
        "modality": "image-text",
        "description": "Evaluate zero-shot image-text classification robustness under image perturbation, visual prompt injection, and transfer attacks.",
        "supported_models": ["clip_vit_b32", "simple_cnn", "resnet18_demo"],
        "supported_attacks": ["fgsm", "pgd", "transfer_pgd", "blackbox_random", "prompt_injection"],
    },
    {
        "scenario_id": "image_text_retrieval",
        "name": "Image-Text Retrieval",
        "task_type": "retrieval",
        "modality": "image-text",
        "description": "Evaluate retrieval ranking robustness under contrastive attacks, transfer-based universal perturbations, and multimodal instruction injection.",
        "supported_models": ["clip_vit_b32", "api_openai_compatible"],
        "supported_attacks": ["fgsm", "pgd", "contrastive_pgd", "transfer_pgd", "blackbox_random", "prompt_injection"],
    },
    {
        "scenario_id": "visual_question_answering",
        "name": "Visual Question Answering",
        "task_type": "vqa",
        "modality": "image-text",
        "description": "Evaluate goal hijacking, visual OCR injection, and transfer-based black-box attacks on multimodal visual question answering.",
        "supported_models": ["api_openai_compatible"],
        "supported_attacks": ["prompt_injection", "blackbox_random", "transfer_pgd"],
    },
    {
        "scenario_id": "image_captioning",
        "name": "Image Captioning",
        "task_type": "captioning",
        "modality": "image-text",
        "description": "Evaluate multimodal caption selection robustness under visual instruction injection and transfer-based perturbations.",
        "supported_models": ["clip_vit_b32", "api_openai_compatible"],
        "supported_attacks": ["contrastive_pgd", "blackbox_random", "prompt_injection", "transfer_pgd"],
    },
]


ATTACK_CATALOG: List[Dict] = [
    {
        "attack_name": "fgsm",
        "type": "gradient",
        "description": "Single-step gradient-sign image attack baseline.",
        "whitebox": True,
    },
    {
        "attack_name": "pgd",
        "type": "gradient",
        "description": "Projected gradient descent image attack baseline.",
        "whitebox": True,
    },
    {
        "attack_name": "contrastive_pgd",
        "type": "contrastive",
        "description": "Contrastive and ranking-based CLIP retrieval attack.",
        "whitebox": True,
    },
    {
        "attack_name": "transfer_pgd",
        "type": "transfer",
        "description": "Transfer-based universal attack optimized on local surrogates or OpenCLIP text-image objectives.",
        "whitebox": False,
    },
    {
        "attack_name": "prompt_injection",
        "type": "visual_text",
        "description": "Visual prompt injection and goal hijacking with overlay text, low-contrast OCR, delayed injection, and task takeover.",
        "whitebox": False,
    },
    {
        "attack_name": "blackbox_random",
        "type": "blackbox_transfer",
        "description": "Transfer-based black-box attack entry evaluated under a fixed query budget.",
        "whitebox": False,
    },
]


def scenario_map() -> Dict[str, Dict]:
    return {item["scenario_id"]: item for item in SCENARIOS}
