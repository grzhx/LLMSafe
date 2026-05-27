from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def _safe_copy(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        target_path.write_bytes(source_path.read_bytes())


def import_coco_captions_subset(
    dataset_id: str,
    dataset_name: str,
    image_root: Path,
    captions_json: Path,
    output_root: Path,
    limit: int = 200,
) -> tuple[Dict, List[Dict]]:
    payload = json.loads(captions_json.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload["images"]}
    grouped: Dict[int, List[str]] = {}
    for ann in payload["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann["caption"].strip())

    samples: List[Dict] = []
    selected_ids = list(grouped.keys())[:limit]
    all_positive_captions: List[str] = []
    for image_id in selected_ids:
        image_info = images.get(image_id)
        if not image_info:
            continue
        file_name = image_info["file_name"]
        source_path = image_root / file_name
        if not source_path.exists():
            continue
        captions = [caption for caption in grouped[image_id] if caption]
        if not captions:
            continue
        all_positive_captions.extend(captions[:5])

    distractor_pool = list(dict.fromkeys(all_positive_captions))
    for image_id in selected_ids:
        image_info = images.get(image_id)
        if not image_info:
            continue
        file_name = image_info["file_name"]
        source_path = image_root / file_name
        if not source_path.exists():
            continue
        width, height = _image_size(source_path)
        captions = [caption for caption in grouped[image_id] if caption][:5]
        if not captions:
            continue
        sample_id = f"coco-{image_id}"
        rel_path = f"storage/datasets/{dataset_id}/images/{file_name}"
        target_path = output_root / file_name
        _safe_copy(source_path, target_path)

        retrieval_candidates = captions[:]
        for candidate in distractor_pool:
            if candidate not in retrieval_candidates:
                retrieval_candidates.append(candidate)
            if len(retrieval_candidates) >= 10:
                break

        sample = {
            "sample_id": sample_id,
            "path": rel_path,
            "label": "captioning",
            "caption": captions[0],
            "captions": captions,
            "caption_candidates": retrieval_candidates[:5],
            "retrieval_candidates": retrieval_candidates,
            "positive_indices": list(range(len(captions))),
            "question": "Which caption best matches the image?",
            "answer": captions[0],
            "answer_candidates": captions[:1],
            "task_types": ["image_captioning", "image_text_retrieval"],
            "tags": ["benchmark", "coco", "captioning", "retrieval"],
            "version": "2.0.0",
            "split": image_info.get("split", "val"),
            "width": width,
            "height": height,
            "benchmark": "MSCOCO",
            "benchmark_image_id": image_id,
        }
        samples.append(sample)

    meta = {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "modality": "image-text",
        "task": "image_captioning,image_text_retrieval",
        "description": "Subset of MSCOCO captions benchmark for captioning and image-text retrieval experiments.",
        "sample_count": len(samples),
        "labels": ["captioning"],
        "version": "2.0.0",
        "benchmark_name": "MSCOCO Captions",
        "source": "local coco captions json",
        "primary_scenarios": ["image_text_retrieval", "image_captioning"],
    }
    return meta, samples


def import_vqav2_subset(
    dataset_id: str,
    dataset_name: str,
    image_root: Path,
    questions_json: Path,
    annotations_json: Path,
    output_root: Path,
    limit: int = 200,
) -> tuple[Dict, List[Dict]]:
    questions_payload = json.loads(questions_json.read_text(encoding="utf-8"))
    annotations_payload = json.loads(annotations_json.read_text(encoding="utf-8"))

    questions = {item["question_id"]: item for item in questions_payload["questions"]}
    answer_counter: Counter[str] = Counter()
    raw_items: List[Dict] = []
    for ann in annotations_payload["annotations"]:
        q = questions.get(ann["question_id"])
        if not q:
            continue
        answer_hist = Counter(answer["answer"].strip().lower() for answer in ann.get("answers", []) if answer.get("answer"))
        if not answer_hist:
            continue
        majority_answer, _ = answer_hist.most_common(1)[0]
        answer_counter.update(answer_hist)
        raw_items.append(
            {
                "question_id": ann["question_id"],
                "image_id": ann["image_id"],
                "question": q["question"].strip(),
                "majority_answer": majority_answer,
                "answer_hist": answer_hist,
            }
        )

    global_answer_pool = [answer for answer, _ in answer_counter.most_common(200)]
    samples: List[Dict] = []
    for item in raw_items[:limit]:
        file_name = f"COCO_val2014_{item['image_id']:012d}.jpg"
        source_path = image_root / file_name
        if not source_path.exists():
            alt_name = f"{item['image_id']:012d}.jpg"
            source_path = image_root / alt_name
        if not source_path.exists():
            continue
        width, height = _image_size(source_path)
        rel_path = f"storage/datasets/{dataset_id}/images/{source_path.name}"
        _safe_copy(source_path, output_root / source_path.name)

        answer_candidates = [item["majority_answer"]]
        for answer, _ in item["answer_hist"].most_common():
            if answer not in answer_candidates:
                answer_candidates.append(answer)
        for answer in global_answer_pool:
            if answer not in answer_candidates:
                answer_candidates.append(answer)
            if len(answer_candidates) >= 8:
                break

        samples.append(
            {
                "sample_id": f"vqav2-{item['question_id']}",
                "path": rel_path,
                "label": item["majority_answer"],
                "caption": "",
                "captions": [],
                "question": item["question"],
                "answer": item["majority_answer"],
                "answer_candidates": answer_candidates,
                "task_types": ["visual_question_answering"],
                "tags": ["benchmark", "vqav2", "vqa"],
                "version": "2.0.0",
                "split": "val",
                "width": width,
                "height": height,
                "benchmark": "VQAv2",
                "benchmark_image_id": item["image_id"],
                "question_id": item["question_id"],
                "answer_distribution": dict(item["answer_hist"].most_common()),
            }
        )

    meta = {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "modality": "image-text",
        "task": "visual_question_answering",
        "description": "Subset of VQAv2 benchmark for multimodal VQA prompt-injection and black-box attack evaluation.",
        "sample_count": len(samples),
        "labels": sorted({sample["answer"] for sample in samples}),
        "version": "2.0.0",
        "benchmark_name": "VQAv2",
        "source": "local vqav2 questions + annotations",
        "primary_scenarios": ["visual_question_answering"],
    }
    return meta, samples
