from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "vqav2"
VQAV2_IMAGE_BASE_URL = "https://images.cocodataset.org/val2014"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small VQAv2 subset for LLMSafe prompt-injection/VQA experiments."
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to v2_OpenEnded_mscoco_val2014_questions.json",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to v2_mscoco_val2014_annotations.json",
    )
    parser.add_argument(
        "--output-name",
        default="vqav2-mini-v1",
        help="Subset directory name under datasets/vqav2/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of question-answer items to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--min-answer-frequency",
        type=int,
        default=2,
        help="Minimum count of majority answer votes required",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def valid_annotation(item: Dict, min_answer_frequency: int) -> bool:
    answers = [answer.get("answer", "").strip().lower() for answer in item.get("answers", []) if answer.get("answer")]
    if not answers:
        return False
    majority_answer, majority_count = Counter(answers).most_common(1)[0]
    return bool(majority_answer) and majority_count >= min_answer_frequency


def sample_subset(
    questions_payload: Dict,
    annotations_payload: Dict,
    limit: int,
    seed: int,
    min_answer_frequency: int,
) -> tuple[List[Dict], List[Dict]]:
    questions = {item["question_id"]: item for item in questions_payload["questions"]}
    valid_annotations = [
        item
        for item in annotations_payload["annotations"]
        if item["question_id"] in questions and valid_annotation(item, min_answer_frequency)
    ]
    rng = random.Random(seed)
    rng.shuffle(valid_annotations)
    chosen_annotations = valid_annotations[:limit]
    chosen_question_ids = {item["question_id"] for item in chosen_annotations}
    chosen_questions = [item for item in questions_payload["questions"] if item["question_id"] in chosen_question_ids]
    return chosen_questions, chosen_annotations


def build_image_url(image_id: int) -> str:
    file_name = f"COCO_val2014_{image_id:012d}.jpg"
    return f"{VQAV2_IMAGE_BASE_URL}/{file_name}"


def write_outputs(
    output_dir: Path,
    subset_questions: List[Dict],
    subset_annotations: List[Dict],
    questions_source: Path,
    annotations_source: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions_subset.json"
    annotations_path = output_dir / "annotations_subset.json"
    urls_path = output_dir / "image_urls.txt"
    ids_path = output_dir / "image_ids.txt"
    preview_path = output_dir / "samples_preview.json"
    import_request_path = output_dir / "import_request.json"
    download_ps1_path = output_dir / "download_images.ps1"

    questions_payload = {
        "info": {
            "description": "LLMSafe VQAv2 subset questions",
            "source_questions": str(questions_source),
            "question_count": len(subset_questions),
        },
        "questions": subset_questions,
    }
    annotations_payload = {
        "info": {
            "description": "LLMSafe VQAv2 subset annotations",
            "source_annotations": str(annotations_source),
            "annotation_count": len(subset_annotations),
        },
        "annotations": subset_annotations,
    }
    questions_path.write_text(json.dumps(questions_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    annotations_path.write_text(json.dumps(annotations_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    unique_image_ids = sorted({item["image_id"] for item in subset_annotations})
    ids_path.write_text("\n".join(str(item) for item in unique_image_ids) + "\n", encoding="utf-8")

    urls = [build_image_url(image_id) for image_id in unique_image_ids]
    urls_path.write_text("\n".join(urls) + "\n", encoding="utf-8")

    question_map = {item["question_id"]: item for item in subset_questions}
    preview = []
    for item in subset_annotations[:20]:
        answers = [answer.get("answer", "") for answer in item.get("answers", []) if answer.get("answer")]
        preview.append(
            {
                "question_id": item["question_id"],
                "image_id": item["image_id"],
                "question": question_map[item["question_id"]]["question"],
                "majority_answer": Counter(answer.lower() for answer in answers).most_common(1)[0][0],
                "image_url": build_image_url(item["image_id"]),
            }
        )
    preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

    import_request = {
        "dataset_id": output_dir.name,
        "dataset_name": output_dir.name,
        "image_root": str((output_dir / "images").resolve()),
        "questions_json": str(questions_path.resolve()),
        "annotations_json": str(annotations_path.resolve()),
        "limit": len(subset_annotations),
    }
    import_request_path.write_text(json.dumps(import_request, ensure_ascii=False, indent=2), encoding="utf-8")

    download_ps1 = [
        "$ProgressPreference = 'SilentlyContinue'",
        f"$outDir = '{(output_dir / 'images').resolve()}'",
        "New-Item -ItemType Directory -Force -Path $outDir | Out-Null",
        f"Get-Content '{urls_path.resolve()}' | ForEach-Object {{",
        "  if (-not $_) { return }",
        "  $fileName = Split-Path $_ -Leaf",
        "  $target = Join-Path $outDir $fileName",
        "  if (-not (Test-Path $target)) { Invoke-WebRequest -Uri $_ -OutFile $target }",
        "}",
    ]
    download_ps1_path.write_text("\n".join(download_ps1) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions).resolve()
    annotations_path = Path(args.annotations).resolve()
    if not questions_path.exists():
        raise SystemExit(f"questions file not found: {questions_path}")
    if not annotations_path.exists():
        raise SystemExit(f"annotations file not found: {annotations_path}")

    questions_payload = load_json(questions_path)
    annotations_payload = load_json(annotations_path)
    subset_questions, subset_annotations = sample_subset(
        questions_payload,
        annotations_payload,
        args.limit,
        args.seed,
        args.min_answer_frequency,
    )
    if not subset_questions or not subset_annotations:
        raise SystemExit("no valid VQAv2 samples selected; try reducing --min-answer-frequency")

    output_dir = DATASETS_ROOT / args.output_name
    write_outputs(output_dir, subset_questions, subset_annotations, questions_path, annotations_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "subset_dir": str(output_dir),
                "question_count": len(subset_questions),
                "annotation_count": len(subset_annotations),
                "unique_image_count": len({item['image_id'] for item in subset_annotations}),
                "next_step": f"Run PowerShell script: {output_dir / 'download_images.ps1'}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
