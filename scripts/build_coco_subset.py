from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "coco"
COCO_IMAGE_BASE_URL = "https://images.cocodataset.org/val2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small COCO captions subset for LLMSafe retrieval/caption experiments."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to captions_val2017.json",
    )
    parser.add_argument(
        "--output-name",
        default="coco-caption-mini-v1",
        help="Subset directory name under datasets/coco/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of images to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--min-captions",
        type=int,
        default=3,
        help="Minimum number of captions required per sampled image",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sample_subset(payload: Dict, limit: int, seed: int, min_captions: int) -> tuple[List[Dict], List[Dict]]:
    images = {item["id"]: item for item in payload["images"]}
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for ann in payload["annotations"]:
        caption = ann.get("caption", "").strip()
        if caption:
            grouped[ann["image_id"]].append(ann)

    candidate_ids = [image_id for image_id, anns in grouped.items() if len(anns) >= min_captions and image_id in images]
    rng = random.Random(seed)
    rng.shuffle(candidate_ids)
    chosen_ids = candidate_ids[:limit]

    subset_images = [images[image_id] for image_id in chosen_ids]
    subset_annotations = []
    for image_id in chosen_ids:
        subset_annotations.extend(grouped[image_id])
    return subset_images, subset_annotations


def write_outputs(output_dir: Path, subset_images: List[Dict], subset_annotations: List[Dict], source_annotations: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_json_path = output_dir / "captions_val2017_subset.json"
    urls_path = output_dir / "image_urls.txt"
    ids_path = output_dir / "image_ids.txt"
    preview_path = output_dir / "samples_preview.json"
    import_request_path = output_dir / "import_request.json"
    download_ps1_path = output_dir / "download_images.ps1"

    payload = {
        "info": {
            "description": "LLMSafe COCO subset",
            "source_annotations": str(source_annotations),
            "image_count": len(subset_images),
            "annotation_count": len(subset_annotations),
        },
        "images": subset_images,
        "annotations": subset_annotations,
    }
    subset_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    image_ids = [str(item["id"]) for item in subset_images]
    ids_path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")

    urls = [f"{COCO_IMAGE_BASE_URL}/{item['file_name']}" for item in subset_images]
    urls_path.write_text("\n".join(urls) + "\n", encoding="utf-8")

    grouped_captions: Dict[int, List[str]] = defaultdict(list)
    for ann in subset_annotations:
        grouped_captions[ann["image_id"]].append(ann["caption"])
    preview = [
        {
            "image_id": item["id"],
            "file_name": item["file_name"],
            "url": f"{COCO_IMAGE_BASE_URL}/{item['file_name']}",
            "captions": grouped_captions[item["id"]][:5],
        }
        for item in subset_images[:20]
    ]
    preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

    import_request = {
        "dataset_id": output_dir.name,
        "dataset_name": output_dir.name,
        "image_root": str((output_dir / "images").resolve()),
        "captions_json": str(subset_json_path.resolve()),
        "limit": len(subset_images),
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
    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        raise SystemExit(f"annotations file not found: {annotations_path}")

    payload = load_annotations(annotations_path)
    subset_images, subset_annotations = sample_subset(payload, args.limit, args.seed, args.min_captions)
    if not subset_images:
        raise SystemExit("no samples selected; check annotations file or reduce --min-captions")

    output_dir = DATASETS_ROOT / args.output_name
    write_outputs(output_dir, subset_images, subset_annotations, annotations_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "subset_dir": str(output_dir),
                "image_count": len(subset_images),
                "annotation_count": len(subset_annotations),
                "next_step": f"Run PowerShell script: {output_dir / 'download_images.ps1'}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
