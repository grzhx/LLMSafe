from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import AttackRequest, run_attack  # noqa: E402


def main() -> None:
    req = AttackRequest(
        dataset_id="vqav2-mini-v1",
        model_name="api_openai_compatible",
        scenario="visual_question_answering",
        attack_name="blackbox_random",
        max_samples=100,
        targeted=True,
        target_label="yes",
        epsilon=0.12,
        alpha=0.04,
        steps=6,
        query_budget=24,
        universal_budget=8,
        api_cooldown_ms=1500,
    )
    result = run_attack(req)
    summary = {
        "experiment_id": result["experiment_id"],
        "dataset_id": result["dataset_id"],
        "model_name": result["model_name"],
        "scenario": result["scenario"],
        "attack_name": result["attack_name"],
        "aggregate_metrics": result["aggregate_metrics"],
        "record_count": len(result["records"]),
        "failed_samples": sum(1 for item in result["records"] if item.get("attack_debug", {}).get("error")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
