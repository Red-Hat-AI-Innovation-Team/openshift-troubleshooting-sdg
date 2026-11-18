"""Phase 1: Generate diverse K8s troubleshooting scenarios.

Runs the scenario_generation flow to produce deployment-ready manifests
for diverse failure modes. Output is saved as scenarios.jsonl.

Usage:
    uv run python scripts/generate_scenarios.py
    uv run python scripts/generate_scenarios.py --num-variants 4
    uv run python scripts/generate_scenarios.py --output my_scenarios.jsonl
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


import open_lightspeed  # noqa: F401 — registers flows + connectors
from open_lightspeed.config import OPENAI_API_KEY, TEACHER_MODEL, NUM_VARIANTS
from open_lightspeed.paths import SCENARIO_GENERATION_FLOW_ID, get_flow_path
from sdg_hub import Flow

REPO_ROOT = Path(__file__).parent.parent
SEED_DATASET = REPO_ROOT / "scenario_seed.parquet"


def main():
    parser = argparse.ArgumentParser(description="Generate K8s troubleshooting scenarios")
    parser.add_argument("--num-variants", type=int, default=NUM_VARIANTS,
                        help=f"Variants per failure category (default: {NUM_VARIANTS}, env: NUM_VARIANTS)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: scenario_pipeline_output/scenarios.jsonl)")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY required")
        sys.exit(1)

    # Create seed if needed
    if not SEED_DATASET.exists():
        print("Creating seed dataset...")
        subprocess.run([
            "uv", "run", "python",
            str(Path(__file__).parent / "create_scenario_seed.py"),
        ], check=True)

    seed_df = pd.read_parquet(SEED_DATASET)
    output_dir = REPO_ROOT / "scenario_pipeline_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "scenarios.jsonl"

    print(f"Seed: {len(seed_df)} failure categories")
    print(f"Variants per category: {args.num_variants}")
    print(f"Teacher model: {TEACHER_MODEL}")
    print(f"Output: {output_path}")
    print()

    flow = Flow.from_yaml(get_flow_path(SCENARIO_GENERATION_FLOW_ID))
    flow.set_model_config(model=TEACHER_MODEL, api_key=OPENAI_API_KEY)

    result = flow.generate(
        seed_df,
        runtime_params={"multiply_category_rows": {"num_samples": args.num_variants}},
        log_dir=str(output_dir),
        checkpoint_dir=str(output_dir / "scenario_checkpoints"),
    )

    if hasattr(result, "to_pandas"):
        result = result.to_pandas()

    result.to_json(output_path, orient="records", lines=True, default_handler=str)
    print(f"\nGenerated {len(result)} scenarios → {output_path}")

    for i, row in result.iterrows():
        print(f"  [{i+1:2d}] {row.get('scenario_name', '?')} ({row.get('difficulty', '?')}) — {row.get('scenario_quality_rating', '?')}")


if __name__ == "__main__":
    main()
