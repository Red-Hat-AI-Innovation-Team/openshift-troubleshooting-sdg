"""Run the OpenShift MCP distillation pipeline end-to-end.

Uses the existing ``mcp_distillation`` flow from sdg_hub, reconfigured at
runtime to use the OpenShift Lightspeed service via ``set_agent_config()``.

Prerequisites:
    - OpenShift cluster with scenarios deployed
    - OpenShift MCP server running on port 8085
    - Lightspeed service running on port 8080
    - OPENAI_API_KEY environment variable set

Usage:
    OPENAI_API_KEY=sk-... uv run python scripts/run_distillation.py --k 3 --num-samples 100
    OPENAI_API_KEY=sk-... uv run python scripts/run_distillation.py --k 2 --num-samples 50
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import open_lightspeed  # noqa: F401 — registers flows + connectors
from open_lightspeed.config import (
    OPENAI_API_KEY, TEACHER_MODEL, LIGHTSPEED_URL, MCP_SERVER_URL,
    NUM_SAMPLES, TOOL_K,
)
from open_lightspeed.formatting import format_training_messages
from open_lightspeed.paths import DISTILLATION_FLOW_ID, get_flow_path
from sdg_hub import Flow

REPO_ROOT = Path(__file__).parent.parent
INPUT_DATASET = REPO_ROOT / "distillation_input.parquet"


def main():
    parser = argparse.ArgumentParser(description="Run OpenShift MCP distillation")
    parser.add_argument(
        "--k", type=int, default=TOOL_K,
        help=f"Number of tools to sample per question (default: {TOOL_K}, env: TOOL_K)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=NUM_SAMPLES,
        help=f"Number of question candidates to generate (default: {NUM_SAMPLES}, env: NUM_SAMPLES)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: distillation_output/k{k}_n{num_samples}_<timestamp>)",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    # Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_name = f"k{args.k}_n{args.num_samples}_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = REPO_ROOT / "distillation_output" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("  OpenShift MCP Distillation")
    print(f"  k={args.k} tools per question, {args.num_samples} candidates")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Step 1: Create input dataset if missing
    if not INPUT_DATASET.exists():
        print("\nInput dataset not found. Creating it...")
        subprocess.run(
            [
                "uv", "run", "python",
                str(Path(__file__).parent / "create_distillation_dataset.py"),
                "--url", MCP_SERVER_URL,
            ],
            check=True,
        )

    # Step 2: Load input dataset
    df = pd.read_parquet(INPUT_DATASET)
    print(f"\nInput: {len(df)} row(s), {len(df['tool_list'][0])} tools")

    # Step 3: Load flow
    flow = Flow.from_yaml(get_flow_path(DISTILLATION_FLOW_ID))

    # Step 4: Configure agent backend (lightspeed)
    flow.set_agent_config(
        agent_framework="lightspeed",
        agent_url=LIGHTSPEED_URL,
    )
    # The response extractor blocks are not auto-detected as agent blocks,
    # so point them at the lightspeed connector explicitly.
    flow.set_agent_config(
        agent_framework="lightspeed",
        blocks=["extract_exploration", "extract_agent_text"],
    )

    # Step 5: Configure teacher LLM
    flow.set_model_config(model=TEACHER_MODEL, api_key=OPENAI_API_KEY)
    print(f"Teacher: {TEACHER_MODEL}")
    print(f"Agent: lightspeed @ {LIGHTSPEED_URL}")

    # Step 6: Run pipeline with runtime overrides for k and num_samples
    print(f"\nRunning pipeline (k={args.k}, num_samples={args.num_samples})...\n")

    result = flow.generate(
        df,
        runtime_params={
            "multiply_tool_rows": {"num_samples": args.num_samples},
            "sample_tools": {"num_samples": args.k},
        },
        log_dir=str(output_dir),
        checkpoint_dir=str(output_dir / "checkpoints"),
    )

    # Step 7: Convert to DataFrame
    if hasattr(result, "to_pandas"):
        result_df = result.to_pandas()
    else:
        result_df = result

    if len(result_df) == 0:
        print("No samples survived quality filters.")
        sys.exit(0)

    # Step 8: Format training messages
    print(f"\nFormatting {len(result_df)} examples into training conversations...")
    formatted = []
    for _, row in result_df.iterrows():
        msgs = format_training_messages(row)
        n_fc = sum(1 for m in msgs if "function_call" in m)
        formatted.append(json.dumps(msgs))
        print(f"  {n_fc} tool call(s) — {row.get('question', '')[:80]}...")

    result_df["messages"] = formatted

    # Step 9: Export
    export_cols = ["messages", "question", "target_tools",
                   "question_quality_rating", "completeness_rating"]
    df_export = result_df[[c for c in export_cols if c in result_df.columns]]

    jsonl_path = output_dir / "training_data.jsonl"
    df_export.to_json(jsonl_path, orient="records", lines=True)
    print(f"\nExported {len(df_export)} training examples to {jsonl_path}")

    # Also save full results
    full_path = output_dir / "full_results.jsonl"
    result_df.to_json(full_path, orient="records", lines=True, default_handler=str)
    print(f"Full results: {full_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE: k={args.k}, {args.num_samples} candidates → {len(df_export)} training examples")
    print(f"  Survival rate: {len(df_export)/args.num_samples*100:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
