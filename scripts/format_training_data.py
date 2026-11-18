"""Format distillation output into training-ready messages.

Reads checkpoint JSONL from a distillation run and converts to the
function-calling conversation format used for SFT training.

Usage:
    uv run python scripts/format_training_data.py distillation_output/<run_dir>
    uv run python scripts/format_training_data.py scenario_pipeline_output/<scenario_dir> --output training.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from open_lightspeed.formatting import format_training_messages


def main():
    parser = argparse.ArgumentParser(description="Format distillation output into training data")
    parser.add_argument("input_dir", help="Distillation output directory (contains checkpoints/)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    checkpoint_dir = input_dir / "checkpoints"

    # Find checkpoint files
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.jsonl"))
    if not checkpoints:
        print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
        sys.exit(1)

    # Read all checkpoint rows
    rows = []
    for cp in checkpoints:
        with open(cp) as f:
            for line in f:
                rows.append(json.loads(line))

    print(f"Found {len(rows)} samples across {len(checkpoints)} checkpoint(s)")

    # Format
    training_data = []
    for i, row in enumerate(rows):
        try:
            msgs = format_training_messages(row)
            n_fc = sum(1 for m in msgs if "function_call" in m)
            training_data.append({
                "messages": json.dumps(msgs),
                "question": row.get("question", ""),
                "target_tools": row.get("target_tools", ""),
                "question_quality_rating": row.get("question_quality_rating", ""),
                "completeness_rating": row.get("completeness_rating", ""),
            })
            q = row.get("question", "")[:80]
            print(f"  [{i+1}] {n_fc} tool call(s) — {q}...")
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")

    # Export
    output_path = Path(args.output) if args.output else input_dir / "training_data.jsonl"
    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nExported {len(training_data)} training examples → {output_path}")

    # Summary
    if training_data:
        total_calls = sum(
            sum(1 for m in json.loads(t["messages"]) if "function_call" in m)
            for t in training_data
        )
        print(f"  Total tool calls across all examples: {total_calls}")
        print(f"  Avg tool calls per example: {total_calls / len(training_data):.1f}")


if __name__ == "__main__":
    main()
