"""End-to-end pipeline: generate scenarios → deploy → distill → cleanup.

Phase 1: Runs the scenario_generation flow to produce diverse K8s
         troubleshooting scenarios with deployment-ready manifests.
Phase 2: For each scenario, deploys it to the cluster, runs the
         mcp_distillation flow to capture expert trajectories, then
         cleans up.

Prerequisites:
    - OpenShift cluster accessible via oc
    - MCP server running on port 8085
    - Lightspeed service running on port 8080
    - OPENAI_API_KEY environment variable set

Usage:
    OPENAI_API_KEY=sk-... uv run python scripts/run_scenario_pipeline.py
    OPENAI_API_KEY=sk-... uv run python scripts/run_scenario_pipeline.py --num-variants 2 --skip-generation
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


import open_lightspeed  # noqa: F401 — registers flows + connectors
from open_lightspeed.config import (
    OPENAI_API_KEY, TEACHER_MODEL, LIGHTSPEED_URL,
    NUM_VARIANTS, NUM_SAMPLES, TOOL_K,
)
from open_lightspeed.formatting import format_training_messages
from open_lightspeed.paths import (
    DISTILLATION_FLOW_ID,
    SCENARIO_GENERATION_FLOW_ID,
    get_flow_path,
)
from sdg_hub import Flow

REPO_ROOT = Path(__file__).parent.parent

SEED_DATASET = REPO_ROOT / "scenario_seed.parquet"
DISTILLATION_INPUT = REPO_ROOT / "distillation_input.parquet"
OUTPUT_DIR = REPO_ROOT / "scenario_pipeline_output"

# Namespaces we must never touch
PROTECTED_NAMESPACES = {"default"}

# Docker Hub image prefixes that will fail on ROSA
DOCKERHUB_PREFIXES = [
    "python:", "busybox", "nginx", "alpine", "node:", "postgres:",
    "redis:", "mysql:", "mongo:", "httpd:", "ubuntu:", "debian:",
]


def run_oc(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run an oc command and return the result."""
    return subprocess.run(
        ["oc"] + args,
        capture_output=True, text=True, timeout=timeout,
    )


def deploy_scenario(manifests: str, namespace: str) -> bool:
    """Deploy a scenario's manifests to the cluster. Returns True on success."""
    # Safety checks
    if namespace in PROTECTED_NAMESPACES:
        print(f"    SKIP: namespace '{namespace}' is protected")
        return False

    for prefix in DOCKERHUB_PREFIXES:
        if prefix in manifests:
            print(f"    SKIP: manifests contain Docker Hub image '{prefix}...'")
            return False

    # Create namespace via oc new-project (works on ROSA where oc apply
    # for Namespace resources is forbidden)
    result = run_oc(["new-project", namespace])
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(f"    FAIL: oc new-project failed: {result.stderr[:200]}")
        return False

    # Strip Namespace resources from manifests (already created above)
    filtered_docs = []
    for doc in manifests.split("---"):
        doc = doc.strip()
        if not doc:
            continue
        if "kind: Namespace" in doc or "kind: Project" in doc:
            continue
        filtered_docs.append(doc)

    if not filtered_docs:
        print("    SKIP: no non-namespace resources in manifests")
        return False

    filtered_manifests = "\n---\n".join(filtered_docs)

    # Write filtered manifests to temp file and apply
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(filtered_manifests)
        manifest_path = f.name

    try:
        result = run_oc(["apply", "-f", manifest_path, "-n", namespace])
        if result.returncode != 0:
            print(f"    FAIL: oc apply failed: {result.stderr[:200]}")
            return False
        print(f"    Applied manifests to namespace '{namespace}'")
        return True
    finally:
        os.unlink(manifest_path)


def wait_for_failure(namespace: str, timeout: int = 120) -> bool:
    """Wait for the failure to manifest (pod status change, events, etc.)."""
    print(f"    Waiting for failure to manifest (up to {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        result = run_oc(["get", "pods", "-n", namespace, "--no-headers"])
        if result.returncode != 0:
            time.sleep(5)
            continue

        lines = result.stdout.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            # Any non-Running/non-Completed status means something's wrong
            if any(s in line for s in [
                "CrashLoopBackOff", "Error", "ImagePullBackOff",
                "Pending", "Init:", "CreateContainerConfigError",
                "ErrImagePull", "0/1",
            ]):
                print(f"    Failure detected: {line.strip()[:80]}")
                return True

        time.sleep(10)

    # Timeout — still proceed, the agent might find something
    print("    Timeout waiting for failure, proceeding anyway")
    return True


def cleanup_scenario(namespace: str):
    """Delete the scenario namespace."""
    if namespace in PROTECTED_NAMESPACES:
        return
    result = run_oc([
        "delete", "project", namespace,
        "--ignore-not-found", "--wait=false",
    ])
    if result.returncode == 0:
        print(f"    Cleaned up namespace '{namespace}'")
    else:
        print(f"    Cleanup warning: {result.stderr[:100]}")


def run_distillation_for_scenario(
    scenario_name: str,
    namespace: str,
    output_dir: Path,
) -> pd.DataFrame | None:
    """Run the distillation flow against a deployed scenario."""
    if not DISTILLATION_INPUT.exists():
        print("    ERROR: distillation_input.parquet not found. Run create_distillation_dataset.py first.")
        return None

    df = pd.read_parquet(DISTILLATION_INPUT)

    flow = Flow.from_yaml(get_flow_path(DISTILLATION_FLOW_ID))

    # Configure lightspeed backend
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

    # Configure teacher LLM
    flow.set_model_config(model=TEACHER_MODEL, api_key=OPENAI_API_KEY)

    scenario_dir = output_dir / scenario_name.replace(" ", "_").lower()
    scenario_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = flow.generate(
            df,
            runtime_params={
                "multiply_tool_rows": {"num_samples": NUM_SAMPLES},
                "sample_tools": {"num_samples": TOOL_K},
            },
            log_dir=str(scenario_dir),
            checkpoint_dir=str(scenario_dir / "checkpoints"),
        )

        if hasattr(result, "to_pandas"):
            result = result.to_pandas()

        if len(result) > 0:
            return result
        else:
            print("    No training examples survived quality filters")
            return None
    except Exception as e:
        print(f"    Distillation error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run scenario generation + distillation pipeline")
    parser.add_argument("--num-variants", type=int, default=NUM_VARIANTS,
                        help=f"Variants per failure category (default: {NUM_VARIANTS}, env: NUM_VARIANTS)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip scenario generation, use existing scenarios.jsonl")
    parser.add_argument("--scenarios-file", type=str, default=None,
                        help="Path to pre-generated scenarios JSONL")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY required")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios_file = Path(args.scenarios_file) if args.scenarios_file else OUTPUT_DIR / "scenarios.jsonl"

    # ── Phase 1: Generate scenarios ──────────────────────────────────
    if not args.skip_generation:
        print("=" * 60)
        print("  Phase 1: Scenario Generation")
        print("=" * 60)

        if not SEED_DATASET.exists():
            print("Creating seed dataset...")
            subprocess.run([
                "uv", "run", "python",
                str(Path(__file__).parent / "create_scenario_seed.py"),
            ], check=True)

        seed_df = pd.read_parquet(SEED_DATASET)
        print(f"Loaded {len(seed_df)} failure categories")

        flow = Flow.from_yaml(get_flow_path(SCENARIO_GENERATION_FLOW_ID))
        flow.set_model_config(model=TEACHER_MODEL, api_key=OPENAI_API_KEY)
        print(f"Teacher model: {TEACHER_MODEL}")

        print(f"\nGenerating scenarios ({args.num_variants} variants per category)...\n")
        result = flow.generate(
            seed_df,
            runtime_params={
                "multiply_category_rows": {"num_samples": args.num_variants},
            },
            log_dir=str(OUTPUT_DIR),
            checkpoint_dir=str(OUTPUT_DIR / "scenario_checkpoints"),
        )

        if hasattr(result, "to_pandas"):
            result = result.to_pandas()

        result.to_json(scenarios_file, orient="records", lines=True, default_handler=str)
        print(f"\nGenerated {len(result)} scenarios → {scenarios_file}")
    else:
        print(f"Skipping generation, loading from {scenarios_file}")

    # ── Phase 2: Deploy + Distill ────────────────────────────────────
    print()
    print("=" * 60)
    print("  Phase 2: Deploy + Distill")
    print("=" * 60)

    with open(scenarios_file) as f:
        scenarios = [json.loads(line) for line in f]

    print(f"Processing {len(scenarios)} scenarios...\n")

    all_training_data = []
    stats = {"deployed": 0, "distilled": 0, "skipped": 0, "failed": 0}

    for i, scenario in enumerate(scenarios):
        name = scenario.get("scenario_name", f"scenario_{i}")
        namespace = scenario.get("namespace", f"scenario-{i}")
        manifests = scenario.get("manifests", "")

        print(f"\n{'━' * 60}")
        print(f"  [{i+1}/{len(scenarios)}] {name}")
        print(f"  Namespace: {namespace}")
        print(f"{'━' * 60}")

        # Deploy
        if not deploy_scenario(manifests, namespace):
            stats["skipped"] += 1
            continue

        stats["deployed"] += 1

        # Wait for failure
        wait_for_failure(namespace)

        # Distill
        print("    Running distillation...")
        result_df = run_distillation_for_scenario(name, namespace, OUTPUT_DIR)

        if result_df is not None and len(result_df) > 0:
            # Format training messages
            for _, row in result_df.iterrows():
                try:
                    msgs = format_training_messages(row)
                    all_training_data.append({
                        "messages": json.dumps(msgs),
                        "question": row.get("question", ""),
                        "target_tools": row.get("target_tools", ""),
                        "scenario_name": name,
                        "scenario_namespace": namespace,
                        "question_quality_rating": row.get("question_quality_rating", ""),
                        "completeness_rating": row.get("completeness_rating", ""),
                    })
                except Exception as e:
                    print(f"    Format error: {e}")

            stats["distilled"] += 1
            print(f"    → {len(result_df)} training examples")
        else:
            stats["failed"] += 1

        # Cleanup
        cleanup_scenario(namespace)

    # ── Export ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)

    if all_training_data:
        merged_path = OUTPUT_DIR / "training_data_all.jsonl"
        with open(merged_path, "w") as f:
            for item in all_training_data:
                f.write(json.dumps(item) + "\n")
        print(f"  Training examples: {len(all_training_data)}")
        print(f"  Output: {merged_path}")
    else:
        print("  No training data generated")

    print("\n  Stats:")
    print(f"    Scenarios generated: {len(scenarios)}")
    print(f"    Deployed: {stats['deployed']}")
    print(f"    Distilled: {stats['distilled']}")
    print(f"    Skipped: {stats['skipped']}")
    print(f"    Failed: {stats['failed']}")


if __name__ == "__main__":
    main()
