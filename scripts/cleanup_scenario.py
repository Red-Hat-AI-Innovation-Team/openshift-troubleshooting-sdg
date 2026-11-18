"""Clean up a scenario namespace from the cluster.

Usage:
    uv run python scripts/cleanup_scenario.py scenario-report-processor
    uv run python scripts/cleanup_scenario.py --all          # cleanup all scenario-* namespaces
    uv run python scripts/cleanup_scenario.py --index 3      # cleanup scenario by index from scenarios.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_FILE = REPO_ROOT / "scenario_pipeline_output" / "scenarios.jsonl"

PROTECTED_NAMESPACES = {"default"}


def delete_namespace(namespace):
    if namespace in PROTECTED_NAMESPACES:
        print(f"  SKIP: '{namespace}' is protected")
        return
    result = subprocess.run(
        ["oc", "delete", "project", namespace, "--ignore-not-found", "--wait=false"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        print(f"  Deleted: {namespace}")
    else:
        print(f"  Warning: {result.stderr[:100]}")


def main():
    parser = argparse.ArgumentParser(description="Clean up scenario namespaces")
    parser.add_argument("namespace", nargs="?", help="Namespace to delete")
    parser.add_argument("--all", action="store_true", help="Delete all scenario-* namespaces")
    parser.add_argument("--index", type=int, help="Delete scenario by index from scenarios.jsonl")
    args = parser.parse_args()

    if args.all:
        # Find all scenario-* projects
        result = subprocess.run(
            ["oc", "projects", "-q"],
            capture_output=True, text=True, timeout=30,
        )
        namespaces = [ns.strip() for ns in result.stdout.strip().split("\n")
                      if ns.strip().startswith("scenario-")]
        if not namespaces:
            print("No scenario-* namespaces found")
            return
        print(f"Cleaning up {len(namespaces)} scenario namespaces:")
        for ns in namespaces:
            delete_namespace(ns)

    elif args.index:
        if not SCENARIOS_FILE.exists():
            print(f"ERROR: {SCENARIOS_FILE} not found")
            sys.exit(1)
        with open(SCENARIOS_FILE) as f:
            scenarios = [json.loads(line) for line in f]
        idx = args.index - 1
        if idx < 0 or idx >= len(scenarios):
            print(f"ERROR: index {args.index} out of range (1-{len(scenarios)})")
            sys.exit(1)
        ns = scenarios[idx].get("namespace", "")
        print(f"Cleaning up [{args.index}] {scenarios[idx].get('scenario_name')}: {ns}")
        delete_namespace(ns)

    elif args.namespace:
        print(f"Cleaning up: {args.namespace}")
        delete_namespace(args.namespace)

    else:
        print("Provide a namespace, --index, or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
