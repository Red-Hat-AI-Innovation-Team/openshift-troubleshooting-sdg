"""Deploy a single scenario to the cluster.

Reads from scenarios.jsonl and deploys the specified scenario by index
or name. Creates the namespace via oc new-project and applies manifests.

Usage:
    uv run python scripts/deploy_scenario.py 3                     # by index (1-based)
    uv run python scripts/deploy_scenario.py --name "Broken Httpd"  # by name substring
    uv run python scripts/deploy_scenario.py --list                 # list all scenarios
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_FILE = REPO_ROOT / "scenario_pipeline_output" / "scenarios.jsonl"

PROTECTED_NAMESPACES = {"default"}
DOCKERHUB_PREFIXES = [
    "python:", "busybox", "nginx", "alpine", "node:", "postgres:",
    "redis:", "mysql:", "mongo:", "httpd:", "ubuntu:", "debian:",
]


def load_scenarios(path=None):
    path = path or SCENARIOS_FILE
    with open(path) as f:
        return [json.loads(line) for line in f]


def list_scenarios(scenarios):
    print(f"{'#':>3}  {'Name':<50} {'Namespace':<35} {'Difficulty'}")
    print("-" * 120)
    for i, s in enumerate(scenarios):
        print(f"{i+1:3d}  {s.get('scenario_name', '?'):<50} {s.get('namespace', '?'):<35} {s.get('difficulty', '?')}")


def deploy(scenario):
    name = scenario.get("scenario_name", "unknown")
    namespace = scenario.get("namespace", "")
    manifests = scenario.get("manifests", "")

    if namespace in PROTECTED_NAMESPACES:
        print(f"SKIP: namespace '{namespace}' is protected")
        return False

    for prefix in DOCKERHUB_PREFIXES:
        if prefix in manifests:
            print(f"SKIP: manifests contain Docker Hub image '{prefix}...'")
            return False

    # Create project
    result = subprocess.run(
        ["oc", "new-project", namespace],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(f"FAIL: oc new-project: {result.stderr[:200]}")
        return False

    # Strip Namespace/Project resources, fix separators
    docs = []
    for doc in manifests.split("---"):
        doc = doc.strip()
        if not doc or "kind: Namespace" in doc or "kind: Project" in doc:
            continue
        docs.append(doc)

    if not docs:
        print("SKIP: no non-namespace resources in manifests")
        return False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("\n---\n".join(docs))
        manifest_path = f.name

    try:
        result = subprocess.run(
            ["oc", "apply", "-f", manifest_path, "-n", namespace],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"FAIL: oc apply: {result.stderr[:300]}")
            return False

        print(f"Deployed '{name}' to namespace '{namespace}'")
        print(f"  Check: oc get pods -n {namespace}")
        print(f"  Events: oc get events -n {namespace}")
        return True
    finally:
        os.unlink(manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Deploy a scenario to the cluster")
    parser.add_argument("index", nargs="?", type=int, help="Scenario index (1-based)")
    parser.add_argument("--name", type=str, help="Scenario name substring to match")
    parser.add_argument("--list", action="store_true", help="List all scenarios")
    parser.add_argument("--file", type=str, default=None, help="Path to scenarios.jsonl")
    args = parser.parse_args()

    scenarios_path = Path(args.file) if args.file else SCENARIOS_FILE
    if not scenarios_path.exists():
        print(f"ERROR: {scenarios_path} not found. Run generate_scenarios.py first.")
        sys.exit(1)

    scenarios = load_scenarios(scenarios_path)

    if args.list:
        list_scenarios(scenarios)
        return

    if args.name:
        matches = [(i, s) for i, s in enumerate(scenarios) if args.name.lower() in s.get("scenario_name", "").lower()]
        if not matches:
            print(f"No scenario matching '{args.name}'")
            list_scenarios(scenarios)
            sys.exit(1)
        if len(matches) > 1:
            print(f"Multiple matches for '{args.name}':")
            for i, s in matches:
                print(f"  [{i+1}] {s.get('scenario_name')}")
            sys.exit(1)
        idx, scenario = matches[0]
    elif args.index:
        idx = args.index - 1
        if idx < 0 or idx >= len(scenarios):
            print(f"ERROR: index {args.index} out of range (1-{len(scenarios)})")
            sys.exit(1)
        scenario = scenarios[idx]
    else:
        print("Provide a scenario index or --name. Use --list to see all.")
        sys.exit(1)

    print(f"Deploying [{idx+1}/{len(scenarios)}]: {scenario.get('scenario_name')}")
    print(f"  Namespace: {scenario.get('namespace')}")
    print(f"  Root cause: {scenario.get('root_cause', '')[:100]}...")
    print()
    deploy(scenario)


if __name__ == "__main__":
    main()
