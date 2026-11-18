"""Create the seed dataset for the scenario generation flow.

Defines 15 failure categories covering diverse Kubernetes troubleshooting
patterns. Each category becomes a row in the seed DataFrame, which the
scenario_generation flow multiplies and varies to produce diverse scenarios.

Usage:
    uv run python scripts/create_scenario_seed.py
    uv run python scripts/create_scenario_seed.py --output scenarios_seed.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Images that work on ROSA/AWS (Docker Hub is rate-limited)
IMAGE_POOL = [
    "registry.access.redhat.com/ubi9/python-311:latest",
    "registry.access.redhat.com/ubi9/ubi-minimal:latest",
    "registry.access.redhat.com/ubi9/httpd-24:latest",
    "quay.io/centos/centos:stream9",
]

# Full MCP tool list (from the OpenShift MCP server)
MCP_TOOLS = [
    "pods_list", "pods_list_in_namespace", "pods_get", "pods_log",
    "pods_top", "pods_exec", "pods_delete", "pods_run",
    "events_list", "namespaces_list", "projects_list",
    "nodes_top", "nodes_log", "nodes_stats_summary",
    "resources_get", "resources_list", "resources_create_or_update",
    "resources_delete", "resources_scale",
    "configuration_contexts_list",
]

EXCLUDED_SCENARIOS = (
    "Do NOT generate scenarios resembling any of these held-out evaluation scenarios: "
    "(1) OOM / memory leak causing CrashLoopBackOff with exit code 137 and memory limits, "
    "(2) NetworkPolicy label mismatch blocking traffic between frontend and backend pods, "
    "(3) Cross-namespace PostgreSQL connection pool exhaustion or database connection leak."
)

CATEGORIES = [
    {
        "failure_category": "crashloop_bad_entrypoint",
        "category_description": "Pod enters CrashLoopBackOff because the container command or entrypoint is wrong — the binary doesn't exist, the script has a syntax error, or the command exits immediately with a non-zero code.",
        "example_root_cause": "Deployment specifies 'python3 app.py' but the image doesn't contain app.py, causing FileNotFoundError on startup.",
        "complexity": "easy",
        "resource_types_pool": ["Deployment", "Pod"],
    },
    {
        "failure_category": "image_pull_backoff",
        "category_description": "Pod stuck in ImagePullBackOff because the container image reference is wrong — typo in image name, non-existent tag, or private registry without pull secret.",
        "example_root_cause": "Deployment references 'quay.io/myorg/myapp:v2.0' but tag v2.0 doesn't exist, only v1.0 does.",
        "complexity": "easy",
        "resource_types_pool": ["Deployment", "Pod"],
    },
    {
        "failure_category": "pending_resource_request",
        "category_description": "Pod stuck in Pending state because its resource requests (CPU or memory) exceed what any node can provide, or the namespace has a ResourceQuota that's been exceeded.",
        "example_root_cause": "Deployment requests 64Gi memory but the largest node only has 32Gi available.",
        "complexity": "easy",
        "resource_types_pool": ["Deployment", "Pod", "ResourceQuota"],
    },
    {
        "failure_category": "init_container_failure",
        "category_description": "Pod stuck in Init:Error or Init:CrashLoopBackOff because an init container fails. The main container never starts.",
        "example_root_cause": "Init container tries to check database connectivity with a wget to a service that doesn't exist yet, exits non-zero.",
        "complexity": "medium",
        "resource_types_pool": ["Deployment", "Pod", "Service"],
    },
    {
        "failure_category": "readiness_probe_misconfiguration",
        "category_description": "Pod is Running but never becomes Ready because the readiness probe targets the wrong port, path, or protocol. The Service never routes traffic to it.",
        "example_root_cause": "App listens on port 8080 but readiness probe checks port 8443. Pod stays Running but 0/1 Ready, endpoints list is empty.",
        "complexity": "medium",
        "resource_types_pool": ["Deployment", "Pod", "Service", "Endpoints"],
    },
    {
        "failure_category": "liveness_probe_misconfiguration",
        "category_description": "Pod repeatedly restarts because the liveness probe is too aggressive — the timeout or period is too short for the app's startup time, or the probe checks the wrong endpoint.",
        "example_root_cause": "Liveness probe has initialDelaySeconds=5 but app takes 30 seconds to start. Pod gets killed before it can become healthy.",
        "complexity": "medium",
        "resource_types_pool": ["Deployment", "Pod"],
    },
    {
        "failure_category": "missing_configmap_reference",
        "category_description": "Pod stuck in CreateContainerConfigError because the Deployment references a ConfigMap that doesn't exist in the namespace.",
        "example_root_cause": "Deployment mounts 'app-config' ConfigMap as a volume but the ConfigMap was never created.",
        "complexity": "easy",
        "resource_types_pool": ["Deployment", "Pod", "ConfigMap"],
    },
    {
        "failure_category": "missing_secret_reference",
        "category_description": "Pod stuck in CreateContainerConfigError because the Deployment references a Secret that doesn't exist or uses the wrong key name.",
        "example_root_cause": "Deployment uses envFrom to load 'db-credentials' Secret, but the Secret is named 'database-credentials' instead.",
        "complexity": "easy",
        "resource_types_pool": ["Deployment", "Pod", "Secret"],
    },
    {
        "failure_category": "wrong_env_var",
        "category_description": "Pod starts but the application crashes or misbehaves because an environment variable has the wrong value — wrong database host, wrong port number, or missing required config.",
        "example_root_cause": "DATABASE_HOST env var points to 'postgres.default.svc' but the database is actually in 'postgres.shared.svc'. App logs show connection refused.",
        "complexity": "medium",
        "resource_types_pool": ["Deployment", "Pod", "Service", "ConfigMap"],
    },
    {
        "failure_category": "missing_pvc",
        "category_description": "Pod stuck in Pending because it mounts a PersistentVolumeClaim that doesn't exist or is already bound to another pod.",
        "example_root_cause": "Deployment mounts 'data-volume' PVC but the PVC was never created. Pod event shows 'persistentvolumeclaim not found'.",
        "complexity": "easy",
        "resource_types_pool": ["Deployment", "Pod", "PersistentVolumeClaim"],
    },
    {
        "failure_category": "service_selector_mismatch",
        "category_description": "Service exists but has zero endpoints because its selector labels don't match any pod's labels. Traffic to the service times out or gets connection refused.",
        "example_root_cause": "Service selector is 'app: my-api' but pods have label 'app: myapi' (missing hyphen). Endpoints list shows 0 addresses.",
        "complexity": "medium",
        "resource_types_pool": ["Deployment", "Pod", "Service", "Endpoints"],
    },
    {
        "failure_category": "wrong_service_port",
        "category_description": "Service routes to pods but targets the wrong container port. Connections to the service fail even though pods are running and healthy.",
        "example_root_cause": "Service targetPort is 8443 but the container only listens on 8080. Connections to the service get 'connection refused'.",
        "complexity": "medium",
        "resource_types_pool": ["Deployment", "Pod", "Service"],
    },
    {
        "failure_category": "http_500_app_error",
        "category_description": "Pod is Running and Ready but the application returns HTTP 500 errors. The app starts fine but crashes on certain requests due to a bug or missing dependency.",
        "example_root_cause": "Python web app starts successfully but a required Python module is missing. All requests to /api return 500 with ImportError in logs.",
        "complexity": "medium",
        "resource_types_pool": ["Deployment", "Pod", "Service"],
    },
    {
        "failure_category": "stuck_rolling_update",
        "category_description": "Deployment is stuck mid-rollout — new ReplicaSet pods are failing (CrashLoopBackOff or not Ready) while old pods still serve traffic. The rollout never completes.",
        "example_root_cause": "Deployment updated to a new image tag that has a broken entrypoint. New pods crash, old pods keep running. 'oc rollout status' hangs.",
        "complexity": "hard",
        "resource_types_pool": ["Deployment", "Pod", "ReplicaSet"],
    },
    {
        "failure_category": "multi_resource_dependency",
        "category_description": "Application pod is Running but can't function because a required backend service or dependency is unavailable. Requires correlating failures across multiple resources.",
        "example_root_cause": "Frontend app depends on 'backend-api' Service, but the backend Deployment has 0 replicas (was scaled down). Frontend logs show 'connection refused to backend-api:8080'.",
        "complexity": "hard",
        "resource_types_pool": ["Deployment", "Pod", "Service"],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Create scenario generation seed dataset")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    rows = []
    for cat in CATEGORIES:
        rows.append({
            **cat,
            "image_registry_pool": IMAGE_POOL,
            "available_mcp_tools": MCP_TOOLS,
            "excluded_scenarios": EXCLUDED_SCENARIOS,
        })

    df = pd.DataFrame(rows)

    repo_root = Path(__file__).parent.parent
    output_path = Path(args.output) if args.output else repo_root / "scenario_seed.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Created seed dataset: {output_path}")
    print(f"  {len(df)} failure categories")
    print(f"  {len(IMAGE_POOL)} base images in pool")
    print(f"  {len(MCP_TOOLS)} MCP tools")
    print()
    for i, row in df.iterrows():
        print(f"  [{i+1:2d}] {row['failure_category']} ({row['complexity']})")

    # Also save as JSON for inspection
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nJSON: {json_path}")


if __name__ == "__main__":
    main()
