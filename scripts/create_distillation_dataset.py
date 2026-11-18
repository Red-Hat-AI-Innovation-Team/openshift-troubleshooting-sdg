"""Fetch tool schemas from the OpenShift MCP server and create the distillation input dataset.

Connects to a running MCP server, discovers all available tools and their
schemas, then produces a single-row dataset suitable for the sdg_hub
``mcp_distillation`` flow.

Prerequisites:
    - openshift-mcp-server running (e.g. kubernetes-mcp-server --config mcp_config.toml)
    - The ``mcp`` Python package installed

Usage:
    uv run python scripts/create_distillation_dataset.py
    uv run python scripts/create_distillation_dataset.py --url http://localhost:8085/mcp
"""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_MCP_URL = "http://localhost:8085/mcp"
SERVER_NAME = "OpenShift MCP Server"
SERVER_DESCRIPTION = (
    "Kubernetes/OpenShift cluster management server. "
    "Provides tools for listing pods, reading logs, viewing events, "
    "inspecting and managing resources, executing commands in pods, "
    "and managing namespaces. Denies access to ServiceAccounts, "
    "Secrets, and RBAC resources."
)


async def fetch_tools(url: str) -> list[dict]:
    """Connect to the MCP server and return tool schemas."""
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tool_list = []
            for tool in tools_response.tools:
                tool_list.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema
                        if hasattr(tool, "inputSchema")
                        else {"type": "object", "properties": {}},
                    }
                )
            return tool_list


def main():
    parser = argparse.ArgumentParser(
        description="Create distillation input dataset from MCP server"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_MCP_URL,
        help=f"MCP server URL (default: {DEFAULT_MCP_URL})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: repo root)",
    )
    args = parser.parse_args()

    tool_list = asyncio.run(fetch_tools(args.url))
    print(f"Fetched {len(tool_list)} tools from {args.url}")
    for t in tool_list:
        desc = t["description"][:60] + "..." if len(t["description"]) > 60 else t["description"]
        print(f"  - {t['name']}: {desc}")

    df = pd.DataFrame(
        {
            "tool_list": [tool_list],
            "mcp_server_name": [SERVER_NAME],
            "mcp_server_description": [SERVER_DESCRIPTION],
        }
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "distillation_input.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved dataset to {parquet_path}")

    json_path = output_dir / "distillation_input.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "tool_list": tool_list,
                "mcp_server_name": SERVER_NAME,
                "mcp_server_description": SERVER_DESCRIPTION,
            },
            f,
            indent=2,
        )
    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
