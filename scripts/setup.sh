#!/usr/bin/env bash
# Bootstrap the open-lightspeed environment.
#
# This script:
#   1. Installs the Python package (sdg_hub + open-lightspeed)
#   2. Creates openai_key.txt for lightspeed-service
#   3. Creates the scenario seed dataset and the distillation input dataset
#      (the latter fetches MCP tool schemas, so openshift-mcp-server must be running)
#
# Prerequisites:
#   - Python 3.10+ and uv installed
#   - openai_key.txt in this repo root (or OPENAI_API_KEY in .env)
#
# Usage:
#   ./scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "============================================================"
echo "  open-lightspeed Setup"
echo "  Root: $REPO_ROOT"
echo "============================================================"

# ── Step 1: Python environment ───────────────────────────────────────
echo ""
echo "=== Step 1: Python environment ==="

echo "Syncing virtual environment..."
uv sync --extra dev

echo "Installed: open-lightspeed + sdg_hub"

# ── Step 2: OpenAI key ───────────────────────────────────────────────
echo ""
echo "=== Step 2: OpenAI API key ==="

if [ -f ".env" ]; then
    echo "Found .env file"
elif [ -f "openai_key.txt" ]; then
    echo "Found openai_key.txt"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY set in environment"
else
    echo "WARNING: No API key found. Create .env with OPENAI_API_KEY or openai_key.txt"
fi

# Create openai_key.txt for lightspeed-service (it reads from file)
if [ ! -f "openai_key.txt" ] && [ -f ".env" ]; then
    # Extract from .env
    KEY=$(grep OPENAI_API_KEY .env | head -1 | sed 's/OPENAI_API_KEY=//' | tr -d '"')
    if [ -n "$KEY" ]; then
        echo "$KEY" > openai_key.txt
        echo "Created openai_key.txt from .env"
    fi
fi

# ── Step 3: Create datasets ─────────────────────────────────────────
echo ""
echo "=== Step 3: Create datasets ==="

# Scenario seed (always, no cluster needed)
if [ ! -f "scenario_seed.parquet" ]; then
    echo "Creating scenario seed dataset..."
    uv run python scripts/create_scenario_seed.py
else
    echo "scenario_seed.parquet exists"
fi

# Distillation input (needs MCP server running)
if [ ! -f "distillation_input.parquet" ]; then
    if curl -s http://localhost:8085/healthz &>/dev/null; then
        echo "Creating distillation input dataset (MCP tool schemas)..."
        uv run python scripts/create_distillation_dataset.py
    else
        echo "MCP server not running — skipping distillation_input.parquet"
        echo "Create it later: uv run python scripts/create_distillation_dataset.py"
    fi
else
    echo "distillation_input.parquet exists"
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Start services (in separate terminals):"
echo "     # Terminal 1: openshift-mcp-server"
echo "     kubernetes-mcp-server --config mcp_config.toml"
echo ""
echo "     # Terminal 2: lightspeed-service (from your lightspeed-service checkout)"
echo "     make run"
echo ""
echo "  2. Generate scenarios:"
echo "     uv run python scripts/generate_scenarios.py"
echo ""
echo "  3. Deploy + distill:"
echo "     uv run python scripts/deploy_scenario.py --list"
echo "     uv run python scripts/deploy_scenario.py 1"
echo "     uv run python scripts/run_distillation.py"
echo ""
echo "  Or run the full pipeline:"
echo "     uv run python scripts/run_scenario_pipeline.py"
echo ""
