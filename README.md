# openshift-troubleshooting-sdg

Training pipeline for OpenShift Lightspeed: it generates synthetic tool-use training data for fine-tuning a smaller model to replace larger frontier models in the Lightspeed troubleshooting agent.

Built on [sdg_hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub).

## How it works

1. **Generate scenarios:** an expert LLM designs diverse K8s troubleshooting scenarios and writes deployment-ready manifests.
2. **Deploy:** broken workloads are deployed to a live OpenShift cluster.
3. **Explore:** the same or a similar expert LLM investigates the broken cluster via MCP tools, producing expert tool-use trajectories.
4. **Format:** trajectories are converted to function-calling conversations for SFT training of a smaller model.

**Note:** the example uses OpenAI models as placeholders, but any expert LLM can be used.

In more detail:

- **Generate scenarios.**
  - `scripts/create_scenario_seed.py` defines 15 failure categories, each with a complexity level, resource types and allowed images.
  - For each category variant, the `scenario_generation` flow in `src/open_lightspeed/flows/` has the teacher model design a concrete scenario: the symptom a user would report, the namespace, the root cause and the tools needed.
  - The teacher model then writes deployable manifests and rates the result for realism, diagnosability, YAML validity and distinctiveness. Only "good" and "excellent" scenarios are kept.
- **Deploy.** Each scenario's manifests are applied to its own project with `oc apply`, and the pipeline waits up to two minutes for a pod to show a failure state.
- **Explore.** sdg_hub's `mcp_distillation` flow runs with [OpenShift Lightspeed](https://github.com/openshift/lightspeed-service) as the agent and [openshift-mcp-server](https://github.com/openshift/openshift-mcp-server) providing the tools:
  - the agent explores the cluster;
  - the teacher model writes questions from sampled tools and the exploration notes, and filters out weak ones;
  - the agent answers each question with real tool calls;
  - the teacher model drops incomplete answers.
- **Format.** Tool calls and results are converted into function-calling conversations (see [Output](#output)).

Two models are involved:

- the **teacher model** (`TEACHER_MODEL`, any LiteLLM model id) drives scenario design, manifest writing, question generation and the quality judges;
- the **agent's model** is whatever expert LLM lightspeed-service is configured with in its `olsconfig.yaml`; it produces the tool-use trajectories.

## Requirements

- An OpenShift cluster, with `oc` logged in as a user that can create and delete projects
- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- Go, to build openshift-mcp-server
- API keys for the teacher model and for the agent's model provider

## Setup

1. Configure the environment:

   ```bash
   cp .env.example .env          # set OPENAI_API_KEY
   set -a && source .env && set +a
   ```

2. Build and start openshift-mcp-server. `mcp_config.toml` sets the port (8085) and the tool policy:

   ```bash
   git clone https://github.com/openshift/openshift-mcp-server.git
   (cd openshift-mcp-server && go build -o kubernetes-mcp-server ./cmd/kubernetes-mcp-server)
   ./openshift-mcp-server/kubernetes-mcp-server --config mcp_config.toml
   ```

3. Install dependencies and create the input datasets. `setup.sh` also writes `openai_key.txt` from `.env` for lightspeed-service:

   ```bash
   ./scripts/setup.sh
   ```

4. Start lightspeed-service with a copy of `olsconfig.example.yaml`, filled in with your provider and model:

   ```bash
   git clone https://github.com/openshift/lightspeed-service.git
   cd lightspeed-service && git checkout a5608fed && uv sync
   OLS_CONFIG_FILE=/path/to/olsconfig.yaml uv run python runner.py    # serves http://localhost:8080
   ```

## Usage

End to end: generate scenarios, then deploy, explore and delete each one in turn.

```bash
uv run python scripts/run_scenario_pipeline.py                     # all categories, NUM_VARIANTS each
uv run python scripts/run_scenario_pipeline.py --skip-generation   # reuse scenario_pipeline_output/scenarios.jsonl
```

Step by step:

```bash
uv run python scripts/generate_scenarios.py           # -> scenario_pipeline_output/scenarios.jsonl
uv run python scripts/deploy_scenario.py --list
uv run python scripts/deploy_scenario.py 1            # deploy scenario 1 into its own project
uv run python scripts/run_distillation.py             # -> distillation_output/k3_n10_<timestamp>/
uv run python scripts/cleanup_scenario.py --index 1   # delete scenario 1's project
```

`scripts/format_training_data.py <run_dir>` rebuilds `training_data.jsonl` from a run's checkpoints.

## Configuration

Environment variables, read by `src/open_lightspeed/config.py`. Some scripts accept flags that override them (see `--help`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none (required) | API key passed to the teacher model |
| `TEACHER_MODEL` | set in `config.py` | LiteLLM model id for the teacher model |
| `LIGHTSPEED_URL` | `http://localhost:8080/v1/query` | Lightspeed query endpoint |
| `MCP_SERVER_URL` | `http://localhost:8085/mcp` | MCP endpoint used to fetch tool schemas |
| `NUM_VARIANTS` | `4` | Scenario variants per failure category |
| `NUM_SAMPLES` | `10` | Question candidates per distillation run |
| `TOOL_K` | `3` | Tools sampled per question |

`mcp_config.toml` configures openshift-mcp-server. Secrets, ServiceAccounts and RBAC resources are denied, write tools are enabled (`read_only = false`), and setting `read_only = true` limits the agent to read-only tools.

The deploy and cleanup scripts skip namespaces listed in `PROTECTED_NAMESPACES`. `cleanup_scenario.py --all` deletes every project whose name starts with `scenario-`.

## Output

- **`scenario_pipeline_output/scenarios.jsonl`** has one generated scenario per line: `scenario_name`, `scenario_description`, `namespace`, `root_cause`, `difficulty`, `expected_tools`, `manifests` and `scenario_quality_rating`.
- **Training data** has one example per line. It is written to `training_data.jsonl` in a distillation run directory, or to `scenario_pipeline_output/training_data_all.jsonl` by the pipeline. Each line has `messages` (a JSON string), `question`, `target_tools`, `question_quality_rating` and `completeness_rating`.

`messages` contains, in order:

1. a system message with the tool declarations, embedded as `<|im_system|>tool_declare<|im_middle|>[...]<|im_end|>`;
2. the user question;
3. for each tool call, an assistant message with `function_call`, followed by a `function` message with the result;
4. the final assistant answer.

The formatter is `src/open_lightspeed/formatting.py`.

## Tests

```bash
uv run pytest
```

## Layout

```
src/open_lightspeed/
  flows/scenario_generation/   sdg_hub flow and prompts for scenario and manifest generation
  connectors/lightspeed.py     sdg_hub agent connector for the Lightspeed /v1/query API
  formatting.py                traces to function-calling conversations
  config.py, paths.py          settings from environment variables, flow IDs
scripts/                       setup, generation, deploy and cleanup, distillation, formatting
mcp_config.toml                openshift-mcp-server configuration
olsconfig.example.yaml         lightspeed-service configuration template
tests/
```

## Dependencies

- **[sdg_hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub):** installed via pip, pinned in `pyproject.toml`
- **[openshift-mcp-server](https://github.com/openshift/openshift-mcp-server):** built from source, tested at commit `37a592c`
- **[lightspeed-service](https://github.com/openshift/lightspeed-service):** tested at commit `a5608fed`
