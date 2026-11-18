"""Flow ID constants and path resolution for open-lightspeed."""

from sdg_hub import FlowRegistry

# Flow IDs
DISTILLATION_FLOW_ID = "new-night-835"  # sdg_hub built-in mcp_distillation
SCENARIO_GENERATION_FLOW_ID = "whole-thunder-38"  # open-lightspeed scenario_generation


def get_flow_path(flow_id: str) -> str:
    """Resolve a flow YAML path by its registry ID."""
    FlowRegistry.discover_flows()
    return FlowRegistry.get_flow_path(flow_id)
