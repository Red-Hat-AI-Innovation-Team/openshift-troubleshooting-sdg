"""OpenShift Lightspeed training pipeline.

Registers custom flows and connectors with sdg_hub on import.

Usage:
    import open_lightspeed  # registers flows + connectors

    from sdg_hub import Flow, FlowRegistry
    FlowRegistry.discover_flows()  # includes open-lightspeed flows
"""

from pathlib import Path

from sdg_hub.core.flow.registry import FlowRegistry

# Register custom flows with sdg_hub
_FLOWS_DIR = Path(__file__).parent / "flows"
FlowRegistry.register_search_path(str(_FLOWS_DIR))

# Register lightspeed connector (import triggers @ConnectorRegistry.register)
from open_lightspeed.connectors.lightspeed import LightspeedConnector  # noqa: F401, E402

__all__ = ["LightspeedConnector"]
