# SPDX-License-Identifier: Apache-2.0
"""OpenShift Lightspeed agent framework connector."""

from typing import Any

from sdg_hub.core.utils.logger_config import setup_logger
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry
from sdg_hub.core.connectors.agent.base import BaseAgentConnector

logger = setup_logger(__name__)


@ConnectorRegistry.register("lightspeed")
class LightspeedConnector(BaseAgentConnector):
    """Connector for OpenShift Lightspeed service.

    OpenShift Lightspeed is an AI-powered troubleshooting assistant that
    uses MCP tools to diagnose Kubernetes/OpenShift cluster issues. This
    connector handles the specific request/response format used by the
    Lightspeed ``/v1/query`` API.

    Lightspeed expects:
    - A query string (extracted from the last user message)
    - A conversation_id for multi-turn tracking
    - Returns the response text plus ``tool_calls`` and ``tool_results`` lists

    Example
    -------
    >>> from sdg_hub.core.connectors.base import ConnectorConfig
    >>> from open_lightspeed.connectors import LightspeedConnector
    >>>
    >>> config = ConnectorConfig(
    ...     url="http://localhost:8080/v1/query",
    ...     api_key="your-api-key",
    ... )
    >>> connector = LightspeedConnector(config=config)
    >>> response = connector.send(
    ...     messages=[{"role": "user", "content": "Investigate the CrashLoopBackOff"}],
    ...     session_id="troubleshooting-001",
    ... )
    """

    def build_request(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Build Lightspeed-specific request payload.

        Lightspeed expects a query string, not a message array.
        We extract the last user message content.

        Parameters
        ----------
        messages : list[dict]
            Messages in standard format.
        session_id : str
            Session identifier, used as conversation_id.

        Returns
        -------
        dict
            Lightspeed API request payload.
        """
        query = self._extract_last_user_message(messages)

        return {
            "query": query,
            "conversation_id": session_id,
        }

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse Lightspeed response.

        Parameters
        ----------
        response : dict
            Raw response from Lightspeed API.

        Returns
        -------
        dict
            Validated response dict.

        Raises
        ------
        ConnectorError
            If response is not a valid dict.
        """
        if not isinstance(response, dict):
            raise ConnectorError(
                f"Expected dict response, got {type(response).__name__}"
            )

        return response

    # ------------------------------------------------------------------
    # Response field extraction
    # ------------------------------------------------------------------

    @classmethod
    def extract_text(cls, response: dict[str, Any]) -> str | None:
        """Extract text content from a Lightspeed response.

        Lightspeed returns the final answer at ``response["response"]``.

        Parameters
        ----------
        response : dict
            Raw Lightspeed API response.

        Returns
        -------
        str or None
            Extracted text, or None if not found.
        """
        text = response.get("response")
        if text is None:
            return None
        return text

    @classmethod
    def extract_session_id(cls, response: dict[str, Any]) -> str | None:
        """Extract conversation ID from a Lightspeed response.

        Parameters
        ----------
        response : dict
            Raw Lightspeed API response.

        Returns
        -------
        str or None
            Extracted conversation ID, or None if not found.
        """
        return response.get("conversation_id")

    @classmethod
    def extract_tool_trace(
        cls, response: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Extract a message trace from a Lightspeed response, if present.

        Returns ``response["messages"]`` when the response includes it. The
        ``/v1/query`` endpoint reports tool calls and results as separate
        ``tool_calls`` / ``tool_results`` lists, which the training formatter
        (``open_lightspeed.formatting``) reads directly.

        Parameters
        ----------
        response : dict
            Raw Lightspeed API response.

        Returns
        -------
        list[dict] or None
            Full message history, or None if not found.
        """
        messages = response.get("messages")
        if messages is None or not isinstance(messages, list):
            return None
        return messages

    def _extract_last_user_message(self, messages: list[dict[str, Any]]) -> str:
        """Extract the last user message content.

        Parameters
        ----------
        messages : list[dict]
            List of messages.

        Returns
        -------
        str
            Content of the last user message.

        Raises
        ------
        ConnectorError
            If no user message is found.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]

        raise ConnectorError(
            "No user message found in messages. "
            "Expected at least one message with role='user' and content."
        )
