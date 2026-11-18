# SPDX-License-Identifier: Apache-2.0
"""Tests for LightspeedConnector."""

from open_lightspeed.connectors.lightspeed import LightspeedConnector
from sdg_hub.core.connectors.base import ConnectorConfig
from sdg_hub.core.connectors.exceptions import ConnectorError
from sdg_hub.core.connectors.registry import ConnectorRegistry
import pytest


class TestLightspeedConnector:
    """Test LightspeedConnector."""

    def test_registered_in_registry(self):
        """Test connector is registered."""
        assert ConnectorRegistry.get("lightspeed") == LightspeedConnector

    def test_build_headers_with_api_key(self):
        """Test Lightspeed uses Bearer Authorization header."""
        connector = LightspeedConnector(
            config=ConnectorConfig(url="http://test", api_key="secret")
        )
        headers = connector._build_headers()
        assert headers["Authorization"] == "Bearer secret"
        assert "x-api-key" not in headers

    def test_build_headers_without_api_key(self):
        """Test headers without API key."""
        connector = LightspeedConnector(config=ConnectorConfig(url="http://test"))
        headers = connector._build_headers()
        assert headers == {"Content-Type": "application/json"}

    def test_build_request(self):
        """Test request building extracts last user message."""
        connector = LightspeedConnector(config=ConnectorConfig(url="http://test"))

        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Investigate the CrashLoopBackOff"},
        ]
        request = connector.build_request(messages, "session-1")

        assert request == {
            "query": "Investigate the CrashLoopBackOff",
            "conversation_id": "session-1",
        }

    def test_build_request_single_message(self):
        """Test request with single user message."""
        connector = LightspeedConnector(config=ConnectorConfig(url="http://test"))

        messages = [{"role": "user", "content": "What pods are failing?"}]
        request = connector.build_request(messages, "s1")

        assert request["query"] == "What pods are failing?"
        assert request["conversation_id"] == "s1"

    def test_build_request_no_user_message(self):
        """Test error when no user message found."""
        connector = LightspeedConnector(config=ConnectorConfig(url="http://test"))

        with pytest.raises(ConnectorError, match="No user message"):
            connector.build_request([{"role": "system", "content": "hi"}], "s1")

    def test_parse_response(self):
        """Test response parsing returns raw dict."""
        connector = LightspeedConnector(config=ConnectorConfig(url="http://test"))

        response = {
            "response": "The pod is OOMKilled",
            "messages": [
                {"role": "user", "content": "What's wrong?"},
                {"role": "assistant", "content": "The pod is OOMKilled"},
            ],
            "conversation_id": "session-1",
        }
        assert connector.parse_response(response) == response

    def test_parse_response_invalid(self):
        """Test error on non-dict response."""
        connector = LightspeedConnector(config=ConnectorConfig(url="http://test"))

        with pytest.raises(ConnectorError, match="Expected dict"):
            connector.parse_response(["not", "a", "dict"])


class TestLightspeedConnectorExtraction:
    """Test LightspeedConnector response field extraction classmethods."""

    def test_extract_text(self):
        """Test extracting response text."""
        response = {"response": "The pod is OOMKilled", "messages": []}
        assert LightspeedConnector.extract_text(response) == "The pod is OOMKilled"

    def test_extract_text_missing(self):
        """Test extracting text when response key is missing."""
        assert LightspeedConnector.extract_text({"messages": []}) is None

    def test_extract_text_none(self):
        """Test extracting text when response value is None."""
        assert LightspeedConnector.extract_text({"response": None}) is None

    def test_extract_session_id(self):
        """Test extracting conversation_id as session_id."""
        response = {"response": "ok", "conversation_id": "conv-123"}
        assert LightspeedConnector.extract_session_id(response) == "conv-123"

    def test_extract_session_id_missing(self):
        """Test extracting session_id when key is missing."""
        assert LightspeedConnector.extract_session_id({"response": "ok"}) is None

    def test_extract_tool_trace(self):
        """Test extracting full messages list as tool trace."""
        messages = [
            {"role": "user", "content": "What's wrong?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "pods_list", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "pod-a: Running"},
            {"role": "assistant", "content": "All pods are running."},
        ]
        response = {"response": "All pods are running.", "messages": messages}
        trace = LightspeedConnector.extract_tool_trace(response)

        assert trace is not None
        assert len(trace) == 4
        assert trace[0]["role"] == "user"
        assert trace[1]["tool_calls"][0]["function"]["name"] == "pods_list"
        assert trace[2]["role"] == "tool"

    def test_extract_tool_trace_missing(self):
        """Test extracting tool trace when messages key is missing."""
        assert LightspeedConnector.extract_tool_trace({"response": "ok"}) is None

    def test_extract_tool_trace_none(self):
        """Test extracting tool trace when messages is None."""
        assert (
            LightspeedConnector.extract_tool_trace({"messages": None, "response": "ok"})
            is None
        )
