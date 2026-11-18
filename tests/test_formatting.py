"""Tests for the shared training-data formatter."""

import json

import pandas as pd

from open_lightspeed.formatting import format_training_messages

TOOLS = [
    {
        "name": "pods_list_in_namespace",
        "description": "List pods in a namespace",
        "inputSchema": {
            "type": "object",
            "properties": {"namespace": {"type": "string"}},
            "required": ["namespace"],
        },
    }
]

AGENT_RESPONSE = {
    "response": "app-0 is stuck because ConfigMap app-config does not exist.",
    "tool_calls": [{"id": "call_1", "name": "pods_list_in_namespace", "args": {"namespace": "demo"}}],
    "tool_results": [
        {"id": "call_1", "name": "pods_list_in_namespace", "status": "success", "content": "app-0  0/1  ContainerCreating"}
    ],
}


def test_formats_tool_schemas_loaded_from_parquet(tmp_path):
    """Tool schemas read back from parquet contain numpy arrays; formatting must still work."""
    path = tmp_path / "distillation_input.parquet"
    pd.DataFrame({"tool_list": [TOOLS]}).to_parquet(path, index=False)
    row = {
        "question": "Why is app-0 not ready?",
        "tool_list": pd.read_parquet(path)["tool_list"][0],
        "agent_response": AGENT_RESPONSE,
    }

    messages = format_training_messages(row)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "function", "assistant"]
    declared = json.loads(messages[0]["content"].split("<|im_middle|>")[1].removesuffix("<|im_end|>"))
    assert declared[0]["function"]["name"] == "pods_list_in_namespace"
    assert declared[0]["function"]["parameters"]["required"] == ["namespace"]
    assert messages[2]["function_call"]["name"] == "pods_list_in_namespace"
    assert json.loads(messages[2]["function_call"]["arguments"]) == {"namespace": "demo"}
    assert messages[3] == {"role": "function", "content": "app-0  0/1  ContainerCreating", "name": "pods_list_in_namespace"}
    assert messages[4]["content"] == AGENT_RESPONSE["response"]
