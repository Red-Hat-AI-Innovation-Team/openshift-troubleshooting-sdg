"""Convert distillation output rows into function-calling SFT conversations."""

import json
from typing import Any


def _json_default(obj: Any) -> Any:
    """Serialize numpy values that pandas/parquet leave inside nested columns."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def format_training_messages(row) -> list[dict]:
    """Convert one distillation row into a function-calling conversation.

    The row needs ``question``, ``tool_list`` (MCP tool schemas) and
    ``agent_response`` (the Lightspeed response, with ``tool_calls``,
    ``tool_results`` and ``response``).
    """
    resp = row["agent_response"]
    tool_list = row["tool_list"]
    question = row["question"]

    tools_json = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in tool_list
        ],
        default=_json_default,
    )
    # Tool declarations are embedded in the system message with
    # <|im_system|>tool_declare<|im_middle|>...<|im_end|> markers; adapt this to
    # the chat template of the model being fine-tuned.
    messages = [
        {"role": "system", "content": f"<|im_system|>tool_declare<|im_middle|>{tools_json}<|im_end|>"},
        {"role": "user", "content": question},
    ]

    tool_calls = resp.get("tool_calls", [])
    tool_results = resp.get("tool_results", [])
    results_by_id = {tr["id"]: tr for tr in tool_results}

    for tc in tool_calls:
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": {
                "name": tc["name"],
                "arguments": json.dumps(tc.get("args", {}), default=_json_default),
            },
        })
        tr = results_by_id.get(tc["id"], {})
        messages.append({"role": "function", "content": tr.get("content", ""), "name": tc["name"]})

    messages.append({"role": "assistant", "content": resp.get("response", "")})
    return messages
