"""Request normalization for native Anthropic Messages endpoints."""

from typing import Any


def normalize_system_messages(body: dict[str, Any]) -> None:
    """Move system messages into the top-level prompt in a serialized body.

    Preserve existing system blocks and append message text in encounter order.
    As in OpenAI conversion, only text blocks contribute to a system prompt.
    The original message and content lists are not modified.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    system_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    if not system_messages:
        return

    system = body.get("system")
    blocks = list(system) if isinstance(system, list) else []
    if isinstance(system, str):
        blocks.append({"type": "text", "text": system})

    for message in system_messages:
        content = message.get("content")
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            blocks.extend(
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

    body["system"] = blocks
    body["messages"] = [
        message
        for message in messages
        if not (isinstance(message, dict) and message.get("role") == "system")
    ]
