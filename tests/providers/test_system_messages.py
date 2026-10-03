"""System-role compatibility at native Anthropic provider boundaries."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from api.models.anthropic import MessagesRequest
from core.anthropic.request import normalize_system_messages
from providers.llamacpp import LlamaCppProvider
from providers.lmstudio import LMStudioProvider
from providers.open_router import OpenRouterProvider


@pytest.fixture(params=[LMStudioProvider, LlamaCppProvider, OpenRouterProvider])
def native_provider(request, provider_config):
    with patch("httpx.AsyncClient"):
        return request.param(provider_config)


@pytest.mark.parametrize(
    "system",
    [None, "Existing", [{"type": "text", "text": "Existing"}]],
)
@pytest.mark.parametrize(
    "content",
    ["First", [{"type": "text", "text": "First"}]],
)
def test_native_providers_move_system_messages(native_provider, system, content):
    request = MessagesRequest.model_validate(
        {
            "model": "test-model",
            "system": system,
            "messages": [
                {"role": "system", "content": content},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Second"},
                        {"type": "text", "text": "Third"},
                    ],
                },
                {"role": "user", "content": "Continue"},
            ],
        }
    )
    original = request.model_dump()

    body = native_provider._build_request_body(request)

    parts = (["Existing"] if system is not None else []) + ["First", "Second", "Third"]
    expected = (
        "\n\n".join(parts)
        if isinstance(native_provider, OpenRouterProvider)
        else [{"type": "text", "text": part} for part in parts]
    )
    assert body["system"] == expected
    assert body["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Continue"},
    ]
    assert request.model_dump() == original


@pytest.mark.parametrize("system", [None, "Existing", []])
def test_requests_without_system_messages_keep_their_prompt(native_provider, system):
    request = MessagesRequest.model_validate(
        {
            "model": "test-model",
            "system": system,
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = native_provider._build_request_body(request)

    assert body.get("system") == system
    assert body["messages"] == [{"role": "user", "content": "Hello"}]


def test_normalization_preserves_block_metadata_without_mutating_source():
    source = {
        "system": [
            {"type": "text", "text": "Existing", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Instructions"},
                    {"type": "image", "source": {"type": "url", "url": "test"}},
                ],
            },
            {"role": "user", "content": "Hello"},
        ],
    }
    original = deepcopy(source)
    body = dict(source)

    normalize_system_messages(body)

    assert body["system"] == [
        original["system"][0],
        {"type": "text", "text": "Instructions"},
    ]
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert source == original


@pytest.mark.parametrize("content", ["", []])
def test_empty_system_messages_are_removed(native_provider, content):
    request = MessagesRequest.model_validate(
        {
            "model": "test-model",
            "system": "Existing",
            "messages": [
                {"role": "system", "content": content},
                {"role": "user", "content": "Hello"},
            ],
        }
    )

    body = native_provider._build_request_body(request)

    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    if isinstance(native_provider, OpenRouterProvider):
        assert body["system"] == "Existing"
    else:
        assert body["system"][0] == {"type": "text", "text": "Existing"}


def test_openrouter_normalizes_extra_body_overrides(provider_config):
    with patch("httpx.AsyncClient"):
        provider = OpenRouterProvider(provider_config)
    request = MessagesRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Original"}],
            "extra_body": {
                "system": "Existing",
                "messages": [
                    {"role": "system", "content": "Instructions"},
                    {"role": "user", "content": "Override"},
                ],
            },
        }
    )
    original = request.model_dump()

    body = provider._build_request_body(request)

    assert body["system"] == "Existing\n\nInstructions"
    assert body["messages"] == [{"role": "user", "content": "Override"}]
    assert request.model_dump() == original
