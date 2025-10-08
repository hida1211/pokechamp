import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Provide a lightweight stub for the anthropic SDK so the module can be imported.
if "anthropic" not in sys.modules:
    anthropic_stub = types.ModuleType("anthropic")

    class _FakeAnthropicClient:
        def __init__(self, *args, **kwargs):
            self.responses = SimpleNamespace(create=MagicMock())
            self.messages = SimpleNamespace(create=MagicMock())
            anthropic_stub.last_client = self

    anthropic_stub.Anthropic = _FakeAnthropicClient
    sys.modules["anthropic"] = anthropic_stub
else:
    anthropic_stub = sys.modules["anthropic"]

from pokechamp.claude_player import ClaudePlayer


@pytest.fixture
def fake_client():
    # Instantiate a new client for each test to reset mocks
    player = ClaudePlayer(api_key="test-key")
    client = anthropic_stub.last_client
    client.responses.create.reset_mock()
    client.messages.create.reset_mock()
    return player, client


def test_claude_player_enforces_schema_and_extracts_action(fake_client):
    player, client = fake_client
    fake_response = SimpleNamespace(
        output=[
            {"type": "thinking", "content": [{"type": "text", "text": "considering options"}]},
            {"type": "output_json", "content": [{"type": "json", "json": {"move": "tackle"}}]},
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, thinking_tokens=3),
        output_text="irrelevant",
    )
    client.responses.create.return_value = fake_response

    output, is_json = player.get_LLM_action(
        system_prompt="system",
        user_prompt="user",
        model="anthropic/claude-3.5-sonnet",
        json_format=True,
        actions=[["tackle"], []],
    )

    assert output == json.dumps({"move": "tackle"}, separators=(",", ":"))
    assert is_json is True
    assert player.prompt_tokens == 11
    assert player.completion_tokens == 7
    assert player.output_tokens == 7
    assert player.thinking_tokens == 3
    assert "considering options" in player.reasoning_traces[-1]

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["output_json_schema"]["json_schema"]["properties"]["move"]["enum"] == ["tackle"]
    assert kwargs["thinking"]["type"] == "enabled"


def test_claude_player_returns_text_when_no_schema(fake_client):
    player, client = fake_client
    fake_response = SimpleNamespace(
        output=[
            {"type": "output_text", "content": [{"type": "text", "text": "battle summary"}]},
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=4, thinking_tokens=0),
        output_text="battle summary",
    )
    client.responses.create.return_value = fake_response

    output, is_json = player.get_LLM_action(
        system_prompt="system",
        user_prompt="user",
        model="anthropic/claude-2",
        json_format=False,
    )

    assert "battle summary" in output
    assert is_json is False
    kwargs = client.responses.create.call_args.kwargs
    assert "output_json_schema" not in kwargs
