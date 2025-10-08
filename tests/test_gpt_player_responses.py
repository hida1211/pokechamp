import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pokechamp.gpt_player import GPTPlayer, LEGACY_MODEL_PROVIDER_CONFIG


class FakeResponse(SimpleNamespace):
    pass


class FakeContent(SimpleNamespace):
    pass


class FakeOutput(SimpleNamespace):
    pass


@pytest.fixture
def mock_responses_client():
    with patch("pokechamp.gpt_player.OpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        yield client


def test_responses_parsing_with_reasoning(mock_responses_client):
    usage = SimpleNamespace(prompt_tokens=21, completion_tokens=11, reasoning_tokens=5)
    reasoning_trace = [
        {"type": "chain_of_thought", "content": [{"type": "text", "text": "Step one"}]}
    ]
    fake_json = {"move": "thunderbolt"}
    response = FakeResponse(
        output=[FakeOutput(content=[FakeContent(json=fake_json)])],
        usage=usage,
        reasoning=reasoning_trace,
    )
    mock_responses_client.responses.create.return_value = response

    player = GPTPlayer(api_key="test-key")
    action, is_json = player.get_LLM_action(
        system_prompt="system",
        user_prompt="user",
        model="gpt-5",
        temperature=0.1,
        json_format=True,
        max_tokens=128,
        actions=[["thunderbolt"], ["pikachu"]],
    )

    assert json.loads(action) == fake_json
    assert is_json is True
    assert player.completion_tokens == usage.completion_tokens
    assert player.prompt_tokens == usage.prompt_tokens
    assert player.reasoning_tokens == usage.reasoning_tokens

    mock_responses_client.responses.create.assert_called_once()
    called_kwargs = mock_responses_client.responses.create.call_args.kwargs
    assert called_kwargs["model"] == LEGACY_MODEL_PROVIDER_CONFIG["gpt-5"]["model"]
    assert "response_format" in called_kwargs
    move_schema = called_kwargs["response_format"]["json_schema"]["schema"]["properties"]["move"]
    assert move_schema["enum"] == ["thunderbolt"]
    assert "reasoning" in called_kwargs
