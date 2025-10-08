import json
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from google import genai

from pokechamp.gemini_player import GeminiPlayer, ReasoningConfig


class _DummyModels:
    def __init__(self, response: Any):
        self._response = response
        self.kwargs: Optional[Dict[str, Any]] = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self._response


class _DummyClient:
    def __init__(self, response: Any):
        self.models = _DummyModels(response)


def _make_usage(prompt: int, candidates: int, thoughts: int = 0) -> genai.types.GenerateContentResponseUsageMetadata:
    return genai.types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
    )


def _make_thinking_response(payload: Dict[str, Any], thought: str) -> Any:
    usage = _make_usage(prompt=10, candidates=5, thoughts=3)
    parts = [
        SimpleNamespace(text=thought, thought=True),
        SimpleNamespace(text=json.dumps(payload), thought=None),
    ]
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(
        parsed=payload,
        text=json.dumps(payload),
        candidates=[candidate],
        usage_metadata=usage,
    )


def _make_text_response(text: str) -> Any:
    usage = _make_usage(prompt=2, candidates=4)
    parts = [SimpleNamespace(text=text, thought=None)]
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(parsed=None, text=text, candidates=[candidate], usage_metadata=usage)


@pytest.fixture
def monkeypatched_client(monkeypatch):
    created: Dict[str, Any] = {}

    def _factory(response: Any):
        dummy_client = _DummyClient(response)

        def _make_client(*args, **kwargs):
            created["client"] = dummy_client
            return dummy_client

        monkeypatch.setattr("pokechamp.gemini_player.genai.Client", _make_client)
        return created

    return _factory


def test_gemini_player_parses_thinking_response(monkeypatched_client):
    payload = {"move": "thunderbolt"}
    response = _make_thinking_response(payload, "Focus on electric STAB.")
    created = monkeypatched_client(response)

    player = GeminiPlayer(api_key="test-key")

    output, is_json = player.get_LLM_action(
        "system",
        "Pick the best move",
        model="gemini-2.5-flash-thinking",
        json_format=True,
    )

    assert is_json is True
    assert json.loads(output) == payload
    assert player.last_thinking_trace == ["Focus on electric STAB."]
    assert player.prompt_tokens == 10
    assert player.completion_tokens == 8

    config = created["client"].models.kwargs["config"]
    assert config.thinking_config.include_thoughts is True
    assert config.response_schema is not None
    assert config.response_mime_type == "application/json"


def test_reasoning_config_overrides_generation(monkeypatched_client):
    response = _make_text_response("{\"switch\": \"Pikachu\"}")
    created = monkeypatched_client(response)

    player = GeminiPlayer(api_key="test-key")

    safety = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
    reasoning_cfg = ReasoningConfig(
        temperature=0.0,
        top_p=0.2,
        top_k=1,
        safety_settings=safety,
        thinking_budget=32,
        include_thoughts=False,
        response_mime_type="application/json",
    )

    output, is_json = player.get_LLM_action(
        "system",
        "Choose wisely",
        model="gemini-2.5-pro-exp",
        temperature=0.9,
        json_format=True,
        stop=["STOP"],
        reasoning_config=reasoning_cfg,
    )

    assert is_json is True
    assert json.loads(output) == {"switch": "Pikachu"}

    config = created["client"].models.kwargs["config"]
    assert config.temperature == 0.0
    assert config.top_p == 0.2
    assert config.top_k == 1
    safety_setting = config.safety_settings[0]
    assert safety_setting.category == genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
    assert safety_setting.threshold == genai.types.HarmBlockThreshold.BLOCK_NONE
    assert config.stop_sequences == ["STOP"]
    assert config.thinking_config.thinking_budget == 32
    # include_thoughts explicitly disabled even though model defaults to reasoning
    assert config.thinking_config.include_thoughts is None
