import json
import pytest

from pokechamp.llm_player import LLMPlayer


class _DummyAccount:
    username = ""


class _DummyClient:
    account_configuration = _DummyAccount()


class _DummyGen:
    gen = 9


def _make_player(backend: str) -> LLMPlayer:
    player = object.__new__(LLMPlayer)
    player.backend = backend
    player.ps_client = _DummyClient()
    player.gen = _DummyGen()
    player.reasoning_provider = player._detect_reasoning_provider(backend)
    player.reasoning_config = player._build_reasoning_config(player.reasoning_provider, backend)
    player.reasoning_telemetry = {}
    player._latest_reasoning_requests = {}
    return player


STATE_PROMPT = "Battle state summary.\n"
STATE_ACTION_PROMPT = "Available actions listed here.\n"
SYSTEM_PROMPT = "You are a battle assistant."


TEST_CASES = [
    (
        "openai",
        "gpt-4o-mini",
        ["move", "switch"],
        "action",
        {"move": "thunderbolt"},
    ),
    (
        "anthropic",
        "anthropic/claude-3-sonnet",
        ["move"],
        "reasoned_action",
        {"thought": "Consider type advantage.", "move": "thunderbolt", "usage": {"total_tokens": 88}},
    ),
    (
        "google",
        "gemini-pro",
        ["move"],
        "action",
        {"move": "thunderbolt", "safety": {"verdict": "safe"}},
    ),
    (
        "ollama",
        "ollama/llama3",
        ["switch"],
        "action",
        {"switch": "charizard", "analysis": "Use charizard to absorb hits."},
    ),
]


EXPECTED_SNAPSHOTS = {
    "openai": {
        "system_prompt": "Return only the JSON object requested.\n\nProvider openai must respond with the exact JSON schema described.\n\nYou are a battle assistant.",
        "user_prompt": "Battle state summary.\nAvailable actions listed here.\nChoose the best action for the current battle state. Respond with JSON matching: {\"move\": \"<move_name>\"} or {\"switch\": \"<switch_pokemon_name>\"}\n",
        "response": {"move": "thunderbolt"},
        "telemetry": {},
        "json_schema": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"move": {"type": "string"}},
                    "required": ["move"],
                    "additionalProperties": True,
                },
                {
                    "type": "object",
                    "properties": {"switch": {"type": "string"}},
                    "required": ["switch"],
                    "additionalProperties": True,
                },
            ]
        },
    },
    "anthropic": {
        "system_prompt": "Return only the JSON object requested.\n\nInclude a concise \"thought\" field summarizing your reasoning before choosing an action.\n\nYou are a battle assistant.",
        "user_prompt": "Battle state summary.\nAvailable actions listed here.\nChoose the best action and include a brief justification in a \"thought\" field (max 3 sentences). Respond with JSON matching: {\"thought\": \"<brief_reasoning>\", \"move\": \"<move_name>\"}\n\nProvide the \"thought\" field in fewer than three sentences.",
        "response": {"thought": "Consider type advantage.", "move": "thunderbolt", "usage": {"total_tokens": 88}},
        "telemetry": {"thought": "Consider type advantage.", "token_usage": 88},
        "json_schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "move": {"type": "string"},
            },
            "required": ["thought", "move"],
            "additionalProperties": True,
        },
    },
    "google": {
        "system_prompt": "Return only the JSON object requested.\n\nFollow Google's safety policies and respond only with the requested JSON.\n\nYou are a battle assistant.",
        "user_prompt": "Battle state summary.\nAvailable actions listed here.\nChoose the best action for the current battle state. Respond with JSON matching: {\"move\": \"<move_name>\"}\n",
        "response": {"move": "thunderbolt", "safety": {"verdict": "safe"}},
        "telemetry": {"safety": {"verdict": "safe"}},
        "json_schema": {
            "type": "object",
            "properties": {"move": {"type": "string"}},
            "required": ["move"],
            "additionalProperties": True,
        },
    },
    "ollama": {
        "system_prompt": "Return only the JSON object requested.\n\nKeep the JSON compact and deterministic.\n\nYou are a battle assistant.",
        "user_prompt": "Battle state summary.\nAvailable actions listed here.\nChoose the best action for the current battle state. Respond with JSON matching: {\"switch\": \"<switch_pokemon_name>\"}\n",
        "response": {"switch": "charizard", "analysis": "Use charizard to absorb hits."},
        "telemetry": {"thought": "Use charizard to absorb hits."},
        "json_schema": {
            "type": "object",
            "properties": {"switch": {"type": "string"}},
            "required": ["switch"],
            "additionalProperties": True,
        },
    },
}


@pytest.mark.parametrize("provider, backend, actions, schema_key, response", TEST_CASES)
def test_reasoning_prompt_snapshots(provider, backend, actions, schema_key, response):
    player = _make_player(backend)

    constraint = player.format_reasoning_request(schema_key, actions)
    request = player._latest_reasoning_requests.get(schema_key)
    user_prompt = STATE_PROMPT + STATE_ACTION_PROMPT + constraint
    system_prompt, user_prompt = player.build_reasoning_prompt(SYSTEM_PROMPT, user_prompt)

    assert "let's think step by step" not in user_prompt.lower()
    assert constraint.strip().endswith("}")

    if isinstance(response, dict):
        player._capture_reasoning_metadata(response)

    snapshot = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "telemetry": player.reasoning_telemetry,
        "json_schema": request.json_schema if request else None,
    }

    assert snapshot == EXPECTED_SNAPSHOTS[provider]
