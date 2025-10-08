"""Reasoning configuration utilities for LLM integrations.

This module centralizes the JSON schema used to enforce
structured outputs from reasoning-capable language models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _unique_sequence(values: Optional[Sequence[str]]) -> List[str]:
    """Return a list with duplicates and falsy values removed while preserving order."""
    seen = set()
    result: List[str] = []
    if not values:
        return result
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass
class ReasoningConfig:
    """Shared configuration for reasoning-focused model integrations."""

    thinking_budget_tokens: int = 1024
    tool_name: str = "choose_battle_action"
    tool_description: str = (
        "Select the next action for the Pokémon battle agent. Choose either a move "
        "to execute or a Pokémon to switch into."
    )
    include_thought_field: bool = True
    allow_additional_properties: bool = False

    def build_action_schema(self, actions: Optional[Sequence[Iterable[str]]] = None) -> Dict[str, Any]:
        """Create a JSON schema describing the valid battle action payload."""
        moves: List[str] = []
        switches: List[str] = []
        if actions:
            if len(actions) > 0:
                moves = _unique_sequence(actions[0])
            if len(actions) > 1:
                switches = _unique_sequence(actions[1])

        properties: Dict[str, Any] = {}
        required_choices: List[Dict[str, Any]] = []

        if moves:
            move_schema = {"type": "string", "enum": moves}
            properties["move"] = move_schema
            # Gimmick actions reuse the move list for validation
            properties["dynamax"] = move_schema
            properties["terastallize"] = move_schema
            required_choices.extend(
                [{"required": ["move"]}, {"required": ["dynamax"]}, {"required": ["terastallize"]}]
            )
        else:
            # When no moves are available we keep gimmick keys optional for completeness.
            properties["dynamax"] = {"type": "string"}
            properties["terastallize"] = {"type": "string"}

        if switches:
            properties["switch"] = {"type": "string", "enum": switches}
            required_choices.append({"required": ["switch"]})
        else:
            properties["switch"] = {"type": "string"}

        if self.include_thought_field:
            properties["thought"] = {"type": "string"}

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": self.allow_additional_properties,
        }

        choice_constraints = [rule for rule in required_choices if rule.get("required")]
        if choice_constraints:
            schema["anyOf"] = choice_constraints

        return schema

    def build_output_schema(self, actions: Optional[Sequence[Iterable[str]]] = None) -> Dict[str, Any]:
        """Return an Anthropic-compatible ``output_json_schema`` payload."""
        return {"type": "json_schema", "json_schema": self.build_action_schema(actions)}

    def build_tool_schema(self, actions: Optional[Sequence[Iterable[str]]] = None) -> Dict[str, Any]:
        """Return an Anthropic tool schema for structured output enforcement."""
        return {
            "type": "function",
            "name": self.tool_name,
            "description": self.tool_description,
            "input_schema": self.build_action_schema(actions),
        }
