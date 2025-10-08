from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from google import genai


@dataclass(frozen=True)
class GeminiModelConfig:
    """Configuration metadata for a Gemini model alias."""

    api_model: str
    requires_reasoning: bool = False


@dataclass
class ReasoningConfig:
    """Provider specific knobs for Gemini reasoning models."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    safety_settings: Optional[Sequence[Any]] = None
    seed: Optional[int] = None
    thinking_budget: Optional[int] = None
    include_thoughts: Optional[bool] = None
    response_mime_type: Optional[str] = None


class GeminiPlayer:
    def __init__(self, api_key: str = "") -> None:
        print("api_key", api_key)
        if api_key == "":
            self.api_key = os.getenv("GEMINI_API_KEY")
        else:
            self.api_key = api_key

        # Configure the Gemini API
        self.client = genai.Client(api_key=self.api_key)

        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.last_thinking_trace: List[str] = []

        # Map common model names to official API names and reasoning requirements
        self.model_mapping: Dict[str, Union[GeminiModelConfig, str]] = {
            # Default to latest Gemini 2.5
            "gemini-flash": GeminiModelConfig("gemini-2.5-flash"),
            "gemini-flash-2.5": GeminiModelConfig("gemini-2.5-flash"),
            "gemini-pro": GeminiModelConfig("gemini-2.5-pro"),
            "gemini-pro-2.5": GeminiModelConfig("gemini-2.5-pro"),

            # Gemini 2.5 reasoning variants
            "gemini-2.5-pro-exp": GeminiModelConfig("gemini-2.5-pro-exp", requires_reasoning=True),
            "gemini-2.5-pro-experimental": GeminiModelConfig("gemini-2.5-pro-exp", requires_reasoning=True),
            "gemini-2.5-flash-thinking": GeminiModelConfig("gemini-2.5-flash-thinking", requires_reasoning=True),
            "gemini-flash-thinking": GeminiModelConfig("gemini-2.5-flash-thinking", requires_reasoning=True),

            # Gemini 2.0 models
            "gemini-2.0-flash": GeminiModelConfig("gemini-2.0-flash"),
            "gemini-2.0-flash-lite": GeminiModelConfig("gemini-2.0-flash-lite"),
            "gemini-2.0-pro": GeminiModelConfig("gemini-2.0-pro-experimental"),
            "gemini-2.0-pro-experimental": GeminiModelConfig("gemini-2.0-pro-experimental"),
            "gemini-2.0-flash-thinking": GeminiModelConfig(
                "gemini-2.0-flash-thinking-exp", requires_reasoning=True
            ),
            "gemini-2.0-flash-thinking-exp": GeminiModelConfig(
                "gemini-2.0-flash-thinking-exp", requires_reasoning=True
            ),

            # Gemini 1.5 models (legacy support)
            "gemini-1.5-flash": GeminiModelConfig("gemini-1.5-flash"),
            "gemini-1.5-pro": GeminiModelConfig("gemini-1.5-pro"),
        }

    def get_LLM_action(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        json_format: bool = False,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        max_tokens: int = 1000,
        actions: Optional[Sequence[Sequence[str]]] = None,
        reasoning_config: Optional[ReasoningConfig] = None,
    ) -> Tuple[str, bool]:
    def get_LLM_action(self, system_prompt, user_prompt, model='gemini-2.0-flash', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=1000, actions=None, response_schema=None, reasoning_effort=None) -> str:
        try:
            model_config = self._resolve_model_config(model)

            generate_config = self._build_generate_config(
                system_prompt=system_prompt,
                base_temperature=temperature,
                json_format=json_format,
                stop_sequences=stop,
                max_output_tokens=max_tokens,
                reasoning_config=reasoning_config,
                requires_reasoning=model_config.requires_reasoning,
                actions=actions,
                seed=seed,
            )

            response = self.client.models.generate_content(
                model=model_config.api_model,
                contents=user_prompt,
                config=generate_config,
            )

            outputs, was_json = self._extract_response_payload(response, json_format)
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            self._update_token_usage(response, combined_prompt, outputs)
            self.last_thinking_trace = self._extract_thinking_traces(response)

            return outputs, was_json

        except Exception as e:  # pragma: no cover - fail fast just like previous implementation
            print(f"Gemini API error: {e}")
            sys.exit(1)

    def get_LLM_query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str = "gemini-2.0-flash",
        json_format: bool = False,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        max_tokens: int = 1000,
        reasoning_config: Optional[ReasoningConfig] = None,
    ) -> Tuple[str, bool]:
        try:
            model_config = self._resolve_model_config(model)
            generate_config = self._build_generate_config(
                system_prompt=system_prompt,
                base_temperature=temperature,
                json_format=json_format,
                stop_sequences=stop,
                max_output_tokens=max_tokens,
                reasoning_config=reasoning_config,
                requires_reasoning=model_config.requires_reasoning,
                actions=None,
                seed=seed,
            )

            response = self.client.models.generate_content(
                model=model_config.api_model,
                contents=user_prompt,
                config=generate_config,
            )

            outputs, was_json = self._extract_response_payload(response, json_format)
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            self._update_token_usage(response, combined_prompt, outputs)
            self.last_thinking_trace = self._extract_thinking_traces(response)

            return outputs, was_json

        except Exception as e:  # pragma: no cover - fail fast just like previous implementation
            print(f"Gemini API error2: {e}")
            sys.exit(1)

    def _resolve_model_config(self, model: str) -> GeminiModelConfig:
        resolved = self.model_mapping.get(model)
        if resolved is None:
            return GeminiModelConfig(api_model=model)
        if isinstance(resolved, GeminiModelConfig):
            return resolved
        return GeminiModelConfig(api_model=resolved)

    def _build_generate_config(
        self,
        *,
        system_prompt: str,
        base_temperature: float,
        json_format: bool,
        stop_sequences: Optional[Sequence[str]],
        max_output_tokens: int,
        reasoning_config: Optional[ReasoningConfig],
        requires_reasoning: bool,
        actions: Optional[Sequence[Sequence[str]]],
        seed: Optional[int],
    ) -> genai.types.GenerateContentConfig:
        reasoning_config = reasoning_config or ReasoningConfig()
        stop_sequences = tuple(stop_sequences) if stop_sequences else None

        temperature = (
            reasoning_config.temperature
            if reasoning_config.temperature is not None
            else base_temperature
        )

        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_output_tokens,
        }

        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if reasoning_config.top_p is not None:
            config_kwargs["top_p"] = reasoning_config.top_p
        if reasoning_config.top_k is not None:
            config_kwargs["top_k"] = reasoning_config.top_k
        if reasoning_config.safety_settings is not None:
            config_kwargs["safety_settings"] = reasoning_config.safety_settings
        if seed is not None:
            config_kwargs["seed"] = seed
        if stop_sequences:
            config_kwargs["stop_sequences"] = list(stop_sequences)

        if json_format:
            config_kwargs["response_schema"] = self._build_action_schema(actions)
            config_kwargs["response_mime_type"] = (
                reasoning_config.response_mime_type or "application/json"
            )

        thinking_kwargs: Dict[str, Any] = {}
        include_thoughts = (
            reasoning_config.include_thoughts
            if reasoning_config.include_thoughts is not None
            else requires_reasoning
        )
        if include_thoughts:
            thinking_kwargs["include_thoughts"] = True
        if reasoning_config.thinking_budget is not None:
            thinking_kwargs["thinking_budget"] = reasoning_config.thinking_budget

        if thinking_kwargs:
            config_kwargs["thinking_config"] = genai.types.ThinkingConfig(**thinking_kwargs)

        return genai.types.GenerateContentConfig(**config_kwargs)

    def _build_action_schema(
        self, actions: Optional[Sequence[Sequence[str]]]
    ) -> genai.types.Schema:
        move_schema = genai.types.Schema(type="string")
        switch_schema = genai.types.Schema(type="string")

        if actions and len(actions) >= 1:
            moves = actions[0]
            if moves:
                move_schema = genai.types.Schema(
                    type="string", enum=[move for move in moves if move]
                )
        if actions and len(actions) >= 2:
            switches = actions[1]
            if switches:
                switch_schema = genai.types.Schema(
                    type="string", enum=[switch for switch in switches if switch]
                )

        properties = {
            "move": move_schema,
            "switch": switch_schema,
            "dynamax": genai.types.Schema(type="boolean"),
            "terastallize": genai.types.Schema(type="string"),
            "action": genai.types.Schema(type="string"),
            "target": genai.types.Schema(type="string"),
            "reason": genai.types.Schema(type="string"),
        }

        return genai.types.Schema(
            type="object",
            properties=properties,
            additional_properties=True,
        )

    def _extract_response_payload(
        self, response: genai.types.GenerateContentResponse, json_format: bool
    ) -> Tuple[str, bool]:
        if json_format:
            parsed = response.parsed
            if parsed is not None:
                if hasattr(parsed, "model_dump"):
                    try:
                        parsed_payload = parsed.model_dump()
                    except Exception:  # pragma: no cover - defensive fallback
                        parsed_payload = getattr(parsed, "__dict__", parsed)
                else:
                    parsed_payload = parsed
                return json.dumps(parsed_payload), True

            # Fallback to raw text parsing if schema parsing failed
            text_output = getattr(response, "text", "") or ""
            try:
                parsed_json = json.loads(text_output)
                return json.dumps(parsed_json), True
            except (TypeError, json.JSONDecodeError):
                return text_output, True

        text_output = getattr(response, "text", "") or ""
        return text_output, False

    def _extract_thinking_traces(
        self, response: genai.types.GenerateContentResponse
    ) -> List[str]:
        traces: List[str] = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            if not content or not content.parts:
                continue
            for part in content.parts:
                if getattr(part, "thought", None):
                    text = getattr(part, "text", None)
                    if text:
                        traces.append(text)
        return traces

    def _update_token_usage(
        self, response: genai.types.GenerateContentResponse, prompt: str, output: str
    ) -> None:
        usage = response.usage_metadata
        if usage:
            if usage.prompt_token_count is not None:
                self.prompt_tokens += usage.prompt_token_count
            if usage.candidates_token_count is not None:
                self.completion_tokens += usage.candidates_token_count
            if usage.thoughts_token_count is not None:
                self.completion_tokens += usage.thoughts_token_count
        else:
            self.prompt_tokens += len(prompt.split()) * 1.3
            self.completion_tokens += len(output.split()) * 1.3
