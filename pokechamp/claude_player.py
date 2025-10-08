"""Anthropic Claude integration for PokéChamp."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

from pokechamp.reasoning_config import ReasoningConfig


class ClaudePlayer:
    """Wrapper around the Anthropic SDK providing PokéChamp's interface."""

    def __init__(self, api_key: str = "") -> None:
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for Claude support. Install it via "
                "`pip install anthropic` or ensure requirements.txt has been installed."
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=self.api_key)  # type: ignore[arg-type]

        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.reasoning_traces: List[str] = []

        self.reasoning_config = ReasoningConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_LLM_action(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-3.5-sonnet",
        temperature: float = 0.7,
        json_format: bool = False,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        max_tokens: int = 200,
        actions: Optional[Sequence[Iterable[str]]] = None,
    ) -> Tuple[str, bool]:
        """Return the model output, enforcing JSON when requested."""

        response, parser = self._invoke_client(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            stop=stop,
            max_tokens=max_tokens,
            json_format=json_format,
            actions=actions,
            seed=seed,
        )

        self._record_usage(response)
        output, is_json = parser(response, json_format=json_format)
        return output, is_json

    def get_LLM_query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str = "claude-3.5-sonnet",
        json_format: bool = False,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        max_tokens: int = 200,
    ) -> Tuple[str, bool]:
        response, parser = self._invoke_client(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            stop=stop,
            max_tokens=max_tokens,
            json_format=json_format,
            actions=None,
            seed=seed,
        )

        self._record_usage(response)
        output, is_json = parser(response, json_format=json_format)
        return output, is_json

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------
    def _invoke_client(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        stop: Optional[Sequence[str]],
        max_tokens: int,
        json_format: bool,
        actions: Optional[Sequence[Iterable[str]]],
        seed: Optional[int],
    ) -> Tuple[Any, Any]:
        normalized_model = self._normalize_model(model)
        thinking = self._build_thinking_config(normalized_model)

        request_args: Dict[str, Any] = {
            "model": normalized_model,
            "system": system_prompt or None,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if stop:
            request_args["stop_sequences"] = list(stop)
        if thinking:
            request_args["thinking"] = thinking
        if seed is not None:
            request_args["metadata"] = {"seed": seed}

        parser = self._parse_responses_output
        if hasattr(self.client, "responses"):
            if json_format and actions is not None:
                request_args["output_json_schema"] = self.reasoning_config.build_output_schema(actions)
            response = self.client.responses.create(**request_args)
        else:  # pragma: no cover - fallback for legacy SDKs
            parser = self._parse_messages_output
            request_args = self._convert_to_messages_request(request_args, actions, json_format)
            response = self.client.messages.create(**request_args)

        return response, parser

    def _convert_to_messages_request(
        self,
        request_args: Dict[str, Any],
        actions: Optional[Sequence[Iterable[str]]],
        json_format: bool,
    ) -> Dict[str, Any]:
        messages_request = {
            "model": request_args["model"],
            "system": request_args.get("system"),
            "messages": request_args["input"],
            "temperature": request_args.get("temperature"),
            "max_output_tokens": request_args.get("max_output_tokens"),
        }
        if "stop_sequences" in request_args:
            messages_request["stop_sequences"] = request_args["stop_sequences"]
        if "thinking" in request_args:
            messages_request["thinking"] = request_args["thinking"]
        if json_format and actions is not None:
            tool_schema = self.reasoning_config.build_tool_schema(actions)
            messages_request["tools"] = [tool_schema]
            messages_request["tool_choice"] = {"type": "tool", "name": tool_schema["name"]}
        return messages_request

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------
    def _parse_responses_output(self, response: Any, json_format: bool) -> Tuple[str, bool]:
        json_payload: Optional[Dict[str, Any]] = None
        text_parts: List[str] = []

        for block in self._iter_blocks(getattr(response, "output", None)):
            block_type = block.get("type")
            if block_type == "thinking":
                reasoning_text = self._extract_text(block.get("content"))
                if reasoning_text:
                    self.reasoning_traces.append(reasoning_text)
            elif block_type == "output_json":
                json_payload = self._extract_json(block.get("content"))
            else:
                text = self._extract_text(block.get("content"))
                if text:
                    text_parts.append(text)

        if json_format and json_payload is not None:
            return self._encode_json(json_payload), True

        fallback = getattr(response, "output_text", None)
        if fallback:
            text_parts.append(fallback)

        return "\n".join([part for part in text_parts if part]), json_format and json_payload is not None

    def _parse_messages_output(self, response: Any, json_format: bool) -> Tuple[str, bool]:
        json_payload: Optional[Dict[str, Any]] = None
        text_parts: List[str] = []

        for block in self._iter_blocks(getattr(response, "content", None)):
            block_type = block.get("type")
            if block_type == "thinking":
                reasoning_text = self._extract_text(block.get("content"))
                if reasoning_text:
                    self.reasoning_traces.append(reasoning_text)
            elif block_type == "tool_use":
                json_payload = block.get("input")
            else:
                text = self._extract_text(block.get("content"))
                if text:
                    text_parts.append(text)

        if json_format and json_payload is not None:
            return self._encode_json(json_payload), True

        return "\n".join([part for part in text_parts if part]), json_format and json_payload is not None

    def _iter_blocks(self, blocks: Optional[Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        if not blocks:
            return []
        normalized: List[Dict[str, Any]] = []
        for block in blocks:
            if hasattr(block, "to_dict"):
                normalized.append(block.to_dict())  # type: ignore[call-arg]
            elif isinstance(block, dict):
                normalized.append(block)
            else:
                normalized.append(getattr(block, "__dict__", {"type": "text", "content": str(block)}))
        return normalized

    def _extract_text(self, content: Any) -> str:
        if not content:
            return ""
        if isinstance(content, str):
            return content
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "thinking":
                    texts.append(self._extract_text(item.get("content")))
        return "\n".join([text for text in texts if text])

    def _extract_json(self, content: Any) -> Optional[Dict[str, Any]]:
        if not content:
            return None
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "json":
                    return item.get("json")
        return None

    def _encode_json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        thinking_tokens = getattr(usage, "thinking_tokens", 0)
        self.prompt_tokens += input_tokens
        self.completion_tokens += output_tokens
        self.output_tokens += output_tokens
        self.thinking_tokens += thinking_tokens

    def _normalize_model(self, model: str) -> str:
        if model.startswith("anthropic/"):
            return model.split("/", 1)[1]
        return model

    def _build_thinking_config(self, model: str) -> Optional[Dict[str, Any]]:
        supported_prefixes = ("claude-3.5-sonnet", "claude-4.5")
        if any(model.startswith(prefix) for prefix in supported_prefixes):
            return {"type": "enabled", "budget_tokens": self.reasoning_config.thinking_budget_tokens}
        return None
