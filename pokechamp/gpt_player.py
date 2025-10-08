import json
import os
from time import sleep

from openai import APITimeoutError, OpenAI, RateLimitError


LEGACY_MODEL_PROVIDER_CONFIG = {
    "gpt-4o": {"model": "gpt-4o-latest"},
    "gpt-4o-mini": {"model": "gpt-4o-mini"},
    "gpt-4.1": {"model": "gpt-4.1"},
    "gpt-4.1-mini": {"model": "gpt-4.1-mini"},
    "gpt-5": {"model": "gpt-5", "reasoning": {"effort": "medium"}},
    "gpt-5-mini": {"model": "gpt-5-mini", "reasoning": {"effort": "low"}},
}


class GPTPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
        self._client = OpenAI(api_key=self.api_key)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.reasoning_tokens = 0

    def get_model_config(self, model_name: str) -> dict:
        if not model_name:
            return {"model": "gpt-4o-latest"}
        return LEGACY_MODEL_PROVIDER_CONFIG.get(model_name, {"model": model_name})

    def _build_actions_schema(self, actions=None):
        moves = []
        switches = []
        if actions and isinstance(actions, (list, tuple)):
            if len(actions) > 0 and isinstance(actions[0], (list, tuple)):
                moves = [m for m in actions[0] if isinstance(m, str)]
            if len(actions) > 1 and isinstance(actions[1], (list, tuple)):
                switches = [s for s in actions[1] if isinstance(s, str)]

        properties = {
            "thought": {"type": "string", "description": "Optional reasoning steps."},
            "move": {"type": "string", "description": "Name of the move to use."},
            "switch": {"type": "string", "description": "Name of the Pokémon to switch to."},
            "dynamax": {"type": "string", "description": "Move to use while Dynamaxing."},
            "terastallize": {"type": "string", "description": "Move to use when Terastallizing."},
            "decision": {
                "type": "object",
                "additionalProperties": True,
                "description": "Final decision selected from provided options.",
            },
            "option_1": {"type": "object", "additionalProperties": True},
            "option_2": {"type": "object", "additionalProperties": True},
            "option_3": {"type": "object", "additionalProperties": True},
        }

        if moves:
            properties["move"]["enum"] = moves
        if switches:
            properties["switch"]["enum"] = switches

        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": True,
        }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "battle_action",
                "schema": schema
            }
        }

    def _build_generic_json_schema(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "generic_json",
                "schema": {"type": "object", "additionalProperties": True}
            }
        }

    def _extract_response_output(self, response):
        text_output = getattr(response, "output_text", None)
        if text_output:
            return text_output

        output_items = getattr(response, "output", None)
        if output_items:
            for item in output_items:
                contents = getattr(item, "content", None)
                if not contents:
                    continue
                for content in contents:
                    if hasattr(content, "json") and content.json is not None:
                        return json.dumps(content.json)
                    if hasattr(content, "text") and content.text is not None:
                        return content.text
        return ""

    def _update_usage(self, response):
        usage = getattr(response, "usage", None)
        if not usage:
            return
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.reasoning_tokens += getattr(usage, "reasoning_tokens", 0) or 0

    def _create_request_payload(self, system_prompt, user_prompt, model_config, temperature, max_tokens, json_format, actions):
        if model_config is None:
            model_config = self.get_model_config(None)
        payload = {
            "model": model_config.get("model"),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if model_config.get("reasoning"):
            payload["reasoning"] = model_config["reasoning"]
        if json_format:
            payload["response_format"] = self._build_actions_schema(actions)

        return payload

    def get_LLM_action(self, system_prompt, user_prompt, model='gpt-4o', temperature=0.7, json_format=False, seed=None, stop=[],
 max_tokens=200, actions=None, model_config=None) -> str:
        resolved_config = model_config or self.get_model_config(model)
        request_payload = self._create_request_payload(system_prompt, user_prompt, resolved_config, temperature, max_tokens, json_format, actions)

        try:
            response = self._client.responses.create(**request_payload)
        except (RateLimitError, APITimeoutError):
            sleep(5)
            print('rate limit or timeout error')
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions)

        outputs = self._extract_response_output(response)
        self._update_usage(response)
    def _extract_response_text(self, response) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        response_dict = None
        if hasattr(response, "model_dump"):
            try:
                response_dict = response.model_dump()
            except Exception:
                response_dict = None
        if response_dict is None and hasattr(response, "to_dict"):
            try:
                response_dict = response.to_dict()
            except Exception:
                response_dict = None
        if isinstance(response_dict, dict):
            output = response_dict.get("output") or []
            chunks = []
            for block in output:
                contents = block.get("content") if isinstance(block, dict) else None
                if not isinstance(contents, list):
                    continue
                for item in contents:
                    item_type = item.get("type") if isinstance(item, dict) else None
                    text = item.get("text") if isinstance(item, dict) else None
                    if item_type in {"output_text", "text"} and text:
                        chunks.append(text)
            if chunks:
                return "".join(chunks)
            if "output_text" in response_dict and response_dict["output_text"]:
                return response_dict["output_text"]
        if hasattr(response, "choices"):
            choices = getattr(response, "choices")
            if choices:
                first = choices[0]
                message = getattr(first, "message", None)
                if message is not None and hasattr(message, "content"):
                    return message.content
        return ""

    def get_LLM_action(self, system_prompt, user_prompt, model='gpt-4o', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, response_schema=None, reasoning_effort=None) -> str:
        client = OpenAI(api_key=self.api_key)
        # client = AzureOpenAI()
        try:
            if json_format and response_schema:
                try:
                    response_kwargs = {
                        "model": model,
                        "input": [
                            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                        ],
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "battle_action",
                                "schema": response_schema,
                            },
                        },
                    }
                    if reasoning_effort:
                        response_kwargs["reasoning"] = {"effort": reasoning_effort}
                    response = client.responses.create(**response_kwargs)
                    outputs = self._extract_response_text(response)
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        self.completion_tokens += getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
                        self.prompt_tokens += getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
                    if outputs:
                        return outputs, True
                except Exception as exc:
                    print(f"Responses API fallback: {exc}")

            if json_format:
                response_format = {"type": "json_object"}
                if response_schema:
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "battle_action",
                            "schema": response_schema,
                        },
                    }
                response = client.chat.completions.create(
                    response_format=response_format,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=False,
                    # seed=seed,
                    stop=stop,
                    max_tokens=max_tokens
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=False,
                    stop=stop,
                    max_tokens=max_tokens
                )
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error')
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions)
        outputs = response.choices[0].message.content
        # log completion tokens
        if hasattr(response, "usage"):
            usage = response.usage
            self.completion_tokens += getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))
            self.prompt_tokens += getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
        if json_format:
            return outputs, True
        return outputs, False

    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gpt-4o', json_format=False, seed=None, stop=[],
 max_tokens=200, model_config=None):
        resolved_config = model_config or self.get_model_config(model)
        request_payload = self._create_request_payload(system_prompt, user_prompt, resolved_config, temperature, max_tokens, json_format, None)
        if json_format:
            request_payload["response_format"] = self._build_generic_json_schema()

        try:
            response = self._client.responses.create(**request_payload)
        except (RateLimitError, APITimeoutError):
            sleep(5)
            print('rate limit or timeout error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)

        message = self._extract_response_output(response)
        self._update_usage(response)

        if json_format:
            return message, True
        return message, False
