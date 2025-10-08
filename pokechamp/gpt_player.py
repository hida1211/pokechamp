from openai import OpenAI
from time import sleep
from openai import RateLimitError
import os

class GPTPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0

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
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gpt-4o', json_format=False, seed=None, stop=[], max_tokens=200):
        client = OpenAI(api_key=self.api_key)
        # client = AzureOpenAI()
        try:
            output_padding = ''
            if json_format:
                output_padding  = '\n{"'
                
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt+output_padding}
                ],
                temperature=temperature,
                stream=False,
                stop=stop,
                max_tokens=max_tokens
            )
            message = response.choices[0].message.content
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        
        if json_format:
            json_start = 0
            json_end = message.find('}') + 1 # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True
        return message, False
