import ast
from copy import copy, deepcopy
import datetime
import json
import os
import random
import sys

from dataclasses import dataclass, field

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import Player, BattleOrder
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from poke_env.environment.move import Move
import time
import json
from poke_env.data.gen_data import GenData
from pokechamp.gpt_player import GPTPlayer
from pokechamp.llama_player import LLAMAPlayer
from pokechamp.openrouter_player import OpenRouterPlayer
from pokechamp.gemini_player import GeminiPlayer
from pokechamp.ollama_player import OllamaPlayer
from pokechamp.data_cache import (
    get_cached_move_effect,
    get_cached_pokemon_move_dict,
    get_cached_ability_effect,
    get_cached_pokemon_ability_dict,
    get_cached_item_effect,
    get_cached_pokemon_item_dict,
    get_cached_pokedex
)
from pokechamp.minimax_optimizer import (
    get_minimax_optimizer,
    initialize_minimax_optimization,
    fast_battle_evaluation,
    create_battle_state_hash,
    OptimizedSimNode
)
from poke_env.player.local_simulation import LocalSim, SimNode
from difflib import get_close_matches
from pokechamp.prompts import get_number_turns_faint, get_status_num_turns_fnt, state_translate, get_gimmick_motivation

DEBUG=False


@dataclass
class ReasoningConfig:
    schema_templates: Dict[str, Dict[str, Any]]
    reasoning: bool = False
    system_messages: List[str] = field(default_factory=list)
    reasoning_prompt: str = ""
    metadata_hints: Dict[str, List[str]] = field(default_factory=dict)
    reasoning_effort: Optional[str] = None


@dataclass
class ReasoningRequest:
    prompt: str
    json_schema: Optional[Dict[str, Any]] = None


class LLMPlayer(Player):
    def __init__(self,
                 battle_format,
                 api_key="",
                 backend="gpt-4-1106-preview",
                 temperature=1.0,
                 prompt_algo="io",
                 log_dir=None,
                 team=None,
                 save_replays=None,
                 account_configuration=None,
                 server_configuration=None,
                 K=2,
                 _use_strat_prompt=False,
                 prompt_translate: Callable=state_translate,
                 device=0,
                 llm_backend=None
                 ):

        super().__init__(battle_format=battle_format,
                         team=team,
                         save_replays=save_replays,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action : Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.backend = backend
        self.temperature = temperature
        self.log_dir = log_dir
        self.api_key = api_key
        self.prompt_algo = prompt_algo
        self.gen = GenData.from_format(battle_format)
        self.genNum = self.gen.gen
        self.prompt_translate = prompt_translate

        self.strategy_prompt = ""
        self.team_str = team
        self.use_strat_prompt = _use_strat_prompt
        
        # Use cached data instead of loading files repeatedly
        self.move_effect = get_cached_move_effect()
        # only used in old prompting method, replaced by statistcal sets data
        self.pokemon_move_dict = get_cached_pokemon_move_dict()
        self.ability_effect = get_cached_ability_effect()
        # only used is old prompting method
        self.pokemon_ability_dict = get_cached_pokemon_ability_dict()
        self.item_effect = get_cached_item_effect()
        # unused
        # with open(f"./poke_env/data/static/items/gen8pokemon_item_dict.json", "r") as f:
        #     self.pokemon_item_dict = json.load(f)
        self.pokemon_item_dict = get_cached_pokemon_item_dict()
        self._pokemon_dict = get_cached_pokedex(self.gen.gen)

        self.last_plan = ""

        if llm_backend is None:
            print(f"Initializing backend: {backend}")  # Debug logging
            if backend.startswith('ollama/'):
                # Ollama models - extract model name after 'ollama/'
                model_name = backend.replace('ollama/', '')
                print(f"Using Ollama with model: {model_name}")
                self.llm = OllamaPlayer(model=model_name, device=device)
            elif 'gpt' in backend and not backend.startswith('openai/'):
                self.llm = GPTPlayer(self.api_key)
            elif 'llama' == backend:
                self.llm = LLAMAPlayer(device=device)
            elif 'gemini' in backend:
                self.llm = GeminiPlayer(self.api_key)
            elif backend.startswith(('openai/', 'anthropic/', 'google/', 'meta/', 'mistral/', 'cohere/', 'perplexity/', 'deepseek/', 'microsoft/', 'nvidia/', 'huggingface/', 'together/', 'replicate/', 'fireworks/', 'localai/', 'vllm/', 'sagemaker/', 'vertex/', 'bedrock/', 'azure/', 'custom/')):
                # OpenRouter supports hundreds of models from various providers
                self.llm = OpenRouterPlayer(self.api_key)
            else:
                raise NotImplementedError('LLM type not implemented:', backend)
        else:
            self.llm = llm_backend
        self.llm_value = self.llm
        self.reasoning_provider = self._detect_reasoning_provider(self.backend)
        self.reasoning_config = self._build_reasoning_config(self.reasoning_provider, self.backend)
        self.reasoning_telemetry: Dict[str, Any] = {}
        self._latest_reasoning_requests: Dict[str, ReasoningRequest] = {}
        self.K = K      # for minimax, SC, ToT
        self.use_optimized_minimax = True  # Enable optimized minimax by default
        self._minimax_initialized = False
        # Configuration for time optimization
        self.use_damage_calc_early_exit = True  # Use damage calculator to exit early when advantageous
        self.use_llm_value_function = True  # Use LLM for leaf node evaluation (vs fast heuristic)
        self.max_depth_for_llm_eval = 2  # Only use LLM evaluation for shallow depths to save time

    @staticmethod
    def _detect_reasoning_provider(backend: str) -> str:
        if not backend:
            return "default"
        backend_lower = backend.lower()
        if backend_lower.startswith("ollama/"):
            return "ollama"
        if backend_lower.startswith("anthropic/") or "claude" in backend_lower:
            return "anthropic"
        if backend_lower.startswith("google/") or backend_lower.startswith("gemini") or "gemini" in backend_lower:
            return "google"
        if backend_lower.startswith("openai/") or backend_lower.startswith("oai/") or backend_lower.startswith("gpt") or "gpt" in backend_lower:
            return "openai"
        if "/" in backend_lower:
            return backend_lower.split("/", 1)[0]
        return backend_lower.split("-", 1)[0] or "default"

    @staticmethod
    def _base_schema_templates() -> Dict[str, Dict[str, Any]]:
        action_schemas = {
            "move": {
                "example": {"move": "<move_name>"},
                "json_schema": {
                    "type": "object",
                    "properties": {"move": {"type": "string"}},
                    "required": ["move"],
                    "additionalProperties": True,
                },
            },
            "switch": {
                "example": {"switch": "<switch_pokemon_name>"},
                "json_schema": {
                    "type": "object",
                    "properties": {"switch": {"type": "string"}},
                    "required": ["switch"],
                    "additionalProperties": True,
                },
            },
            "dynamax": {
                "example": {"dynamax": "<move_name>"},
                "json_schema": {
                    "type": "object",
                    "properties": {"dynamax": {"type": "string"}},
                    "required": ["dynamax"],
                    "additionalProperties": True,
                },
            },
            "terastallize": {
                "example": {"terastallize": "<move_name>"},
                "json_schema": {
                    "type": "object",
                    "properties": {"terastallize": {"type": "string"}},
                    "required": ["terastallize"],
                    "additionalProperties": True,
                },
            },
        }
        reasoned_schemas = {
            "move": {
                "example": {"thought": "<brief_reasoning>", "move": "<move_name>"},
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
            "switch": {
                "example": {"thought": "<brief_reasoning>", "switch": "<switch_pokemon_name>"},
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "switch": {"type": "string"},
                    },
                    "required": ["thought", "switch"],
                    "additionalProperties": True,
                },
            },
            "dynamax": {
                "example": {"thought": "<brief_reasoning>", "dynamax": "<move_name>"},
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "dynamax": {"type": "string"},
                    },
                    "required": ["thought", "dynamax"],
                    "additionalProperties": True,
                },
            },
            "terastallize": {
                "example": {"thought": "<brief_reasoning>", "terastallize": "<move_name>"},
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "terastallize": {"type": "string"},
                    },
                    "required": ["thought", "terastallize"],
                    "additionalProperties": True,
                },
            },
        }
        return {
            "action": {
                "instruction": "Choose the best action for the current battle state.",
                "output_prefix": "Respond with JSON matching: ",
                "schemas": action_schemas,
            },
            "reasoned_action": {
                "instruction": "Choose the best action and include a brief justification in a \"thought\" field (max 3 sentences).",
                "output_prefix": "Respond with JSON matching: ",
                "schemas": reasoned_schemas,
            },
            "options": {
                "instruction": "Generate up to {max_k} candidate actions (k<={max_k}) ranked from best to worst.",
                "output_prefix": "Respond with JSON matching: ",
                "max_k": 3,
            },
            "decision": {
                "instruction": "Select the best option from the following choices by considering their consequences: [OPTIONS].",
                "output_prefix": "Respond with JSON matching: ",
                "default_action": "move",
            },
        }

    def _build_reasoning_config(self, provider: str, backend: Optional[str] = None) -> ReasoningConfig:
        templates = deepcopy(self._base_schema_templates())
        metadata_hints: Dict[str, List[str]] = {
            "thought": ["thought", "analysis", "reasoning"],
            "token_usage": ["usage.total_tokens", "usage.totalTokens", "token_usage.total"],
        }
        system_messages = ["Return only the JSON object requested."]
        reasoning_prompt = ""
        reasoning_enabled = False
        reasoning_effort: Optional[str] = None
        provider_lower = provider.lower() if provider else "default"
        backend_lower = backend.lower() if backend else ""

        if provider_lower == "anthropic":
            reasoning_enabled = True
            system_messages.append("Include a concise \"thought\" field summarizing your reasoning before choosing an action.")
            reasoning_prompt = "\nProvide the \"thought\" field in fewer than three sentences."
            metadata_hints["thinking"] = ["thinking", "claude_reasoning"]
        elif provider_lower == "openai":
            if "gpt-5" in backend_lower:
                reasoning_effort = "medium"
            system_messages.append("Provider openai must respond with the exact JSON schema described.")
        elif provider_lower in {"google", "gemini"}:
            system_messages.append("Follow Google's safety policies and respond only with the requested JSON.")
            metadata_hints["safety"] = ["safety", "safetyAnnotations"]
        elif provider_lower == "ollama":
            system_messages.append("Keep the JSON compact and deterministic.")
        else:
            system_messages.append(f"Provider {provider_lower} must respond with the exact JSON schema described.")

        return ReasoningConfig(
            schema_templates=templates,
            reasoning=reasoning_enabled,
            system_messages=system_messages,
            reasoning_prompt=reasoning_prompt,
            metadata_hints=metadata_hints,
            reasoning_effort=reasoning_effort,
        )

    def build_reasoning_prompt(self, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
        config = getattr(self, "reasoning_config", None)
        if config is None:
            return system_prompt, user_prompt
        system_segments = [segment for segment in config.system_messages if segment]
        system_segments.append(system_prompt)
        final_system_prompt = "\n\n".join(system_segments)
        final_user_prompt = user_prompt
        if config.reasoning and config.reasoning_prompt:
            final_user_prompt = f"{user_prompt}{config.reasoning_prompt}"
        return final_system_prompt, final_user_prompt

    def _build_json_schema(
        self,
        prompt_type: str,
        allowed_actions: List[str],
        template: Dict[str, Any],
        effective_max: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if prompt_type in {"action", "reasoned_action"}:
            schema_map = template.get("schemas", {})
            variants: List[Dict[str, Any]] = []
            for action in allowed_actions:
                schema_entry = schema_map.get(action)
                if schema_entry and schema_entry.get("json_schema"):
                    variants.append(deepcopy(schema_entry["json_schema"]))
            if not variants and schema_map:
                for schema_entry in schema_map.values():
                    json_schema = schema_entry.get("json_schema")
                    if json_schema:
                        variants.append(deepcopy(json_schema))
                        break
            if not variants:
                return None
            if len(variants) == 1:
                return variants[0]
            return {"oneOf": variants}

        if prompt_type == "options":
            effective_max = effective_max or template.get("max_k", 3)
            action_enum = allowed_actions or ["move", "switch", "dynamax", "terastallize"]
            option_schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": action_enum},
                    "target": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": False,
            }
            properties = {
                f"option_{index}": deepcopy(option_schema)
                for index in range(1, effective_max + 1)
            }
            return {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
                "minProperties": 1,
                "maxProperties": effective_max,
            }

        if prompt_type == "decision":
            action_enum = allowed_actions or ["move", "switch", "dynamax", "terastallize"]
            return {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": action_enum},
                            "target": {"type": "string"},
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    }
                },
                "required": ["decision"],
                "additionalProperties": False,
            }

        return None

    def _get_reasoning_request(self, prompt_type: str, fallback_prompt: str = "") -> ReasoningRequest:
        stored = self._latest_reasoning_requests.get(prompt_type)
        if stored is None:
            stored = ReasoningRequest(prompt=fallback_prompt)
        elif fallback_prompt and stored.prompt != fallback_prompt:
            stored = ReasoningRequest(prompt=fallback_prompt, json_schema=stored.json_schema)
        self._latest_reasoning_requests[prompt_type] = stored
        return stored

    def _coerce_reasoning_request(self, value: Any, prompt_type: str) -> ReasoningRequest:
        if isinstance(value, ReasoningRequest):
            self._latest_reasoning_requests[prompt_type] = value
            return value
        text = value or ""
        return self._get_reasoning_request(prompt_type, text)

    def _reasoning_effort_param(self) -> Optional[str]:
        config = getattr(self, "reasoning_config", None)
        if config and config.reasoning_effort:
            return config.reasoning_effort
        return None

    def format_reasoning_request(self, prompt_type: str, allowed_actions: List[str], max_k: int = 3) -> str:
        config = getattr(self, "reasoning_config", None)
        templates = config.schema_templates if config else self._base_schema_templates()
        template = templates.get(prompt_type, {})
        instruction = template.get("instruction", "")
        output_prefix = template.get("output_prefix", "")
        cleaned_actions: List[str] = []
        for action in allowed_actions:
            if action not in cleaned_actions:
                cleaned_actions.append(action)

        if prompt_type in {"action", "reasoned_action"}:
            schema_map = template.get("schemas", {})
            examples: List[str] = []
            for action in cleaned_actions:
                schema = schema_map.get(action)
                if schema:
                    examples.append(json.dumps(schema.get("example", {})))
            if not examples and schema_map:
                fallback = next(iter(schema_map.values()))
                examples.append(json.dumps(fallback.get("example", {})))
            example_string = " or ".join(examples)
            prompt = f"{instruction} {output_prefix}{example_string}\n"
            request = ReasoningRequest(
                prompt=prompt,
                json_schema=self._build_json_schema(prompt_type, cleaned_actions, template),
            )
            self._latest_reasoning_requests[prompt_type] = request
            return request.prompt

        if prompt_type == "options":
            template_max = template.get("max_k", max_k)
            effective_max = min(max_k, template_max)
            example_options: Dict[str, Dict[str, str]] = {}
            for index, action in enumerate(cleaned_actions[:effective_max], start=1):
                example_options[f"option_{index}"] = {
                    "action": action,
                    "target": f"<{action}_target>",
                }
            if not example_options:
                example_options["option_1"] = {"action": "move", "target": "<move_target>"}
            example_string = json.dumps(example_options)
            prompt = f"{instruction.format(max_k=effective_max)} {output_prefix}{example_string}\n"
            request = ReasoningRequest(
                prompt=prompt,
                json_schema=self._build_json_schema(prompt_type, cleaned_actions, template, effective_max),
            )
            self._latest_reasoning_requests[prompt_type] = request
            return request.prompt

        if prompt_type == "decision":
            default_action = cleaned_actions[0] if cleaned_actions else template.get("default_action", "move")
            example_decision = {"decision": {"action": default_action, "target": f"<{default_action}_target>"}}
            example_string = json.dumps(example_decision)
            prompt = f"{instruction} {output_prefix}{example_string}\n"
            request = ReasoningRequest(
                prompt=prompt,
                json_schema=self._build_json_schema(prompt_type, cleaned_actions, template),
            )
            self._latest_reasoning_requests[prompt_type] = request
            return request.prompt

        request = ReasoningRequest(prompt=instruction)
        self._latest_reasoning_requests[prompt_type] = request
        return request.prompt

    def _allow_gimmick_actions(self) -> bool:
        account_configuration = getattr(getattr(self, "ps_client", None), "account_configuration", None)
        username = getattr(account_configuration, "username", "") or ""
        return "pokellmon" not in username.lower()

    def _allowed_actions(self, battle: Battle, *, moves: bool, switches: bool) -> List[str]:
        actions: List[str] = []
        if moves:
            actions.append("move")
            if self._allow_gimmick_actions():
                if getattr(battle, "can_dynamax", False):
                    actions.append("dynamax")
                if getattr(battle, "can_tera", False):
                    actions.append("terastallize")
        if switches:
            actions.append("switch")
        ordered: List[str] = []
        for action in actions:
            if action not in ordered:
                ordered.append(action)
        return ordered

    def _resolve_metadata_path(self, payload: Dict[str, Any], path: str) -> Any:
        current: Any = payload
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _capture_reasoning_metadata(self, payload: Dict[str, Any]) -> None:
        self.reasoning_telemetry = {}
        config = getattr(self, "reasoning_config", None)
        if not config or not config.metadata_hints:
            return
        for label, paths in config.metadata_hints.items():
            for path in paths:
                value = self._resolve_metadata_path(payload, path)
                if value is not None:
                    self.reasoning_telemetry[label] = value
                    break

    def _battle_order_from_dict(
        self,
        action_payload: Dict[str, Any],
        battle: Battle,
        sim: LocalSim,
        *,
        dont_verify: bool = False,
        state_action_prompt: str = "",
    ) -> Optional[BattleOrder]:
        if "output" in action_payload and isinstance(action_payload["output"], dict):
            nested = self._battle_order_from_dict(
                action_payload["output"],
                battle,
                sim,
                dont_verify=dont_verify,
                state_action_prompt=state_action_prompt,
            )
            if nested is not None:
                return nested
        if "response" in action_payload and isinstance(action_payload["response"], dict):
            nested = self._battle_order_from_dict(
                action_payload["response"],
                battle,
                sim,
                dont_verify=dont_verify,
                state_action_prompt=state_action_prompt,
            )
            if nested is not None:
                return nested
        dynamax = "dynamax" in action_payload
        tera = "terastallize" in action_payload
        if "move" in action_payload or dynamax or tera:
            if dynamax:
                llm_move_id = action_payload["dynamax"].strip()
            elif tera:
                llm_move_id = action_payload["terastallize"].strip()
            else:
                llm_move_id = action_payload["move"].strip()
            move_list = battle.available_moves
            if dont_verify and battle.opponent_active_pokemon is not None:
                move_list = battle.opponent_active_pokemon.moves.values()
            for move in move_list:
                if move.id.lower().replace(' ', '') == llm_move_id.lower().replace(' ', ''):
                    return self.create_order(move, dynamax=dynamax, terastallize=tera)
            if dont_verify and llm_move_id and state_action_prompt:
                normalized = llm_move_id.lower().replace(' ', '')
                if normalized in state_action_prompt:
                    return self.create_order(Move(normalized, self.gen.gen), dynamax=dynamax, terastallize=tera)

        if "switch" in action_payload:
            llm_switch_species = action_payload["switch"].strip()
            switch_list = battle.available_switches
            if dont_verify:
                observable_switches = []
                for _, opponent_pokemon in battle.opponent_team.items():
                    if not opponent_pokemon.active:
                        observable_switches.append(opponent_pokemon)
                switch_list = observable_switches
            for pokemon in switch_list:
                if pokemon.species.lower().replace(' ', '') == llm_switch_species.lower().replace(' ', ''):
                    return self.create_order(pokemon)
        return None

    def parse_new(self, llm_output: str, battle: Battle, sim: LocalSim, state_action_prompt: str = "") -> Optional[BattleOrder]:
        try:
            parsed = json.loads(llm_output)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        self._capture_reasoning_metadata(parsed)
        payload: Dict[str, Any]
        if "decision" in parsed and isinstance(parsed["decision"], dict):
            decision = parsed["decision"]
            action_type = decision.get("action")
            target = decision.get("target")
            payload = {}
            if action_type == "switch":
                payload["switch"] = target or ""
            elif action_type in {"dynamax", "terastallize"}:
                payload[action_type] = target or ""
            elif action_type:
                payload["move"] = target or ""
            else:
                payload = decision
        else:
            payload = parsed
        return self._battle_order_from_dict(payload, battle, sim, state_action_prompt=state_action_prompt)

    def get_LLM_action(
        self,
        system_prompt,
        user_prompt,
        model,
        temperature=0.7,
        json_format=False,
        seed=None,
        stop=[],
        max_tokens=200,
        actions=None,
        llm=None,
        response_schema: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        if llm is None:
            output, _ = self.llm.get_LLM_action(
                system_prompt,
                user_prompt,
                model,
                temperature,
                json_format,
                seed,
                stop,
                max_tokens=max_tokens,
                actions=actions,
                response_schema=response_schema,
                reasoning_effort=reasoning_effort,
            )
        else:
            output, _ = llm.get_LLM_action(
                system_prompt,
                user_prompt,
                model,
                temperature,
                json_format,
                seed,
                stop,
                max_tokens=max_tokens,
                actions=actions,
                response_schema=response_schema,
                reasoning_effort=reasoning_effort,
            )
        return output
    
    def check_all_pokemon(self, pokemon_str: str) -> Pokemon:
        valid_pokemon = None
        if pokemon_str in self._pokemon_dict:
            valid_pokemon = pokemon_str
        else:
            closest = get_close_matches(pokemon_str, self._pokemon_dict.keys(), n=1, cutoff=0.8)
            if len(closest) > 0:
                valid_pokemon = closest[0]
        if valid_pokemon is None:
            return None
        pokemon = Pokemon(species=pokemon_str, gen=self.genNum)
        return pokemon

    def choose_move(self, battle: AbstractBattle):
        sim = LocalSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    self.strategy_prompt,
                    format=self.format,
                    prompt_translate=self.prompt_translate
        )
        if battle.turn <=1 and self.use_strat_prompt:
            self.strategy_prompt = sim.get_llm_system_prompt(self.format, self.llm, team_str=self.team_str, model='gpt-4o-2024-05-13')
        
        if battle.active_pokemon:
            if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
                next_action = BattleOrder(battle.available_switches[0])
                return next_action
            elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1 and len(battle.available_switches) == 0:
                return self.choose_max_damage_move(battle)
        elif len(battle.available_moves) <= 1 and len(battle.available_switches) == 0:
            return self.choose_max_damage_move(battle)

        system_prompt, state_prompt, state_action_prompt = sim.state_translate(battle) # add lower case
        moves = [move.id for move in battle.available_moves]
        switches = [pokemon.species for pokemon in battle.available_switches]
        actions = [moves, switches]

        if battle.active_pokemon.fainted or len(battle.available_moves) == 0:
            allowed_actions = self._allowed_actions(battle, moves=False, switches=True)
        elif len(battle.available_switches) == 0:
            allowed_actions = self._allowed_actions(battle, moves=True, switches=False)
        else:
            allowed_actions = self._allowed_actions(battle, moves=True, switches=True)

        constraint_prompt_io = self.format_reasoning_request("action", allowed_actions)
        constraint_prompt_cot = self.format_reasoning_request("reasoned_action", allowed_actions)
        constraint_prompt_tot_1 = self.format_reasoning_request("options", allowed_actions)
        constraint_prompt_tot_2 = self.format_reasoning_request("decision", allowed_actions)

        constraint_request_io = self._get_reasoning_request("action", constraint_prompt_io)
        constraint_request_cot = self._get_reasoning_request("reasoned_action", constraint_prompt_cot)
        constraint_request_tot_1 = self._get_reasoning_request("options", constraint_prompt_tot_1)
        constraint_request_tot_2 = self._get_reasoning_request("decision", constraint_prompt_tot_2)

        state_prompt_io = state_prompt + state_action_prompt + constraint_request_io.prompt
        state_prompt_cot = state_prompt + state_action_prompt + constraint_request_cot.prompt
        state_prompt_tot_1 = state_prompt + state_action_prompt + constraint_request_tot_1.prompt
        state_prompt_tot_2 = state_prompt + state_action_prompt + constraint_request_tot_2.prompt

        retries = 10
        # Chain-of-thought
        if self.prompt_algo == "io":
            return self.io(retries, system_prompt, state_prompt, constraint_request_cot, constraint_request_io, state_action_prompt, battle, sim, actions=actions)

        # Self-consistency with k = 3
        elif self.prompt_algo == "sc":
            return self.sc(retries, system_prompt, state_prompt, constraint_request_cot, constraint_request_io, state_action_prompt, battle, sim)

        # Tree of thought, k = 3
        elif self.prompt_algo == "tot":
            llm_output1 = ""
            next_action = None
            for i in range(retries):
                try:
                    tot_system_prompt, tot_prompt_1 = self.build_reasoning_prompt(system_prompt, state_prompt_tot_1)
                    llm_output1 = self.get_LLM_action(system_prompt=tot_system_prompt,
                                               user_prompt=tot_prompt_1,
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=200,
                                               json_format=True,
                                               response_schema=constraint_request_tot_1.json_schema,
                                               reasoning_effort=self._reasoning_effort_param())
                    break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if llm_output1 == "":
                return self.choose_max_damage_move(battle)

            for i in range(retries):
                try:
                    tot_prompt_2 = state_prompt_tot_2.replace("[OPTIONS]", llm_output1)
                    tot_system_prompt_2, tot_prompt_2 = self.build_reasoning_prompt(system_prompt, tot_prompt_2)
                    llm_output2 = self.get_LLM_action(system_prompt=tot_system_prompt_2,
                                               user_prompt=tot_prompt_2,
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=100,
                                               json_format=True,
                                               response_schema=constraint_request_tot_2.json_schema,
                                               reasoning_effort=self._reasoning_effort_param())

                    next_action = self.parse_new(llm_output2, battle, sim, state_action_prompt)
                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(json.dumps({"turn": battle.turn,
                                            "system_prompt": system_prompt,
                                            "user_prompt1": state_prompt_tot_1,
                                            "user_prompt2": state_prompt_tot_2,
                                            "llm_output1": llm_output1,
                                            "llm_output2": llm_output2,
                                            "battle_tag": battle.battle_tag
                                            }) + "\n")
                    if next_action is not None:     break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            return next_action

        elif self.prompt_algo == "minimax":
            try:
                # Initialize minimax optimizer if not already done
                if self.use_optimized_minimax and not self._minimax_initialized:
                    self._initialize_minimax_optimizer(battle)
                    
                if self.use_optimized_minimax:
                    return self.tree_search_optimized(retries, battle)
                else:
                    return self.tree_search(retries, battle)
            except Exception as e:
                print(f'minimax step failed ({e}). Using dmg calc')
                print(f'Exception: {e}', 'passed')
                return self.choose_max_damage_move(battle)

        
    def io(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle: Battle, sim, dont_verify=False, actions=None):
        next_action = None
        request_io = self._coerce_reasoning_request(constraint_prompt_io, "action")
        self._coerce_reasoning_request(constraint_prompt_cot, "reasoned_action")
        state_prompt_io = state_prompt + state_action_prompt + request_io.prompt
        system_prompt_io, user_prompt_io = self.build_reasoning_prompt(system_prompt, state_prompt_io)
        response_schema = request_io.json_schema
        reasoning_effort = self._reasoning_effort_param()

        for i in range(retries):
            try:
                llm_output = self.get_LLM_action(system_prompt=system_prompt_io,
                                            user_prompt=user_prompt_io,
                                            model=self.backend,
                                            temperature=self.temperature,
                                            max_tokens=300,
                                            # stop=["reason"],
                                            json_format=True,
                                            actions=actions,
                                            response_schema=response_schema,
                                            reasoning_effort=reasoning_effort)

                # load when llm does heavylifting for parsing
                if DEBUG:
                    print(f"Raw LLM output: {llm_output}")
                llm_action_json = json.loads(llm_output)
                if DEBUG:
                    print(f"Parsed JSON: {llm_action_json}")
                if isinstance(llm_action_json, dict):
                    self._capture_reasoning_metadata(llm_action_json)
                next_action = self._battle_order_from_dict(
                    llm_action_json if isinstance(llm_action_json, dict) else {},
                    battle,
                    sim,
                    dont_verify=dont_verify,
                    state_action_prompt=state_action_prompt,
                )

                if next_action is None and not isinstance(llm_action_json, dict):
                    raise ValueError('No valid action')

                # with open(f"{self.log_dir}/output.jsonl", "a") as f:
                #     f.write(json.dumps({"turn": battle.turn,
                #                         "system_prompt": system_prompt,
                #                         "user_prompt": state_prompt_io,
                #                         "llm_output": llm_output,
                #                         "battle_tag": battle.battle_tag
                #                         }) + "\n")
                
                if next_action is not None:
                    break
            except Exception as e:
                print(f'Exception: {e}', 'passed')
                continue
        if next_action is None:
            print('No action found. Choosing max damage move')
            try:
                print('No action found', llm_action_json, actions, dont_verify)
            except:
                pass
            print()
            # raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
            next_action = self.choose_max_damage_move(battle)
        return next_action

    def sc(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim):
        actions = [self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim) for i in range(self.K)]
        action_message = [action.message for action in actions]
        _, counts = np.unique(action_message, return_counts=True)
        index = np.argmax(counts)
        return actions[index]
    
    def estimate_matchup(self, sim: LocalSim, battle: Battle, mon: Pokemon, mon_opp: Pokemon, is_opp: bool=False) -> Tuple[Move, int]:
        hp_remaining = []
        moves = list(mon.moves.keys())
        if is_opp:
            moves = sim.get_opponent_current_moves(mon=mon)
        if battle.active_pokemon.species == mon.species and not is_opp:
            moves = [move.id for move in battle.available_moves]
        for move_id in moves:
            move = Move(move_id, gen=sim.gen.gen)
            t = np.inf
            if move.category == MoveCategory.STATUS:
                # apply stat boosting effects to see if it will KO in fewer turns
                t = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
            else:
                t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
            hp_remaining.append(t)
            # _, hp2, _, _ = sim.calculate_remaining_hp(battle.active_pokemon, battle.opponent_active_pokemon, move, None)
            # hp_remaining.append(hp2)
        hp_best_index = np.argmin(hp_remaining)
        best_move = moves[hp_best_index]
        best_move_turns = hp_remaining[hp_best_index]
        best_move = Move(best_move, gen=sim.gen.gen)
        best_move = self.create_order(best_move)
        # check special moves: tera/dyna
        # dyna for gen 8
        if sim.battle._data.gen == 8 and sim.battle.can_dynamax:
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen).dynamaxed
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, dynamax=True)
                        best_move_turns = t
        # tera for gen 9
        elif sim.battle._data.gen == 9 and sim.battle.can_tera:
            mon.terastallize()
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen)
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, terastallize=True)
                        best_move_turns = t
            mon.unterastallize()
            
        return best_move, best_move_turns

    def dmg_calc_move(self, battle: AbstractBattle, return_move: bool=False):
        sim = LocalSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    format=self.format
        )
        best_action = None
        best_action_turns = np.inf
        if battle.available_moves and not battle.active_pokemon.fainted:
            # try moves and find hp remaining for opponent
            mon = battle.active_pokemon
            mon_opp = battle.opponent_active_pokemon
            best_action, best_action_turns = self.estimate_matchup(sim, battle, mon, mon_opp)
        if return_move:
            if best_action is None:
                return None, best_action_turns
            return best_action.order, best_action_turns
        if best_action_turns > 4:
            return None, np.inf
        if best_action is not None:
            return best_action, best_action_turns
        return self.choose_random_move(battle), 1
    
    
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score
    
    def _get_fast_heuristic_evaluation(self, battle_state):
        """Fast heuristic evaluation for leaf nodes when LLM is not used."""
        try:
            player_hp = int(battle_state.active_pokemon.current_hp_fraction * 100) if battle_state.active_pokemon else 0
            opp_hp = int(battle_state.opponent_active_pokemon.current_hp_fraction * 100) if battle_state.opponent_active_pokemon else 0
            player_remaining = len([p for p in battle_state.team.values() if not p.fainted])
            opp_remaining = len([p for p in battle_state.opponent_team.values() if not p.fainted])
            
            # Use cached fast evaluation
            return fast_battle_evaluation(
                player_hp, opp_hp, 
                player_remaining, opp_remaining,
                battle_state.turn
            )
        except:
            # Ultimate fallback to basic hp difference
            try:
                from poke_env.player.local_simulation import LocalSim
                sim = LocalSim(battle_state, 
                            self.move_effect,
                            self.pokemon_move_dict,
                            self.ability_effect,
                            self.pokemon_ability_dict,
                            self.item_effect,
                            self.pokemon_item_dict,
                            self.gen,
                            self._dynamax_disable,
                            format=self.format
                )
                return sim.get_hp_diff()
            except:
                return 50  # Neutral fallback score
    
    def _initialize_minimax_optimizer(self, battle):
        """Initialize the minimax optimizer with current battle state."""
        try:
            initialize_minimax_optimization(
                battle=battle,
                move_effect=self.move_effect,
                pokemon_move_dict=self.pokemon_move_dict,
                ability_effect=self.ability_effect,
                pokemon_ability_dict=self.pokemon_ability_dict,
                item_effect=self.item_effect,
                pokemon_item_dict=self.pokemon_item_dict,
                gen=self.gen,
                _dynamax_disable=self._dynamax_disable,
                format=self.format,
                prompt_translate=self.prompt_translate
            )
            self._minimax_initialized = True
            print("🚀 Minimax optimizer initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize minimax optimizer: {e}")
            self.use_optimized_minimax = False  # Fallback to original

    def check_timeout(self, start_time, battle):
        if time.time() - start_time > 30:
            print('default due to time')
            move, _ = self.dmg_calc_move(battle)
            return move
        else:
            return None
    
    def tree_search(self, retries, battle, sim=None, return_opp = False) -> BattleOrder:
        # generate local simulation
        root = SimNode(battle, 
                        self.move_effect,
                        self.pokemon_move_dict,
                        self.ability_effect,
                        self.pokemon_ability_dict,
                        self.item_effect,
                        self.pokemon_item_dict,
                        self.gen,
                        self._dynamax_disable,
                        depth=1,
                        format=self.format,
                        prompt_translate=self.prompt_translate,
                        sim=sim
                        ) 
        q = [
                root
            ]
        leaf_nodes = []
        # create node and add to q B times
        start_time = time.time()
        while len(q) != 0:
            node = q.pop(0)
            # choose node for expansion
            # generate B actions
            player_actions = []
            system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(return_actions=True)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # end if terminal
            if node.simulation.is_terminal() or node.depth == self.K:
                try:
                    # value estimation for leaf nodes
                    value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' +\
                                    'Add points based on the effectiveness of current available moves.' +\
                                    'Award points for each pokemon remaining on the player\'s team, weighted by their strength' +\
                                    'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' +\
                                    'Subtract points for excessive switching.' +\
                                    'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' +\
                                    'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                    cot_prompt = 'Briefly justify your total score, up to 100 words. Then, conclude with the score in the JSON format: {"score": <total_points>}. '
                    state_prompt_io = state_prompt + value_prompt + cot_prompt
                    system_prompt_value, user_prompt_value = self.build_reasoning_prompt(system_prompt, state_prompt_io)
                    llm_output = self.get_LLM_action(system_prompt=system_prompt_value,
                                                    user_prompt=user_prompt_value,
                                                    model=self.backend,
                                                    temperature=self.temperature,
                                                    max_tokens=500,
                                                    json_format=True,
                                                    llm=self.llm_value
                                                    )
                    # load when llm does heavylifting for parsing
                    llm_action_json = json.loads(llm_output)
                    node.hp_diff = int(llm_action_json['score'])
                except Exception as e:
                    node.hp_diff = node.simulation.get_hp_diff()                    
                    print(e)
                
                leaf_nodes.append(node)
                continue
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # estimate opp
            try:
                action_opp, opp_turns = self.estimate_matchup(node.simulation, node.simulation.battle, node.simulation.battle.opponent_active_pokemon, node.simulation.battle.active_pokemon, is_opp=True)
            except:
                action_opp = None
                opp_turns = np.inf
            ##############################
            # generate players's action  #
            ##############################
            if not node.simulation.battle.active_pokemon.fainted and len(battle.available_moves) > 0:
                # get dmg calc move
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(node.simulation.battle)
                if dmg_calc_out is not None:
                    if dmg_calc_turns <= opp_turns:
                        try:
                            # ask LLM to use heuristic tool or minimax search
                            tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                                1. Damage calculator advantages:
                                - Quick and efficient for finding optimal damaging moves
                                - Useful when a clear type advantage or high-power move is available
                                - Effective when the opponent's is not switching and current pokemon is likely to KO opponent

                                2. Minimax tree search advantages:
                                - Can model opponent behavior and predict future moves
                                - Useful in complex situations with multiple viable options
                                - Effective when long-term strategy is crucial

                                3. Current battle state:
                                - Remaining Pokémon on each side
                                - Health of active Pokémon
                                - Type matchups
                                - Available moves and their effects
                                - Presence of status conditions or field effects

                                4. Uncertainty level:
                                - How predictable is the opponent's next move?
                                - Are there multiple equally viable options for your next move?

                                Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

                                {"choice":"damage calculator"} or {"choice":"minimax"}'''

                            state_prompt_io = state_prompt + tool_prompt
                            system_prompt_tool, user_prompt_tool = self.build_reasoning_prompt(system_prompt, state_prompt_io)
                            llm_output = self.get_LLM_action(system_prompt=system_prompt_tool,
                                                            user_prompt=user_prompt_tool,
                                                            model=self.backend,
                                                            temperature=0.6,
                                                            max_tokens=100,
                                                            json_format=True,
                                                            )
                            # load when llm does heavylifting for parsing
                            llm_action_json = json.loads(llm_output)
                            if 'choice' in llm_action_json.keys():
                                if llm_action_json['choice']  != 'minimax':
                                    if return_opp:
                                        # use tool to save time and llm when move makes bigger difference
                                        return dmg_calc_out, action_opp
                                    return dmg_calc_out
                        except:
                            print('defaulting to minimax')
                    player_actions.append(dmg_calc_out)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # get llm switch
            if len(node.simulation.battle.available_switches) != 0:# or opp_turns < dmg_calc_turns):
                state_action_prompt_switch = state_action_prompt + action_prompt_switch + '\nYou can only choose to switch this turn.\n'
                switch_actions = self._allowed_actions(node.simulation.battle, moves=False, switches=True)
                constraint_prompt_io = self.format_reasoning_request("action", switch_actions)
                constraint_request_io = self._get_reasoning_request("action", constraint_prompt_io)
                for i in range(2):
                    action_llm_switch = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_request_io, state_action_prompt_switch, node.simulation.battle, node.simulation)
                    if len(player_actions) == 0:
                        player_actions.append(action_llm_switch)
                    elif action_llm_switch.message != player_actions[-1].message:
                        player_actions.append(action_llm_switch)

            if not node.simulation.battle.active_pokemon.fainted and len(battle.available_moves) > 0:# and not opp_turns < dmg_calc_turns:
                # get llm move
                state_action_prompt_move = state_action_prompt + action_prompt_move + '\nYou can only choose to move this turn.\n'
                move_actions = self._allowed_actions(node.simulation.battle, moves=True, switches=False)
                constraint_prompt_io = self.format_reasoning_request("action", move_actions)
                constraint_request_io = self._get_reasoning_request("action", constraint_prompt_io)
                action_llm_move = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_request_io, state_action_prompt_move, node.simulation.battle, node.simulation)
                if len(player_actions) == 0:
                    player_actions.append(action_llm_move)
                elif action_llm_move.message != player_actions[0].message:
                    player_actions.append(action_llm_move)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            ##############################
            # generate opponent's action #
            ##############################
            opponent_actions = []
            tool_is_optimal = False
            # dmg calc suggestion
            # action_opp, opp_turns = self.estimate_matchup(node.simulation, node.simulation.battle, node.simulation.battle.opponent_active_pokemon, node.simulation.battle.active_pokemon, is_opp=True)
            if action_opp is not None:
                tool_is_optimal = True
                opponent_actions.append(self.create_order(action_opp))
            # heuristic matchup switch action
            best_score = np.inf
            best_action = None
            for mon in node.simulation.battle.opponent_team.values():
                if mon.species == node.simulation.battle.opponent_active_pokemon.species:
                    continue
                score = self._estimate_matchup(mon, node.simulation.battle.active_pokemon)
                if score < best_score:
                    best_score = score
                    best_action = mon
            if best_action is not None:
                opponent_actions.append(self.create_order(best_action))
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # create opponent prompt from battle sim
            system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o = node.simulation.get_opponent_prompt(system_prompt)
            action_o = self.io(2, system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o, node.simulation.battle, node.simulation, dont_verify=True)
            is_repeat_action_o = np.array([action_o.message == opponent_action.message for opponent_action in opponent_actions]).any()
            if not is_repeat_action_o:
                opponent_actions.append(action_o)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # simulate outcome
            if node.depth < self.K:
                for action_p in player_actions:
                    for action_o in opponent_actions:
                        node_new = copy(node)
                        node_new.simulation.battle = copy(node.simulation.battle)
                        # if not tool_is_optimal:
                        node_new.children = []
                        node_new.depth = node.depth + 1
                        node_new.action = action_p
                        node_new.action_opp = action_o
                        node_new.parent_node = node
                        node_new.parent_action = node.action
                        node.children.append(node_new)
                        node_new.simulation.step(action_p, action_o)
                        q.append(node_new)

        # choose best action according to max or min rule
        def get_tree_action(root: SimNode):
            if len(root.children) == 0:
                return root.action, root.hp_diff, root.action_opp
            score_dict = {}
            action_dict = {}
            opp_dict = {}
            for child in root.children:
                action = str(child.action.order)
                _, score, _ = get_tree_action(child)
                if action in score_dict.keys():
                    # imitation
                    # score_dict[action] = score + score_dict[action]
                    # minimax
                    score_dict[action] = min(score, score_dict[action])
                else:
                    score_dict[action] = score
                    action_dict[action] = child.action
                    opp_dict[action] = child.action_opp
            scores = list(score_dict.values())
            best_action_str = list(action_dict.keys())[np.argmax(scores)]
            return action_dict[best_action_str], score_dict[best_action_str], opp_dict[best_action_str]
        
        action, _, action_opp = get_tree_action(root)
        end_time = time.time()
        if return_opp:
            return action, action_opp
        return action

    def tree_search_optimized(self, retries, battle, sim=None, return_opp=False) -> BattleOrder:
        """
        Optimized version of tree_search using object pooling and caching.
        
        This version provides significant performance improvements for minimax:
        - Object pooling for LocalSim instances
        - LLM choice between damage calculator and minimax upfront
        - Battle state caching to avoid repeated computations
        """
        optimizer = get_minimax_optimizer()
        start_time = time.time()
        
        try:
            # Create optimized root node
            root = optimizer.create_optimized_root(battle)
            
            # Get battle state information for LLM decision
            system_prompt, state_prompt, _, _, _, _, _ = root.simulation.get_player_prompt(return_actions=True)
            
            # Ask LLM upfront whether to use minimax or damage calculator
            if not battle.active_pokemon.fainted and len(battle.available_moves) > 0:
                # Get dmg calc move for potential early return
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(battle)
                if dmg_calc_out is not None:
                    try:
                        # Ask LLM to choose between damage calculator tool or minimax search upfront
                        tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                        1. Damage calculator advantages:
                        - Quick and efficient for finding optimal damaging moves
                        - Useful when a clear type advantage or high-power move is available
                        - Effective when the opponent is not switching and current pokemon is likely to KO opponent

                        2. Minimax tree search advantages:
                        - Can model opponent behavior and predict future moves
                        - Useful in complex situations with multiple viable options
                        - Effective when long-term strategy is crucial

                        3. Current battle state:
                        - Remaining Pokémon on each side
                        - Health of active Pokémon
                        - Type matchups
                        - Available moves and their effects
                        - Presence of status conditions or field effects

                        4. Uncertainty level:
                        - How predictable is the opponent's next move?
                        - Are there multiple equally viable options for your next move?

                        Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

                        {"choice":"damage calculator"} or {"choice":"minimax"}'''

                        state_prompt_io = state_prompt + tool_prompt
                        system_prompt_tool, user_prompt_tool = self.build_reasoning_prompt(system_prompt, state_prompt_io)
                        llm_output = self.get_LLM_action(system_prompt=system_prompt_tool,
                                                        user_prompt=user_prompt_tool,
                                                        model=self.backend,
                                                        temperature=0.6,
                                                        max_tokens=100,
                                                        json_format=True,
                                                        )
                        # Load when llm does heavylifting for parsing
                        llm_action_json = json.loads(llm_output)
                        if 'choice' in llm_action_json.keys():
                            if llm_action_json['choice'] != 'minimax':
                                # LLM chose damage calculator - return it directly
                                print("LLM chose damage calculator over minimax")
                                if return_opp:
                                    try:
                                        action_opp, _ = self.estimate_matchup(root.simulation, battle, 
                                                                           battle.opponent_active_pokemon, 
                                                                           battle.active_pokemon, is_opp=True)
                                        return dmg_calc_out, self.create_order(action_opp) if action_opp else None
                                    except:
                                        return dmg_calc_out, None
                                return dmg_calc_out
                    except Exception as e:
                        print(f'LLM choice failed ({e}), defaulting to minimax')
            
            print("Using minimax tree search")
            
            q = [root]
            leaf_nodes = []
            
            while len(q) != 0:
                node = q.pop(0)
                
                # Get available actions efficiently 
                player_actions = []
                system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(return_actions=True)
                
                # Check if terminal node or reached depth limit
                if node.simulation.is_terminal() or node.depth == self.K:
                    try:
                        # Use LLM value function for leaf nodes evaluation
                        value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' +\
                                        'Add points based on the effectiveness of current available moves.' +\
                                        'Award points for each pokemon remaining on the player\'s team, weighted by their strength' +\
                                        'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' +\
                                        'Subtract points for excessive switching.' +\
                                        'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' +\
                                        'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                        cot_prompt = 'Briefly justify your total score, up to 100 words. Then, conclude with the score in the JSON format: {"score": <total_points>}. '
                        state_prompt_io = state_prompt + value_prompt + cot_prompt
                        system_prompt_value, user_prompt_value = self.build_reasoning_prompt(system_prompt, state_prompt_io)
                        llm_output = self.get_LLM_action(system_prompt=system_prompt_value,
                                                        user_prompt=user_prompt_value,
                                                        model=self.backend,
                                                        temperature=self.temperature,
                                                        max_tokens=500,
                                                        json_format=True,
                                                        llm=self.llm_value
                                                        )
                        # Load when llm does heavylifting for parsing
                        llm_action_json = json.loads(llm_output)
                        node.hp_diff = int(llm_action_json['score'])
                    except Exception as e:
                        # Fallback to damage calculator based evaluation
                        try:
                            damage_calc_move, damage_calc_turns = self.dmg_calc_move(node.simulation.battle)
                            if damage_calc_turns < float('inf'):
                                # Score based on how many turns to KO opponent vs how many they need to KO us
                                try:
                                    opp_action, opp_turns = self.estimate_matchup(
                                        node.simulation, node.simulation.battle,
                                        node.simulation.battle.opponent_active_pokemon,
                                        node.simulation.battle.active_pokemon,
                                        is_opp=True
                                    )
                                    # Higher score if we can KO faster than opponent
                                    if opp_turns > damage_calc_turns:
                                        node.hp_diff = 75  # We have advantage
                                    elif opp_turns == damage_calc_turns:
                                        node.hp_diff = 50  # Even
                                    else:
                                        node.hp_diff = 25  # Opponent has advantage
                                except:
                                    node.hp_diff = 50  # Neutral if opponent estimation fails
                            else:
                                # Use basic hp difference if damage calc fails
                                node.hp_diff = node.simulation.get_hp_diff()
                        except:
                            # Ultimate fallback to basic hp difference
                            node.hp_diff = node.simulation.get_hp_diff()
                        print(f"LLM value function failed, using damage calculator fallback: {e}")
                    
                    leaf_nodes.append(node)
                    continue
                
                # Estimate opponent action (reuse existing logic)
                try:
                    action_opp, opp_turns = self.estimate_matchup(
                        node.simulation, node.simulation.battle, 
                        node.simulation.battle.opponent_active_pokemon, 
                        node.simulation.battle.active_pokemon, 
                        is_opp=True
                    )
                except:
                    action_opp = None
                    opp_turns = float('inf')
                
                # Get player actions - damage calculator move
                if not node.simulation.battle.active_pokemon.fainted and len(battle.available_moves) > 0:
                    # Get dmg calc move
                    dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(node.simulation.battle)
                    if dmg_calc_out is not None:
                        player_actions.append(dmg_calc_out)

                # Generate opponent actions (reuse existing logic)
                opponent_actions = []
                if action_opp is not None:
                    opponent_actions.append(self.create_order(action_opp))
                
                # Get more opponent actions via LLM (simplified)
                try:
                    system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o = node.simulation.get_opponent_prompt(system_prompt)
                    action_o = self.io(2, system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o, node.simulation.battle, node.simulation, dont_verify=True)
                    if action_o not in opponent_actions:
                        opponent_actions.append(action_o)
                except:
                    pass  # Use what we have
                
                # Generate a few additional actions
                try:
                    action_io = self.io(2, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, node.simulation.battle, node.simulation, actions=player_actions)
                    if action_io not in player_actions:
                        player_actions.append(action_io)
                except:
                    pass
                
                # Create child nodes efficiently (if not at depth limit)
                if node.depth < self.K and player_actions and opponent_actions:
                    for action_p in player_actions[:2]:  # Limit to 2 player actions for performance
                        for action_o in opponent_actions[:2]:  # Limit to 2 opponent actions for performance
                            try:
                                child_node = node.create_child_node(action_p, action_o)
                                q.append(child_node)
                            except Exception as e:
                                print(f"Failed to create child node: {e}")
                                continue
            
            # Choose best action using original logic
            def get_tree_action(root_node):
                if len(root_node.children) == 0:
                    return root_node.action, root_node.hp_diff, root_node.action_opp
                    
                score_dict = {}
                action_dict = {}
                opp_dict = {}
                
                for child in root_node.children:
                    action = str(child.action.order)
                    if action not in score_dict:
                        score_dict[action] = []
                        action_dict[action] = child.action
                        opp_dict[action] = child.action_opp
                    score_dict[action].append(child.hp_diff)
                
                # Use max score for each action
                for action in score_dict:
                    score_dict[action] = max(score_dict[action])
                
                best_action_str = max(score_dict, key=score_dict.get)
                return action_dict[best_action_str], score_dict[best_action_str], opp_dict[best_action_str]
            
            action, _, action_opp = get_tree_action(root)
            
            # Cleanup resources
            optimizer.cleanup_tree(root)
            
            # Log performance stats
            end_time = time.time()
            stats = optimizer.get_performance_stats()
            print(f"⚡ Optimized minimax: {end_time - start_time:.2f}s, "
                  f"Pool reuse: {stats['pool_stats']['reuse_rate']:.2f}, "
                  f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2f}")
            
            if return_opp:
                return action, action_opp
            return action
            
        except Exception as e:
            print(f"Optimized minimax failed: {e}, falling back to damage calculator")
            # Cleanup any resources
            try:
                optimizer.cleanup_tree(root)
            except:
                pass
            # Fallback to damage calculator instead of original tree search
            try:
                dmg_calc_move, _ = self.dmg_calc_move(battle)
                if dmg_calc_move is not None:
                    if return_opp:
                        try:
                            action_opp, _ = self.estimate_matchup(None, battle, 
                                                               battle.opponent_active_pokemon, 
                                                               battle.active_pokemon, is_opp=True)
                            return dmg_calc_move, self.create_order(action_opp) if action_opp else None
                        except:
                            return dmg_calc_move, None
                    return dmg_calc_move
            except:
                pass
            # Ultimate fallback to max damage move
            return self.choose_max_damage_move(battle)
 
    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: Battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)
