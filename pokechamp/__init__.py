"""
PokéChamp LLM Player Module

This module contains the LLM-based Pokemon battle player implementation,
including prompts, translation utilities, and various LLM backend integrations.

Environment loading
-------------------
Automatically load a local .env if present so API keys configured in
`pokechamp/.env` are available without exporting in the shell.
"""

# Best-effort .env autoload (no-op if dependency missing)
try:  # pragma: no cover - environment bootstrap
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from current working dir or parents
except Exception:
    pass

# Main classes - import when needed to avoid circular imports
# from pokechamp.llm_player import LLMPlayer  
# from pokechamp.gemini_player import GeminiPlayer
# from pokechamp.gpt_player import GPTPlayer
# from pokechamp.llama_player import LLAMAPlayer  
# from pokechamp.openrouter_player import OpenRouterPlayer
# Prompts not imported here to avoid circular imports with poke_env
# prompt_eval not imported here to avoid circular imports

__all__ = [
    # All imports handled directly to avoid circular imports
    # Import pokechamp.llm_player, pokechamp.gpt_player, etc. directly
]

__version__ = "1.0.0"
