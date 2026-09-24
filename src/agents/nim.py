"""
src/agents/nim.py

Shared NVIDIA NIM access for the CrewAI agents.

The model name lives HERE, in code -- a deliberate project decision,
unlike the transcript pipeline's GROQ_WHISPER_MODEL / GROQ_LLM_MODEL
env vars: the agents' model choice is treated as architecture, not
deployment config, so no *_MODEL variable is ever read from .env for
the agents. Only the API keys come from .env.

Both the extractor and the synthesizer run
nvidia/nemotron-3-ultra-550b-a55b, routed through LiteLLM's
nvidia_nim provider (CrewAI's LLM layer), which targets NVIDIA's
OpenAI-compatible endpoint at https://integrate.api.nvidia.com/v1.

Key rotation mirrors the Groq pipeline convention -- primary key
first, fallback second -- but the rotation happens in the agents' own
retry loops (at crew level), since LiteLLM does not rotate keys on
its own.
"""

from __future__ import annotations

import os

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

NIM_MODEL = "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b"


def candidate_llms(temperature: float) -> list[tuple[str, LLM]]:
    """
    One LLM instance per configured key, in rotation order: primary
    first, then fallback. Unset keys are skipped; an empty list means
    "no key configured" and callers raise their own descriptive error.
    """
    candidates: list[tuple[str, LLM]] = []
    for label, variable in (
        ("primary", "NVIDIA_API_KEY"),
        ("fallback", "FALLBACK_NVIDIA_API_KEY"),
    ):
        key = os.environ.get(variable, "")
        if key:
            candidates.append(
                (label, LLM(model=NIM_MODEL, api_key=key, temperature=temperature))
            )
    return candidates
