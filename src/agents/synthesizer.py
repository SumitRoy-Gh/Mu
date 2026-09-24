"""
src/agents/synthesizer.py

Synthesizer agent (CrewAI). Stage 3 of the notes pipeline, run exactly
once over ALL chunk-level extractions produced by the extractor:
merges topics that got split across chunk boundaries, removes
redundant restatements, and arranges the material into one coherent
hierarchical outline (headings, subheadings, bullets) with timestamp
ranges so the outline maps back onto the video.

This is the reasoning-heavy step -- only this agent sees every
chunk's findings together, so only it can merge across boundaries and
deduplicate globally. Per the pipeline decision it runs on the same
nvidia/nemotron-3-ultra-550b-a55b model as the extractor (model name
in src/agents/nim.py, never .env).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from crewai import Agent, Crew, Task

from .extractor import ChunkExtractionResult
from .nim import candidate_llms

logger = logging.getLogger("synthesizer_agent")

SYNTHESIZER_TEMPERATURE = 0.3


class SynthesisError(Exception):
    """Raised when the outline could not be synthesized on any key."""


_TASK_DESCRIPTION = (
    "You are given JSON extracted from every chunk of one video, in "
    "chronological order. Each chunk reports its timestamps (start_time "
    "and end_time in seconds) and the topics found in it, each with "
    "key_points, definitions, examples and processes.\n\n"
    "EXTRACTIONS:\n"
    "\"\"\"\n{extractions_json}\n\"\"\"\n\n"
    "Write the final study outline for the entire video:\n"
    "- Merge topics that continue across chunk boundaries into one "
    "section; never present the same topic twice.\n"
    "- Remove redundant restatements of the same idea; keep the "
    "clearest wording.\n"
    "- Keep chronological order unless the material clearly groups "
    "better thematically.\n"
    "- Never invent content that is not present in the extractions.\n"
    "- Give every major section heading a timestamp range, formatted "
    "like [MM:SS - MM:SS], derived from the contributing chunks' "
    "start_time and end_time.\n\n"
    "Output plain Markdown only: one top-level heading (a descriptive "
    "title for the video), second-level headings per merged topic with "
    "its timestamp range, third-level subheadings where a topic has "
    "distinct sub-areas, and bullet lists for the points, definitions, "
    "examples and process steps. No preamble, no closing remarks."
)


def _build_agent(llm: Any) -> Agent:
    return Agent(
        role="Study notes synthesizer",
        goal=(
            "Turn the per-chunk extractions from an entire video into one "
            "coherent, deduplicated hierarchical study outline."
        ),
        backstory=(
            "You are the final agent in a multi-agent notes pipeline. "
            "Extractor agents mined every transcript chunk for topics, "
            "key points, definitions, examples and processes; only you "
            "see all of their findings at once, so only you can merge "
            "topics that straddle chunk boundaries and cut redundant "
            "restatements. Students will study directly from your "
            "outline, so it must be organized, complete and free of "
            "repetition."
        ),
        llm=llm,
        allow_delegation=False,
    )


def _crew_text(result: Any) -> str:
    raw = getattr(result, "raw", None)
    return raw if isinstance(raw, str) else str(result)


def synthesize_notes(
    results: list[ChunkExtractionResult],
    llms: Optional[list[tuple[str, Any]]] = None,
) -> str:
    """
    Runs the synthesizer crew once over every chunk extraction and
    returns the merged hierarchical outline as Markdown. Failed calls
    rotate to the fallback NVIDIA key before giving up.
    """
    if not results:
        raise ValueError("results is empty -- nothing to synthesize")

    extractions_json = json.dumps(
        [
            {
                "chunk_index": r.chunk_index,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "topics": [topic.model_dump() for topic in r.extraction.topics],
            }
            for r in results
        ],
        indent=2,
    )

    candidates = llms if llms is not None else candidate_llms(SYNTHESIZER_TEMPERATURE)
    if not candidates:
        raise SynthesisError(
            "No NVIDIA API key is set. Add NVIDIA_API_KEY (and optionally "
            "FALLBACK_NVIDIA_API_KEY) to your .env file."
        )

    last_error: Exception | None = None
    for label, llm in candidates:
        try:
            agent = _build_agent(llm)
            task = Task(
                description=_TASK_DESCRIPTION.format(extractions_json=extractions_json),
                expected_output=(
                    "A hierarchical Markdown outline: one top-level title "
                    "heading, second-level topic sections with timestamp "
                    "ranges in their headings, occasional third-level "
                    "subheadings, and bullet points covering all key "
                    "points, definitions, examples and step-by-step "
                    "processes. Plain Markdown text, nothing else."
                ),
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task])
            outline = _crew_text(crew.kickoff()).strip()
            logger.info("outline synthesized (%s key)", label)
            return outline
        except Exception as exc:
            last_error = exc
            logger.warning("synthesis with %s key failed: %s", label, exc)

    raise SynthesisError(
        f"synthesis failed on all available keys (last error: {last_error})"
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.synthesizer <youtube_url>")
        sys.exit(1)

    from ..transcript_pipeline.translation import get_english_transcript
    from .extractor import extract_chunks
    from .segmentor import chunk_transcript

    transcript = get_english_transcript(sys.argv[1])
    results = extract_chunks(chunk_transcript(transcript))
    print(synthesize_notes(results))
