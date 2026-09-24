"""
src/agents/extractor.py

Extractor agent (CrewAI). Runs once per TranscriptChunk -- chunks can
be processed in parallel -- and reads the chunk's raw text to pull out
what is actually worth noting: key points, definitions, examples, and
step-by-step processes, organized by topic. The chunk's timestamp
range is attached by this code; the model is never asked to compute
timestamps.

One crew per chunk: a single Agent + Task whose response is parsed and
validated against the ChunkExtraction pydantic schema. A malformed
response earns one stricter repair attempt on the same key; a
transport or auth failure rotates straight to the fallback NVIDIA key
(mirroring the Groq pipeline's convention, where a second key only
plausibly fixes transient or auth failures).

Model: nvidia/nemotron-3-ultra-550b-a55b via NVIDIA NIM, through
LiteLLM's nvidia_nim provider. The model name lives in code
(src/agents/nim.py), never in .env. Extraction is structural work
rather than deep reasoning, but per the pipeline decision both agents
share the nemotron-3-ultra model.

Given the segmentor's 700K-token chunk budget, most videos produce a
SINGLE chunk, so the common case is one crew kickoff, not a batch.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from crewai import Agent, Crew, Task
from pydantic import BaseModel, Field, ValidationError

from .nim import candidate_llms
from .segmentor import TranscriptChunk

logger = logging.getLogger("extractor_agent")

MAX_PARALLEL_CHUNKS = 4
MAX_REPAIR_ATTEMPTS = 1
EXTRACTOR_TEMPERATURE = 0.2


class ExtractionError(Exception):
    """Raised when a chunk could not be extracted after all keys/retries."""


class Definition(BaseModel):
    term: str
    explanation: str


class Process(BaseModel):
    name: str
    steps: list[str] = Field(default_factory=list)


class TopicExtraction(BaseModel):
    title: str = Field(description="Short, specific title for this topic.")
    key_points: list[str] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    processes: list[Process] = Field(default_factory=list)


class ChunkExtraction(BaseModel):
    topics: list[TopicExtraction]


@dataclass
class ChunkExtractionResult:
    chunk_index: int
    start_time: float
    end_time: float
    extraction: ChunkExtraction


_JSON_SCHEMA_HINT = (
    "Respond with ONLY a JSON object of this exact shape, no commentary, "
    "no markdown code fences:\n"
    "{\n"
    '  "topics": [\n'
    "    {\n"
    '      "title": "short specific topic title",\n'
    '      "key_points": ["...", "..."],\n'
    '      "definitions": [{"term": "...", "explanation": "..."}],\n'
    '      "examples": ["...", "..."],\n'
    '      "processes": [{"name": "...", "steps": ["...", "..."]}]\n'
    "    }\n"
    "  ]\n"
    "}"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _build_task_description(chunk_text: str, repair: bool) -> str:
    prefix = (
        "Your previous response was not valid JSON matching the required "
        "shape. Re-read the instructions carefully and respond again, "
        "following the schema exactly.\n\n"
        if repair
        else ""
    )
    return (
        f"{prefix}"
        "Identify the distinct topics covered in the transcript below, and "
        "for each one extract its key points, any definitions given, any "
        "examples used, and any step-by-step processes described. Skip "
        "filler, tangents, and restatements -- extract substance only. If "
        "the transcript contains nothing substantive, return an empty "
        "topics list.\n\n"
        f"{_JSON_SCHEMA_HINT}\n\n"
        f"Transcript:\n{chunk_text}"
    )


def _build_agent(llm: Any) -> Agent:
    return Agent(
        role="Video transcript extractor",
        goal=(
            "Extract the topics, key points, definitions, examples and "
            "step-by-step processes actually present in one transcript "
            "chunk -- completely and precisely, never inventing content."
        ),
        backstory=(
            "You are one extractor crewmate in a multi-agent notes "
            "pipeline that turns YouTube videos into study notes. You "
            "see exactly one chunk of the transcript; other extractor "
            "agents handle the remaining chunks in parallel, and a "
            "synthesizer agent later merges everyone's findings into "
            "one outline. Your output is consumed as JSON by machines, "
            "so you respond with schema-exact JSON and nothing else."
        ),
        llm=llm,
        allow_delegation=False,
    )


def _crew_text(result: Any) -> str:
    raw = getattr(result, "raw", None)
    return raw if isinstance(raw, str) else str(result)


def _parse_extraction(raw: str) -> ChunkExtraction:
    """
    Strips accidental markdown code fences, then parses and validates
    against the ChunkExtraction schema. Raises json.JSONDecodeError or
    pydantic.ValidationError, which the retry loop understands.
    """
    match = _FENCE_RE.search(raw)
    cleaned = match.group(1).strip() if match else raw.strip()
    return ChunkExtraction.model_validate(json.loads(cleaned))


def _extract_once(chunk: TranscriptChunk, llm: Any, repair: bool) -> ChunkExtraction:
    agent = _build_agent(llm)
    task = Task(
        description=_build_task_description(chunk.text, repair),
        expected_output=(
            "A single JSON object with one key, topics: a list of objects "
            "each with title, key_points, definitions, examples and "
            "processes. No markdown fences, no text outside the JSON."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    return _parse_extraction(_crew_text(crew.kickoff()))


def extract_chunk(
    chunk: TranscriptChunk,
    llms: Optional[list[tuple[str, Any]]] = None,
) -> ChunkExtractionResult:
    """
    Runs the extractor crew over one chunk and returns its extraction
    with the chunk's identity and timestamp range attached. Invalid
    JSON gets up to MAX_REPAIR_ATTEMPTS stricter retries per key;
    transport errors rotate immediately to the next key.
    """
    candidates = llms if llms is not None else candidate_llms(EXTRACTOR_TEMPERATURE)
    if not candidates:
        raise ExtractionError(
            "No NVIDIA API key is set. Add NVIDIA_API_KEY (and optionally "
            "FALLBACK_NVIDIA_API_KEY) to your .env file."
        )

    last_error: Exception | None = None
    for label, llm in candidates:
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            try:
                extraction = _extract_once(chunk, llm, repair=attempt > 0)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "chunk %d: %s key returned invalid JSON (attempt %d/%d): %s",
                    chunk.index,
                    label,
                    attempt + 1,
                    MAX_REPAIR_ATTEMPTS + 1,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "chunk %d: %s key call failed: %s", chunk.index, label, exc
                )
                break
            else:
                logger.info(
                    "chunk %d: %d topic(s) extracted (%s key)",
                    chunk.index,
                    len(extraction.topics),
                    label,
                )
                return ChunkExtractionResult(
                    chunk_index=chunk.index,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    extraction=extraction,
                )
        logger.warning(
            "chunk %d: %s key exhausted, trying next key", chunk.index, label
        )

    raise ExtractionError(
        f"chunk {chunk.index}: extraction failed on all available keys "
        f"(last error: {last_error})"
    )


def extract_chunks(
    chunks: list[TranscriptChunk],
    max_workers: int = MAX_PARALLEL_CHUNKS,
    llms: Optional[list[tuple[str, Any]]] = None,
) -> list[ChunkExtractionResult]:
    """
    Extracts every chunk in parallel (independent crews on a thread
    pool), returning results in chunk order. Raises ExtractionError
    at the first chunk that fails on every key: a partial notes
    document is worse than a clear failure, since the synthesizer
    downstream assumes complete coverage of the video.
    """
    if not chunks:
        return []
    if llms is None and not candidate_llms(EXTRACTOR_TEMPERATURE):
        raise ExtractionError(
            "No NVIDIA API key is set. Add NVIDIA_API_KEY (and optionally "
            "FALLBACK_NVIDIA_API_KEY) to your .env file."
        )
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(pool.map(lambda chunk: extract_chunk(chunk, llms), chunks))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.extractor <youtube_url>")
        sys.exit(1)

    from ..transcript_pipeline.translation import get_english_transcript
    from .segmentor import chunk_transcript

    transcript = get_english_transcript(sys.argv[1])
    results = extract_chunks(chunk_transcript(transcript))

    for r in results:
        print(f"\n=== Chunk {r.chunk_index} [{r.start_time:.0f}s-{r.end_time:.0f}s] ===")
        for topic in r.extraction.topics:
            print(f"\n  # {topic.title}")
            for point in topic.key_points:
                print(f"    - {point}")
            for d in topic.definitions:
                print(f"    * {d.term}: {d.explanation}")
            for ex in topic.examples:
                print(f"    e.g. {ex}")
            for p in topic.processes:
                print(f"    steps [{p.name}]: " + " -> ".join(p.steps))
