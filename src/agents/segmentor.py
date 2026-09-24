"""
src/agents/segmentor.py

Splits a transcript into chunks sized by *estimated token count*, not
fixed clock time. This replaces the earlier time-boxed (~4 min window)
version now that the pipeline targets NVIDIA NIM's
nvidia/nemotron-3-ultra-550b-a55b (1M token context) instead of Groq
(6K-12K TPM free-tier ceiling).

Why token-based instead of time-based now:
  With Groq, the binding constraint was tokens-per-minute, so a small,
  fixed time window was the safe default regardless of transcript
  density. With a 1M-context model, the binding constraint is "does
  this fit in one call at all" -- a question about total size, not
  elapsed video time. A dense, fast-talking 20-minute video and a
  sparse, slow 20-minute video have very different token counts; a
  token-based threshold adapts to that, a fixed time window doesn't.

Behavior:
  - If the whole transcript fits comfortably under MAX_CHUNK_TOKENS,
    it is returned as a single chunk -- for most realistic videos
    (up to several hours) this means NO chunking happens at all.
  - Only transcripts that exceed the budget get split, into the
    minimum number of roughly equal, segment-boundary-aligned chunks
    needed to fit -- e.g. a huge outlier video gets 2-3 large chunks,
    never 100+ small ones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.transcript_pipeline.transcription import TranscriptResult, TranscriptSegment

# nemotron-3-ultra-550b-a55b's context window is 1M tokens. We target
# well under that (not right up to the edge) to leave headroom for the
# system/instruction prompt, the section titles and extracted notes
# the model has to generate back out, and any additional context
# (carry-forward hints, prior chunk summaries) added later in the
# pipeline. 700K is a deliberately conservative default -- adjust
# down if you see truncated output, or up once you've confirmed your
# actual prompt + output overhead per call.
DEFAULT_MAX_CHUNK_TOKENS = int(os.environ.get("MAX_CHUNK_TOKENS", "700000"))

# Rough words-to-tokens ratio for English text (~1.3 tokens/word is a
# standard estimate for BPE-style tokenizers). This is an approximation
# for chunk-sizing purposes only -- it doesn't need to be exact, it
# needs to keep us safely under the real limit. Swap in the model's
# actual tokenizer here if you want precision instead of a safety margin.
TOKENS_PER_WORD_ESTIMATE = 1.3


@dataclass
class TranscriptChunk:
    index: int          # 0-based chunk position, in order
    start_time: float
    end_time: float
    segments: list[TranscriptSegment]
    estimated_tokens: int

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD_ESTIMATE)


def chunk_transcript(
    result: TranscriptResult, max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
) -> list[TranscriptChunk]:
    """
    Groups result.segments into the minimum number of chunks needed so
    that no chunk's estimated token count exceeds max_chunk_tokens.
    Returns a single chunk (the whole transcript) whenever it already
    fits -- which is the common case at this context-window size.
    """
    if not result.segments:
        return []

    total_tokens = sum(_estimate_tokens(s.text) for s in result.segments)

    if total_tokens <= max_chunk_tokens:
        return [
            TranscriptChunk(
                index=0,
                start_time=result.segments[0].start,
                end_time=result.segments[-1].end,
                segments=result.segments,
                estimated_tokens=total_tokens,
            )
        ]

    # Split into the smallest number of chunks that fits the budget,
    # then divide the transcript as evenly as possible across them --
    # avoids the "3 full chunks + 1 tiny leftover chunk" shape you'd
    # get from greedily filling each chunk to the max.
    num_chunks = -(-total_tokens // max_chunk_tokens)  # ceil division
    target_tokens_per_chunk = total_tokens / num_chunks

    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_tokens = 0

    for seg in result.segments:
        current.append(seg)
        current_tokens += _estimate_tokens(seg.text)

        is_last_segment = seg is result.segments[-1]
        chunks_remaining = num_chunks - len(chunks)
        if (
            not is_last_segment
            and chunks_remaining > 1
            and current_tokens >= target_tokens_per_chunk
        ):
            chunks.append(
                TranscriptChunk(
                    index=len(chunks),
                    start_time=current[0].start,
                    end_time=current[-1].end,
                    segments=current,
                    estimated_tokens=current_tokens,
                )
            )
            current = []
            current_tokens = 0

    if current:
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                start_time=current[0].start,
                end_time=current[-1].end,
                segments=current,
                estimated_tokens=current_tokens,
            )
        )

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.segmentor <youtube_url>")
        sys.exit(1)

    from src.transcript_pipeline.translation import get_english_transcript

    transcript = get_english_transcript(sys.argv[1])
    chunks = chunk_transcript(transcript)

    print(f"\n{len(chunks)} chunk(s) (budget: {DEFAULT_MAX_CHUNK_TOKENS:,} tokens each):\n")
    for c in chunks:
        print(
            f"  [{c.start_time:7.1f}s - {c.end_time:7.1f}s] "
            f"~{c.estimated_tokens:,} tokens, {len(c.segments)} segments"
        )