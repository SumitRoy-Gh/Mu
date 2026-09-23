"""
transcript_pipeline: YouTube URL -> English transcript (with timestamps).

Stage 1 -- transcription.py
    Layered fallback: free YouTube captions -> TranscriptAPI (paid,
    1 credit per success) -> Groq Whisper audio translation. Entry:
    get_transcript(url).

Stage 2 -- translation.py
    Guarantees English output. English transcripts pass through at
    zero cost; non-English transcripts are translated by a Groq LLM
    with segment timestamps preserved. Entry: get_english_transcript(url)
    (the combined pipeline the rest of the app should call).
"""

from .transcription import (
    get_transcript,
    extract_video_id,
    TranscriptResult,
    TranscriptSegment,
    TranscriptSource,
    TranscriptionError,
)
from .translation import (
    ensure_english,
    get_english_transcript,
    is_english_language,
    TranslationError,
)

__all__ = [
    "get_transcript",
    "extract_video_id",
    "TranscriptResult",
    "TranscriptSegment",
    "TranscriptSource",
    "TranscriptionError",
    "ensure_english",
    "get_english_transcript",
    "is_english_language",
    "TranslationError",
]
