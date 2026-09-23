"""
transcription.py

Turns a YouTube URL into transcript text with timestamped segments,
using a layered fallback strategy that avoids expensive Whisper
inference whenever possible.

Fallback chain (cheapest / fastest first):
  1. YouTube's own caption tracks, via youtube-transcript-api (free).
     - If an English track exists, use it directly (near-instant).
     - If only non-English tracks exist, ask the API for YouTube's own
       translation of that track to English (still free, near-instant).
  2. TranscriptAPI (https://transcriptapi.com) -- paid, 1 credit per
     successful request. Absorbs the IP-blocking that hits
     youtube-transcript-api at scale, plus caption edge cases.
  3. Audio download (yt-dlp) + Groq cloud Whisper (whisper-large-v3)
     via the translations endpoint, which outputs English regardless of
     the source language. Only fires when the video has no captions at
     all. Audio is chunked into ~20 MB pieces to respect Groq's 25 MB
     free-tier upload cap and keep disk usage bounded on small
     free-tier deployments (e.g. Render).

Language reporting:
  TranscriptResult.detected_language always describes the language of
  the returned *text*: "en" for native and YouTube-translated captions,
  the TranscriptAPI-resolved code (e.g. "hi", "asr-en") for layer 2,
  and "en" for Whisper translation output. The translation stage
  (src.transcript_pipeline/translation.py) uses it to decide whether
  an LLM translation pass is needed -- no language classifier is
  required.

API keys (read from .env via python-dotenv):
  Layer 2 reads Youtube_to_Transcript_API_KEY.
  Layer 3 reads GROQ_API_KEY (primary) and FALLBACK_GROQ_API_KEY --
  the same request is retried with the fallback key on 429/401.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import requests
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

load_dotenv()

logger = logging.getLogger("transcription_service")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

TRANSCRIPTAPI_KEY = os.environ.get("Youtube_to_Transcript_API_KEY", "")
TRANSCRIPTAPI_URL = "https://transcriptapi.com/api/v2/youtube/transcript"
TRANSCRIPTAPI_TIMEOUT = int(os.environ.get("TRANSCRIPTAPI_TIMEOUT", "30"))
TRANSCRIPTAPI_MAX_RETRIES = int(os.environ.get("TRANSCRIPTAPI_MAX_RETRIES", "2"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
FALLBACK_GROQ_API_KEY = os.environ.get("FALLBACK_GROQ_API_KEY", "")
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")

# Groq free tier caps uploads at 25 MB. We target ~20 MB chunks to leave
# a comfortable margin for container overhead and encoding variance.
MAX_CHUNK_SIZE_MB = int(os.environ.get("MAX_CHUNK_SIZE_MB", "20"))

# Cap video duration we'll ever attempt. With Groq's cloud inference the
# bottleneck is download + upload time, not CPU, so we can be generous.
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", str(30 * 60 * 60)))  # 30h

# TranscriptAPI statuses that are worth retrying (per their docs):
# 408 timeout / bot-detection, 429 rate limit, 503 unavailable.
_RETRYABLE_STATUSES = {408, 429, 503}


class TranscriptSource(str, Enum):
    CAPTIONS_NATIVE_EN = "captions_native_en"
    CAPTIONS_TRANSLATED = "captions_translated"
    TRANSCRIPT_API = "transcript_api"
    WHISPER_TRANSLATED = "whisper_translated"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    segments: list[TranscriptSegment]
    source: TranscriptSource
    detected_language: Optional[str]  # language of `text`, e.g. "en", "hi", "asr-en"
    video_id: str
    was_translated: bool = False  # set by the translation stage, if it ran


class TranscriptionError(Exception):
    """Raised when every fallback in the chain has been exhausted."""


# --------------------------------------------------------------------------
# Step 0: URL parsing
# --------------------------------------------------------------------------

_YT_ID_PATTERNS = (
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"),
)


def extract_video_id(url: str) -> str:
    for pattern in _YT_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract a YouTube video id from: {url!r}")


# --------------------------------------------------------------------------
# Layer 1: caption-based fast path (free, no audio download at all)
# --------------------------------------------------------------------------

def _try_captions(video_id: str) -> Optional[TranscriptResult]:
    """
    Attempts to get English text straight from YouTube's caption tracks.
    Returns None (never raises for the "no captions" case) so the caller
    can fall through to the next layer cleanly. Only raises for truly
    unexpected errors, which the caller also catches defensively.
    """
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
    except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound):
        logger.info("video_id=%s no caption tracks available", video_id)
        return None
    except Exception:
        # Includes IpBlocked / RequestBlocked / PoTokenRequired -- exactly
        # the failures the paid TranscriptAPI layer exists to absorb.
        logger.exception("video_id=%s unexpected error listing transcripts", video_id)
        return None

    # Prefer a manually-created or auto-generated English track if one exists.
    try:
        transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        source = TranscriptSource.CAPTIONS_NATIVE_EN
        detected_language = "en"
    except NoTranscriptFound:
        # No English track. Grab whatever track exists and ask YouTube's
        # own translation layer for English -- this is still just an API
        # call against YouTube, no audio download, no Whisper.
        try:
            available = next(iter(transcript_list))
        except StopIteration:
            return None

        if not available.is_translatable:
            logger.info(
                "video_id=%s only non-English, non-translatable track found "
                "(lang=%s) -- falling through",
                video_id,
                available.language_code,
            )
            return None

        try:
            transcript = available.translate("en")
            source = TranscriptSource.CAPTIONS_TRANSLATED
            # The text is English after YouTube's translation layer.
            detected_language = "en"
            logger.info(
                "video_id=%s using YouTube's own translation of the %s caption track",
                video_id,
                available.language_code,
            )
        except Exception:
            logger.exception(
                "video_id=%s translation of caption track failed", video_id
            )
            return None

    try:
        raw = transcript.fetch()
    except Exception:
        logger.exception("video_id=%s fetching caption track failed", video_id)
        return None

    # v1.x of youtube-transcript-api returns dataclass snippets with
    # .text/.start/.duration attributes.
    segments = [
        TranscriptSegment(start=item.start, end=item.start + item.duration, text=item.text)
        for item in raw
    ]
    full_text = " ".join(s.text for s in segments).strip()

    if not full_text:
        return None

    return TranscriptResult(
        text=full_text,
        segments=segments,
        source=source,
        detected_language=detected_language,
        video_id=video_id,
    )


# --------------------------------------------------------------------------
# Layer 2: TranscriptAPI (paid, 1 credit per success)
# --------------------------------------------------------------------------

def _error_detail(response: requests.Response) -> str:
    """Extracts a human-readable message from a TranscriptAPI error body."""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:200]
    detail = payload.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail or payload)[:200]


def _retry_delay_seconds(response: Optional[requests.Response], attempt: int) -> float:
    """Respects the Retry-After header when present, else exponential backoff."""
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass
    return float(2 ** attempt)


def _parse_transcriptapi_payload(
    video_id: str, payload: dict
) -> Optional[TranscriptResult]:
    raw = payload.get("transcript")
    if not isinstance(raw, list) or not raw:
        logger.warning("video_id=%s TranscriptAPI returned no transcript segments", video_id)
        return None

    segments: list[TranscriptSegment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        start = float(item.get("start") or 0.0)
        duration = float(item.get("duration") or 0.0)
        segments.append(TranscriptSegment(start=start, end=start + duration, text=text))

    if not segments:
        logger.warning("video_id=%s TranscriptAPI transcript had no usable text", video_id)
        return None

    return TranscriptResult(
        text=" ".join(s.text for s in segments),
        segments=segments,
        source=TranscriptSource.TRANSCRIPT_API,
        detected_language=payload.get("language"),
        video_id=video_id,
    )


def _try_transcriptapi(video_id: str) -> Optional[TranscriptResult]:
    """
    Fetches the transcript via TranscriptAPI. Returns None on any
    failure so the caller falls through to the Whisper layer; details
    are logged. Retries transient failures (network errors, 408/429/503)
    with exponential backoff, honouring the Retry-After header.
    """
    if not TRANSCRIPTAPI_KEY:
        logger.warning(
            "Youtube_to_Transcript_API_KEY is not set -- skipping TranscriptAPI layer"
        )
        return None

    headers = {"Authorization": f"Bearer {TRANSCRIPTAPI_KEY}"}
    params = {"video_url": video_id, "format": "json", "include_timestamp": "true"}

    max_attempts = TRANSCRIPTAPI_MAX_RETRIES + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                TRANSCRIPTAPI_URL,
                params=params,
                headers=headers,
                timeout=TRANSCRIPTAPI_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt < max_attempts:
                delay = _retry_delay_seconds(None, attempt)
                logger.warning(
                    "video_id=%s TranscriptAPI network error (attempt %d/%d): %s "
                    "-- retrying in %.0fs",
                    video_id, attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
                continue
            logger.warning(
                "video_id=%s TranscriptAPI network error, retries exhausted: %s",
                video_id, exc,
            )
            return None

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                logger.warning(
                    "video_id=%s TranscriptAPI returned invalid JSON", video_id
                )
                return None
            return _parse_transcriptapi_payload(video_id, payload)

        if response.status_code in _RETRYABLE_STATUSES and attempt < max_attempts:
            delay = _retry_delay_seconds(response, attempt)
            logger.warning(
                "video_id=%s TranscriptAPI HTTP %d (attempt %d/%d) "
                "-- retrying in %.0fs",
                video_id, response.status_code, attempt, max_attempts, delay,
            )
            time.sleep(delay)
            continue

        # Non-retryable (400/401/402/404/422) or retries exhausted:
        # log loudly, then fall through to Whisper. 402 = out of credits,
        # 401 = bad key, 404 = no captions anywhere -- the last one is
        # exactly what the Whisper layer exists for.
        logger.warning(
            "video_id=%s TranscriptAPI failed: HTTP %d -- %s",
            video_id, response.status_code, _error_detail(response),
        )
        return None

    return None


# --------------------------------------------------------------------------
# Layer 3: Groq Whisper fallback (audio download + cloud inference)
# --------------------------------------------------------------------------

@contextmanager
def _downloaded_audio(video_id: str) -> Iterator[str]:
    """
    Downloads the audio-only stream as a compressed mp3 to a temp directory.
    Yields the path to the downloaded mp3 file.
    Cleanup is guaranteed by TemporaryDirectory on context exit.
    """
    import yt_dlp

    with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmp_dir:
        out_template = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "worstaudio/bestaudio",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "5",
                }
            ],
            # Downsample to 16kHz mono -- Whisper's native format.
            # Cuts file size dramatically for long videos.
            "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        }

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise TranscriptionError(f"Audio download failed for {video_id}: {exc}") from exc

        duration = info.get("duration") or 0
        if duration and duration > MAX_AUDIO_SECONDS:
            raise TranscriptionError(
                f"Video duration {duration}s exceeds MAX_AUDIO_SECONDS="
                f"{MAX_AUDIO_SECONDS}s limit for this deployment."
            )

        # Find the mp3 file yt-dlp produced
        mp3_path = os.path.join(tmp_dir, f"{info['id']}.mp3")
        if not os.path.exists(mp3_path):
            candidates = [f for f in os.listdir(tmp_dir) if f.endswith(".mp3")]
            if not candidates:
                raise TranscriptionError("Audio postprocessing did not produce an .mp3 file.")
            mp3_path = os.path.join(tmp_dir, candidates[0])

        yield mp3_path


def _ffprobe_duration_seconds(audio_path: str) -> float:
    """Returns the audio file's duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _split_audio_into_chunks(audio_path: str, tmp_dir: str) -> list[str]:
    """
    Splits an audio file into chunks of approximately MAX_CHUNK_SIZE_MB
    each. Returns a list of file paths to the chunk files.

    Chunking serves two constraints at once: Groq's 25 MB upload cap on
    the free tier, and the small ephemeral disks of free-tier deployments
    (Render etc.) when long videos would otherwise produce very large
    audio files.

    Splitting is done by streaming through ffmpeg (never loading the
    file into RAM), estimating how many seconds of audio fit into
    MAX_CHUNK_SIZE_MB based on the file's actual bitrate.
    """
    file_size_bytes = os.path.getsize(audio_path)

    if file_size_bytes <= MAX_CHUNK_SIZE_MB * 1024 * 1024:
        # File is already small enough, no splitting needed
        logger.info(
            "audio file %.1f MB, under %d MB limit -- no chunking needed",
            file_size_bytes / (1024 * 1024),
            MAX_CHUNK_SIZE_MB,
        )
        return [audio_path]

    duration_seconds = _ffprobe_duration_seconds(audio_path)

    # Calculate how many seconds of audio fit in one chunk
    bytes_per_second = file_size_bytes / duration_seconds
    seconds_per_chunk = (MAX_CHUNK_SIZE_MB * 1024 * 1024) / bytes_per_second
    num_chunks = math.ceil(duration_seconds / seconds_per_chunk)

    logger.info(
        "audio file %.1f MB (%.0fs), splitting into %d chunks of ~%.0fs each",
        file_size_bytes / (1024 * 1024),
        duration_seconds,
        num_chunks,
        seconds_per_chunk,
    )

    chunk_paths: list[str] = []
    for i in range(num_chunks):
        start = i * seconds_per_chunk
        end = min((i + 1) * seconds_per_chunk, duration_seconds)
        chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.mp3")

        # Stream-copy the slice; mp3 frame granularity (~26 ms) is plenty
        # accurate for transcript stitching, and -c copy is near-instant.
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", audio_path,
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-c", "copy",
                chunk_path,
            ],
            check=True,
        )
        chunk_paths.append(chunk_path)
        logger.info(
            "chunk %d/%d exported: %.1f MB",
            i + 1,
            num_chunks,
            os.path.getsize(chunk_path) / (1024 * 1024),
        )

    return chunk_paths


def _groq_transcribe_chunk(chunk_path: str, api_key: str):
    """
    Sends a single audio chunk to Groq's Whisper *translations* endpoint,
    which returns English text regardless of the source language.
    Returns the raw API response object (verbose_json: .text, .segments,
    .duration).
    """
    from groq import Groq

    client = Groq(api_key=api_key)

    with open(chunk_path, "rb") as audio_file:
        response = client.audio.translations.create(
            model=GROQ_WHISPER_MODEL,
            file=audio_file,
            response_format="verbose_json",
        )

    return response


def _transcribe_chunk_with_fallback(chunk_path: str, chunk_index: int):
    """
    Tries translating a chunk with the primary API key first. If it hits
    a rate-limit (429) or auth error (401), retries with the fallback key.
    """
    from groq import APIStatusError

    keys = [(GROQ_API_KEY, "primary"), (FALLBACK_GROQ_API_KEY, "fallback")]

    for api_key, label in keys:
        if not api_key:
            continue
        try:
            logger.info("chunk %d: attempting Groq translation with %s key", chunk_index, label)
            result = _groq_transcribe_chunk(chunk_path, api_key)
            logger.info("chunk %d: success with %s key", chunk_index, label)
            return result
        except APIStatusError as exc:
            if exc.status_code in (401, 429) and label == "primary" and FALLBACK_GROQ_API_KEY:
                logger.warning(
                    "chunk %d: %s key returned HTTP %d, falling back to %s key",
                    chunk_index,
                    label,
                    exc.status_code,
                    "fallback",
                )
                continue
            raise TranscriptionError(
                f"Groq API error for chunk {chunk_index}: HTTP {exc.status_code} - {exc.message}"
            ) from exc
        except Exception as exc:
            raise TranscriptionError(
                f"Groq API call failed for chunk {chunk_index}: {exc}"
            ) from exc

    raise TranscriptionError("No valid GROQ API keys configured. Check your .env file.")


def _try_whisper(video_id: str) -> TranscriptResult:
    """
    Last-resort path for videos with no captions at all: download audio,
    chunk it, and translate each chunk to English via Groq's cloud Whisper
    API (translations endpoint, so the output is English by construction).
    Raises TranscriptionError on any failure -- there's nothing left to
    fall back to after this.
    """
    with _downloaded_audio(video_id) as audio_path:
        tmp_dir = os.path.dirname(audio_path)
        chunk_paths = _split_audio_into_chunks(audio_path, tmp_dir)

        all_segments: list[TranscriptSegment] = []
        all_text_parts: list[str] = []
        time_offset = 0.0  # cumulative offset for stitching chunk timestamps
        source_language: Optional[str] = None

        for i, chunk_path in enumerate(chunk_paths):
            response = _transcribe_chunk_with_fallback(chunk_path, i)

            reported_language = getattr(response, "language", None)
            if reported_language:
                source_language = reported_language

            chunk_text = response.text if hasattr(response, "text") else ""
            if chunk_text:
                all_text_parts.append(chunk_text.strip())

            # Extract segments with timestamps if available
            chunk_segments = getattr(response, "segments", None) or []
            for seg in chunk_segments:
                seg_start = getattr(seg, "start", 0.0) or 0.0
                seg_end = getattr(seg, "end", 0.0) or 0.0
                seg_text = getattr(seg, "text", "") or ""
                all_segments.append(
                    TranscriptSegment(
                        start=seg_start + time_offset,
                        end=seg_end + time_offset,
                        text=seg_text.strip(),
                    )
                )

            # Calculate this chunk's duration so the next chunk's timestamps
            # are offset correctly
            chunk_duration = getattr(response, "duration", None)
            if chunk_duration:
                time_offset += chunk_duration
            elif chunk_segments:
                # Fallback: use the last segment's end time as chunk duration
                last_seg = chunk_segments[-1]
                time_offset += getattr(last_seg, "end", 0.0) or 0.0

    full_text = " ".join(all_text_parts).strip()
    if not full_text:
        raise TranscriptionError(f"Groq Whisper produced empty output for video_id={video_id}")

    if source_language:
        logger.info(
            "video_id=%s Whisper detected source language: %s (output is English)",
            video_id,
            source_language,
        )

    return TranscriptResult(
        text=full_text,
        segments=all_segments,
        source=TranscriptSource.WHISPER_TRANSLATED,
        detected_language="en",  # translations endpoint always outputs English
        video_id=video_id,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def get_transcript(url: str) -> TranscriptResult:
    """
    The one function the transcription stage exposes. Runs the full
    fallback chain (captions -> TranscriptAPI -> Whisper) and returns as
    soon as one layer succeeds.

    Note: the returned text may be non-English (only possible via the
    TranscriptAPI layer). Use src.transcript_pipeline.ensure_english, or
    the combined pipeline src.transcript_pipeline.get_english_transcript,
    to guarantee English output.
    """
    video_id = extract_video_id(url)
    logger.info("video_id=%s starting transcript resolution", video_id)

    captions_result = _try_captions(video_id)
    if captions_result is not None:
        logger.info(
            "video_id=%s resolved via %s (free, no Whisper needed)",
            video_id,
            captions_result.source.value,
        )
        return captions_result

    logger.info("video_id=%s captions unavailable -- trying TranscriptAPI", video_id)
    api_result = _try_transcriptapi(video_id)
    if api_result is not None:
        logger.info(
            "video_id=%s resolved via TranscriptAPI (language=%s)",
            video_id,
            api_result.detected_language,
        )
        return api_result

    logger.info("video_id=%s TranscriptAPI unavailable -- falling back to Groq Whisper", video_id)
    return _try_whisper(video_id)
