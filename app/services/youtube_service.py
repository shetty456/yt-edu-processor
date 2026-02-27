"""
youtube_service.py
──────────────────
Uses Agno's YouTubeTools directly — no paid YouTube API, no yt-dlp.

How Agno YouTubeTools works under the hood:
  get_youtube_video_data(url)     → calls YouTube's FREE oEmbed endpoint
                                    returns: title, author_name, thumbnail_url
                                    (no duration — oEmbed doesn't provide it)
  get_youtube_video_captions(url) → calls youtube_transcript_api (FREE library)
                                    returns: plain-text transcript string

Duration is computed from the last transcript snippet's timing data,
which youtube_transcript_api gives us for free as well.

All blocking calls run in asyncio's thread-pool executor so the
FastAPI event loop is never blocked.
"""
from __future__ import annotations

import asyncio
import json
import re
from functools import partial

from agno.tools.youtube import YouTubeTools
from youtube_transcript_api import YouTubeTranscriptApi

from app.config import get_settings
from app.schemas import VideoMeta
from app.utils import logger

settings = get_settings()
_yt = YouTubeTools()   # stateless — safe to share across requests

# ── Educational keyword check ─────────────────────────────────────────────────
_EDU_KEYWORDS = frozenset({
    "tutorial", "course", "lesson", "lecture", "learn", "learning",
    "study", "explained", "explanation", "how to", "how-to", "guide",
    "introduction", "beginner", "advanced", "fundamentals", "basics",
    "programming", "coding", "software", "algorithm", "data structure",
    "math", "mathematics", "calculus", "statistics",
    "physics", "chemistry", "biology", "neuroscience",
    "history", "economics", "finance", "investing",
    "science", "engineering", "machine learning", "deep learning",
    "artificial intelligence", "psychology", "philosophy",
    "workshop", "training", "bootcamp", "masterclass", "crash course",
    "mit", "stanford", "harvard", "coursera", "khan academy", "edx",
    "university", "college", "professor", "dr.", "phd",
    "chapter", "module", "documentary", "analysis", "breakdown",
})


def _is_educational(title: str, author: str) -> bool:
    combined = f"{title} {author}".lower()
    return any(kw in combined for kw in _EDU_KEYWORDS)


# ── Transcript cleaning ───────────────────────────────────────────────────────

def _clean(raw: str) -> str:
    """
    YouTubeTools returns plain joined text from caption snippets.
    We strip noise artefacts and deduplicate repeated phrases.
    """
    text = re.sub(r"\[[^\]]{1,40}\]", " ", raw)    # [Music], [Applause]
    text = re.sub(r"\([^)]{1,40}\)", " ", text)    # (inaudible)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove repeated 8-word n-grams (auto-caption loop artefact)
    words = text.split()
    out: list[str] = []
    N = 8
    for i, w in enumerate(words):
        if i < N:
            out.append(w)
            continue
        ngram = " ".join(words[i: i + N])
        if ngram in " ".join(out[-N * 4:]):
            continue
        out.append(w)

    text = " ".join(out)
    text = re.sub(r" ([.,!?;:])", r"\1", text)
    return text.strip()


# ── Blocking helpers (run in executor) ───────────────────────────────────────

def _get_metadata(url: str) -> dict:
    """YouTubeTools → YouTube oEmbed (free, no API key)."""
    raw = _yt.get_youtube_video_data(url)
    if not raw or raw.startswith("Error"):
        raise ValueError(f"Could not fetch video metadata: {raw}")
    return json.loads(raw)


def _get_captions(url: str) -> str:
    """YouTubeTools → youtube_transcript_api (free Python library)."""
    result = _yt.get_youtube_video_captions(url)
    if not result or result.startswith("Error") or result == "No captions found for video":
        raise ValueError(
            f"No captions available: {result}. "
            "The video may not have English captions."
        )
    return result


def _get_duration(video_id: str) -> int:
    """
    Compute duration from last transcript snippet timing.
    Free — uses youtube_transcript_api, same library YouTubeTools wraps.
    Returns 0 if unavailable (duration check becomes a no-op).
    """
    try:
        snippets = YouTubeTranscriptApi().fetch(video_id)
        if snippets:
            last = snippets[-1]
            return int(last.start + last.duration)
    except Exception:
        pass
    return 0


# ── Public entry point ────────────────────────────────────────────────────────

async def extract_video(youtube_url: str) -> VideoMeta:
    log  = logger.bind(url=youtube_url)
    loop = asyncio.get_running_loop()

    # 1. Parse video ID (uses YouTubeTools built-in parser)
    video_id = _yt.get_youtube_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"Cannot parse a video ID from: {youtube_url}")

    # 2. Metadata  (title, author via oEmbed — free)
    log.info("yt_fetching_metadata")
    meta = await asyncio.wait_for(
        loop.run_in_executor(None, partial(_get_metadata, youtube_url)), timeout=30
    )
    title  = meta.get("title")       or f"Video {video_id}"
    author = meta.get("author_name") or ""
    log.info("yt_metadata_ok", title=title)

    # 3. Duration  (from transcript snippet timing — free)
    log.info("yt_fetching_duration")
    try:
        duration = await asyncio.wait_for(
            loop.run_in_executor(None, partial(_get_duration, video_id)), timeout=30
        )
    except Exception:
        duration = 0
        log.warning("yt_duration_unavailable")

    # 4. Validate duration
    if duration > 0 and duration > settings.max_video_duration_seconds:
        raise ValueError(
            f"Video is {duration / 3600:.1f} h long. Max supported is 2 hours."
        )

    # 5. Educational check
    if not _is_educational(title, author):
        raise ValueError(
            "Video does not appear to be educational. "
            "Only tutorials, lectures, courses, and similar content are supported."
        )

    # 6. Transcript  (via YouTubeTools → youtube_transcript_api — free)
    log.info("yt_fetching_captions")
    raw = await asyncio.wait_for(
        loop.run_in_executor(None, partial(_get_captions, youtube_url)), timeout=60
    )

    # 7. Clean
    transcript = _clean(raw)
    word_count = len(transcript.split())
    if word_count < 50:
        raise ValueError(f"Transcript too short after cleaning ({word_count} words).")

    log.info("yt_done", word_count=word_count, duration_s=duration)
    return VideoMeta(
        title=title,
        duration_seconds=duration,
        transcript=transcript,
        word_count=word_count,
    )