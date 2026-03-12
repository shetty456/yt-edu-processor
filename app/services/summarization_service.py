from __future__ import annotations

import asyncio
import json
import re
import logging
from typing import List, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import get_settings
from app.schemas import ChunkSummary, MergedSummary
from app.utils import get_sarvam_client, logger, strip_think


settings = get_settings()
_client = get_sarvam_client()


# ── Utility functions ──────────────────────────────────────────────────────────

def _retry(n: int = 3):
    return retry(
        stop=stop_after_attempt(n),
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
        before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
        reraise=True,
    )


def _strip(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    brace = text.find("{")
    bracket = text.find("[")
    if brace >= 0 and (bracket < 0 or brace <= bracket):
        text = text[brace:]
    elif bracket >= 0:
        text = text[bracket:]
    rbrace = text.rfind("}")
    rbracket = text.rfind("]")
    end = max(rbrace, rbracket)
    if end >= 0 and end < len(text) - 1:
        text = text[:end + 1]
    return text.strip()


# ── 1. Chunking ───────────────────────────────────────────────────────────────

def chunk_transcript(
    text: str,
    word_limit: int | None = None,
    target_words: int | None = None,
) -> List[str]:
    limit = word_limit if word_limit is not None else settings.chunk_word_limit
    target = target_words if target_words is not None else settings.chunk_target_words
    words = text.split()
    if len(words) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    wc = 0
    for s in sentences:
        sw = len(s.split())
        if wc + sw > target and current:
            chunks.append(" ".join(current))
            current, wc = [s], sw
        else:
            current.append(s)
            wc += sw
    if current:
        chunks.append(" ".join(current))

    logger.info("chunked", total=len(chunks))
    return chunks


# ── 2. Per-chunk summary ──────────────────────────────────────────────────────

_CHUNK_SYS = """\
You are an expert educational content analyst.

RULES:
1. Every claim must come directly from the transcript chunk — do NOT invent anything.
2. If a category has nothing suitable in the chunk, return an empty list [].
3. Output ONLY the raw JSON object — no markdown fences, no commentary.
"""

_CHUNK_USR = """\
Analyse this transcript chunk ({words} words) and return ONLY this JSON:

{{
  "core_concepts":       ["Concept: 1-2 sentence explanation"],
  "important_examples":  ["Concrete example or analogy from the text"],
  "key_points":          ["Specific, memorable insight from this chunk"],
  "definitions":         ["Term: its definition as given in the transcript"]
}}

Targets: core_concepts 3-7, important_examples 0-5, key_points 4-10, definitions 0-8.
Empty list is fine if the category has no material in this chunk.

CHUNK:
\"\"\"
{chunk}
\"\"\"
"""


@_retry()
async def summarise_chunk(chunk: str, idx: int, language: str = "English") -> ChunkSummary:
    log = logger.bind(chunk=idx, words=len(chunk.split()))
    log.info("chunk_start")
    sys_prompt = _CHUNK_SYS
    if language != "English":
        sys_prompt += f"\nExtract and write all content in {language}."
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.4,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": _CHUNK_USR.format(chunk=chunk, words=len(chunk.split()))},
        ],
    )
    content = resp.choices[0].message.content or ""
    raw = _strip(strip_think(content))
    if not raw:
        # Model put JSON inside <think> block — extract brace content directly
        raw = _strip(content)
    if not raw:
        raise ValueError("Empty response from model")
    result = ChunkSummary(**json.loads(raw))
    log.info("chunk_done", key_points=len(result.key_points))
    return result


# ── 3. Merge ──────────────────────────────────────────────────────────────────

def merge_summaries(summaries: List[ChunkSummary]) -> MergedSummary:
    def dedup(items: List[str]) -> List[str]:
        seen: dict[str, str] = {}
        for item in items:
            key = item.split(":")[0].strip().lower()[:40]
            if key not in seen or len(item) > len(seen[key]):
                seen[key] = item
        return sorted(seen.values())

    return MergedSummary(
        core_concepts=dedup([c for s in summaries for c in s.core_concepts]),
        important_examples=dedup([e for s in summaries for e in s.important_examples]),
        key_points=dedup([p for s in summaries for p in s.key_points]),
        definitions=dedup([d for s in summaries for d in s.definitions]),
    )


# ── 4. Notes generation ──────────────────────────────────────────────────────

_NOTES_SYS = """\
You are a thoughtful writer and educator. Your job is to turn a structured concept summary
into a clear, engaging blog-style article that a curious reader can enjoy and learn from.

RULES:
1. Write in second person, active voice, present tense.
2. Use flowing prose within sections — not just bullet dumps.
3. Be specific and concrete. If a concept needs an example to land, include one.
4. Do NOT reference "the video", "the speaker", "the transcript", or "the notes".
5. Write as if you are explaining these ideas directly to the reader.
"""

_NOTES_USR = """\
Write a blog-style article titled "{title}".

Source material (structured concept extraction):
{summary}

Use EXACTLY these four H2 headings — write naturally under each:

## Overview
(What is this topic about and why does it matter? Write 3-5 substantive paragraphs.
Set the stage, give the reader a reason to care, and provide enough background
that someone unfamiliar with the subject can follow what comes next.)

## Key Ideas
(Explain each core concept with clarity and depth. Use a sub-heading per concept.
Under each sub-heading: define it, explain why it matters, and give a concrete
example or analogy that makes it stick. Aim for 2-4 sentences per concept minimum.)

## In Practice
(Translate the ideas into real-world application. Use 3-4 paragraphs or concrete
scenarios. What changes for a person who genuinely understands this? What mistakes
do people make who don't? Give vivid, grounded examples.)

## Worth Knowing
(Go deeper — nuances, surprising context, common misconceptions corrected,
historical background, or an illustrative story. 2-4 paragraphs. This section
should make the reader feel they got more than they expected.)
"""


@_retry()
async def generate_notes(merged: MergedSummary, title: str, language: str = "English") -> str:
    logger.info("notes_start")
    summary_text = json.dumps(merged.model_dump(), indent=2)

    sys_prompt = _NOTES_SYS
    if language != "English":
        sys_prompt += (
            f"\nRespond entirely in {language}. Write all prose, sub-headings, and explanations in {language}. "
            f"The four H2 section headings (## Overview, ## Key Ideas, ## In Practice, ## Worth Knowing) "
            f"must remain in English exactly as written."
        )
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.4,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": _NOTES_USR.format(title=title, summary=summary_text)},
        ],
    )

    content = resp.choices[0].message.content or ""
    notes = strip_think(content)
    if not notes and content.strip():
        inner = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        notes = inner.group(1).strip() if inner else content.strip()
    notes = re.sub(r"^```(?:markdown|md)?\s*\n?", "", notes.strip())
    notes = re.sub(r"\n?```\s*$", "", notes).strip()
    logger.info("notes_done", chars=len(notes))
    return notes


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_summarisation(
    transcript: str,
    title: str,
    chunk_word_limit: int | None = None,
    chunk_target_words: int | None = None,
    language: str = "English",
) -> Tuple[MergedSummary, str]:
    chunks = chunk_transcript(transcript, chunk_word_limit, chunk_target_words)
    summaries = await asyncio.gather(
        *[summarise_chunk(c, i, language=language) for i, c in enumerate(chunks)]
    )
    merged = merge_summaries(list(summaries))
    notes = await generate_notes(merged, title, language=language)
    return merged, notes