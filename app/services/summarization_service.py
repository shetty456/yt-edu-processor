"""
summarization_service.py
─────────────────────────
1. chunk_transcript   → list[str]
2. summarise_chunk    → ChunkSummary  (Sarvam-m, temp 0.4)
3. merge_summaries    → MergedSummary
4. generate_notes     → str (markdown)
5. extract_objectives → list[str]
"""
from __future__ import annotations

import json
import re
import textwrap
import logging

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import get_settings
from app.schemas import ChunkSummary, MergedSummary
from app.utils import get_sarvam_client, logger

settings = get_settings()
_client = get_sarvam_client()


def _retry(n: int = 3):
    return retry(
        stop=stop_after_attempt(n),
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
        before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
        reraise=True,
    )


def _strip(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    return re.sub(r"\n?```\s*$", "", text).strip()


# ── 1. Chunking ───────────────────────────────────────────────────────────────

def chunk_transcript(text: str) -> list[str]:
    words = text.split()
    if len(words) <= settings.chunk_word_limit:
        return [text]

    # Split on sentence endings, then pack into chunks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    wc = 0
    for s in sentences:
        sw = len(s.split())
        if wc + sw > settings.chunk_target_words and current:
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
async def summarise_chunk(chunk: str, idx: int) -> ChunkSummary:
    log = logger.bind(chunk=idx, words=len(chunk.split()))
    log.info("chunk_start")
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _CHUNK_SYS},
            {"role": "user", "content": _CHUNK_USR.format(chunk=chunk, words=len(chunk.split()))},
        ],
    )
    raw = _strip(resp.choices[0].message.content or "")
    result = ChunkSummary(**json.loads(raw))
    log.info("chunk_done", key_points=len(result.key_points))
    return result


# ── 3. Merge ──────────────────────────────────────────────────────────────────

def merge_summaries(summaries: list[ChunkSummary]) -> MergedSummary:
    def dedup(items: list[str]) -> list[str]:
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


# ── 4. Final notes ────────────────────────────────────────────────────────────

_NOTES_SYS = """\
You are a master educator writing high-quality study notes.

RULES:
1. Use ONLY information from the structured summary — do NOT add outside knowledge.
2. If a section has no material, write one sentence saying so — do NOT fabricate.
3. Write in second person, active voice, present tense.
4. Be specific and concrete — never vague.
"""

_NOTES_USR = """\
Write deep study notes for: "{title}"

Use ONLY the structured summary below as your source.

{summary}

Use EXACTLY these H2 headings (with emoji):

## 🎯 Big Picture Overview
## 🧠 Core Concepts Explained
## ⚡ Key Points (Quick Revision)
## 📖 Stories / Anecdotes / Examples
## 🛠️ Practical Applications
## ⚠️ Common Mistakes
## 🏁 Final Takeaways
"""

_REQUIRED_SECTIONS = [
    "Big Picture Overview", "Core Concepts Explained", "Key Points",
    "Stories / Anecdotes", "Practical Applications", "Common Mistakes", "Final Takeaways",
]


@_retry()
async def generate_notes(merged: MergedSummary, title: str) -> str:
    logger.info("notes_start")
    summary_text = json.dumps(merged.model_dump(), indent=2)
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _NOTES_SYS},
            {"role": "user",   "content": _NOTES_USR.format(title=title, summary=summary_text)},
        ],
    )
    notes = (resp.choices[0].message.content or "").strip()
    missing = [s for s in _REQUIRED_SECTIONS if s not in notes]
    if missing:
        raise ValueError(f"Notes missing sections: {missing}")
    logger.info("notes_done", chars=len(notes))
    return notes


# ── 5. Learning objectives ────────────────────────────────────────────────────

_OBJ_SYS = """\
You write precise, measurable learning objectives using Bloom's Taxonomy verbs
(Define, Explain, Apply, Analyse, Compare, Implement, Evaluate…).

Output ONLY a JSON array of strings — no other text, no fences.
"""

_OBJ_USR = """\
From the study notes below, extract exactly 20-25 specific learning objectives.

Steps: read notes → list distinct knowledge areas → write 1-2 objectives each →
remove duplicates → confirm count is 20-25 → output JSON array.

NOTES:
{notes}
"""


@_retry()
async def extract_objectives(notes: str) -> list[str]:
    logger.info("objectives_start")
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _OBJ_SYS},
            {"role": "user",   "content": _OBJ_USR.format(notes=notes)},
        ],
    )
    raw = _strip(resp.choices[0].message.content or "")
    objectives: list[str] = json.loads(raw)
    if not (20 <= len(objectives) <= 25):
        raise ValueError(f"Expected 20-25 objectives, got {len(objectives)}")
    logger.info("objectives_done", count=len(objectives))
    return objectives


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_summarisation(transcript: str, title: str) -> tuple[MergedSummary, str, list[str]]:
    """Returns (merged_summary, notes_markdown, learning_objectives)."""
    chunks = chunk_transcript(transcript)
    summaries = [await summarise_chunk(c, i) for i, c in enumerate(chunks)]
    merged = merge_summaries(summaries)
    notes  = await generate_notes(merged, title)
    objs   = await extract_objectives(notes)
    return merged, notes, objs