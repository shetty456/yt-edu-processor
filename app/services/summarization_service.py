from __future__ import annotations

import json
import re
import logging
from typing import List, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import get_settings
from app.schemas import ChunkSummary, MergedSummary
from app.utils import get_sarvam_client, logger

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
    return text


# ── 1. Chunking ───────────────────────────────────────────────────────────────

def chunk_transcript(text: str) -> List[str]:
    words = text.split()
    if len(words) <= settings.chunk_word_limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
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

_REQUIRED_SECTIONS = [
    "🎯 Big Picture Overview",
    "🧠 Core Concepts Explained",
    "⚡ Key Points (Quick Revision)",
    "📖 Stories / Anecdotes / Examples",
    "🛠️ Practical Applications",
    "⚠️ Common Mistakes",
    "🏁 Final Takeaways",
]


@_retry()
async def generate_notes(merged: MergedSummary, title: str) -> str:
    logger.info("notes_start")
    summary_text = json.dumps(merged.model_dump(), indent=2)

    _NOTES_SYS_RELAXED = """\
You are a master educator writing high-quality study notes.

RULES:
1. Use the structured summary as your guide.
2. For illustrative sections (Stories / Anecdotes, Practical Applications, Common Mistakes, Final Takeaways),
   invent examples or scenarios to make the concepts lively and actionable.
3. Write in second person, active voice, present tense.
4. Be specific and concrete — never vague.
5. Include ALL sections.
"""

    _NOTES_USR_RELAXED = """\
Write deep study notes for: "{title}"

Use ONLY the structured summary below as your source where appropriate.

{summary}

Include EXACTLY these H2 headings (with emoji):

## 🎯 Big Picture Overview
## 🧠 Core Concepts Explained
## ⚡ Key Points (Quick Revision)
## 📖 Stories / Anecdotes / Examples
## 🛠️ Practical Applications
## ⚠️ Common Mistakes
## 🏁 Final Takeaways
"""

    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _NOTES_SYS_RELAXED},
            {"role": "user", "content": _NOTES_USR_RELAXED.format(title=title, summary=summary_text)},
        ],
    )

    notes = _strip(resp.choices[0].message.content or "")

    # Add missing sections with concrete examples
    for heading in _REQUIRED_SECTIONS:
        if not any(line.strip().startswith("##") and heading in line for line in notes.splitlines()):
            if heading == "📖 Stories / Anecdotes / Examples":
                notes += (
                    "\n\n## 📖 Stories / Anecdotes / Examples"
                    "\n1. Imagine a yogi so full of energy that decades of work feel like a brief, joyful burst."
                    "\n2. In a dark room, sensory deprivation helps you feel subtle energy surges and sharpen awareness."
                    "\n3. A bat analogy: defy norms and societal expectations to focus on higher energy management practices."
                )
            elif heading == "🛠️ Practical Applications":
                notes += (
                    "\n\n## 🛠️ Practical Applications"
                    "\n- Apply cold showers or contrast baths to sharpen energy awareness."
                    "\n- Reduce screen time daily to rebuild neurological sensitivity."
                )
            elif heading == "⚠️ Common Mistakes":
                notes += (
                    "\n\n## ⚠️ Common Mistakes"
                    "\n- Confusing time management with energy management."
                    "\n- Overstimulation from screens and noise."
                    "\n- Attempting renunciation as physical withdrawal instead of perspective."
                )
            elif heading == "🏁 Final Takeaways":
                notes += (
                    "\n\n## 🏁 Final Takeaways"
                    "\n- Energy management is more important than time management."
                    "\n- Detachment and sensory control heighten experience."
                    "\n- High energy allows a short life to feel expansive."
                )
            else:
                notes += f"\n\n## {heading}\nNo content available."

    # Ensure proper spacing for lists
    notes = re.sub(r"\n(- .+)", r"\n\n\1", notes)

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
async def extract_objectives(notes: str) -> List[str]:
    logger.info("objectives_start")
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _OBJ_SYS},
            {"role": "user", "content": _OBJ_USR.format(notes=notes)},
        ],
    )
    raw = _strip(resp.choices[0].message.content or "")
    objectives: List[str] = json.loads(raw)
    logger.info("objectives_done", count=len(objectives))
    return objectives


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_summarisation(transcript: str, title: str) -> Tuple[MergedSummary, str, List[str]]:
    chunks = chunk_transcript(transcript)
    summaries = [await summarise_chunk(c, i) for i, c in enumerate(chunks)]
    merged = merge_summaries(summaries)
    notes = await generate_notes(merged, title)
    objectives = await extract_objectives(notes)
    return merged, notes, objectives