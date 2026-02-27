"""
quiz_service.py
────────────────
Generates exactly 15 MCQs from study notes + learning objectives.
Attempt 1 → standard prompt → Pydantic validate
Attempt 2 → strict repair prompt → Pydantic validate
Raises ValueError if both fail.
Temperature: 0.25
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.config import get_settings
from app.schemas import MCQItem, QuizPayload
from app.utils import get_sarvam_client, logger

settings = get_settings()
_client = get_sarvam_client()


def _strip(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    return re.sub(r"\n?```\s*$", "", text).strip()


_QUIZ_SYS = """\
You are a senior exam writer for an online learning platform.

RULES:
1. Every question, option, and description must be based on the study notes.
2. Do NOT invent statistics, names, or claims not in the notes.
3. Only ONE option should be clearly correct.
4. The "description" must explain WHY the answer is correct AND why the others are wrong.
5. Output ONLY the raw JSON — no markdown fences, no text outside the object.
"""

_QUIZ_USR = """\
Generate EXACTLY 15 MCQs from the study notes below.

Difficulty: 6 easy (direct recall) + 9 medium (understanding / application).

Process:
1. List 15 distinct topics to test (one per question).
2. For each: write question, 4 distinct options (only one correct), answer key, description.
3. Verify every answer key is correct.
4. Output ONLY this JSON — nothing else:

{{
  "quiz": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "description": "A is correct because... B is wrong because..."
    }}
  ]
}}

STUDY NOTES:
{notes}

LEARNING OBJECTIVES (use as topic guidance):
{objectives}
"""

_REPAIR_SYS = """\
Your previous response failed JSON schema validation.
Output ONLY the raw JSON — no markdown, no explanation.
The schema requires: top-level key "quiz", array of EXACTLY 15 objects,
each with: question (str), options (A/B/C/D), answer (A|B|C|D), description (≥20 chars).
All 15 questions must be distinct. All 4 options per question must be distinct.
"""


async def _call(system: str, user: str) -> str:
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.25,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def _validate(raw: str) -> QuizPayload:
    return QuizPayload(**json.loads(_strip(raw)))


async def generate_quiz(notes_md: str, objectives: list[str]) -> list[MCQItem]:
    log = logger.bind(step="quiz")
    obj_text = "\n".join(f"- {o}" for o in objectives)
    user_prompt = _QUIZ_USR.format(notes=notes_md, objectives=obj_text)

    # Attempt 1
    log.info("quiz_attempt_1")
    raw1 = await _call(_QUIZ_SYS, user_prompt)
    try:
        return _validate(raw1).quiz
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        log.warning("quiz_attempt_1_failed", error=str(e))

    # Attempt 2 — strict repair
    log.info("quiz_attempt_2")
    raw2 = await _call(_REPAIR_SYS, user_prompt)
    try:
        return _validate(raw2).quiz
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        log.error("quiz_both_failed", error=str(e))
        raise ValueError(f"Quiz generation failed after 2 attempts: {e}") from e