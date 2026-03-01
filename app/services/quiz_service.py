"""
quiz_service.py
────────────────
Generates 10-15 MCQs from video content (transcript or merged concept summary).
Questions are framed as universal knowledge tests — no reference to "the video" or
"the speaker". Attempt 1 → standard prompt → Pydantic validate.
Attempt 2 → strict repair prompt with failed response as context → Pydantic validate.
Raises ValueError if both fail.
Temperature: 0.25
"""
from __future__ import annotations

import json
import random
import re

from pydantic import ValidationError

from app.config import get_settings
from app.schemas import MCQItem, MCQOptions, QuizPayload
from app.utils import get_sarvam_client, logger

settings = get_settings()
_client = get_sarvam_client()


def _strip(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    rbrace = text.rfind("}")
    if rbrace >= 0 and rbrace < len(text) - 1:
        text = text[:rbrace + 1]
    return text.strip()


_QUIZ_SYS = """\
You are an expert quiz writer for an online learning platform.

Your questions will be shown to learners WITHOUT any reference material —
they must stand alone as knowledge tests.

ABSOLUTE RULES:
1. NEVER reference "the video", "the transcript", "the speaker", "the article",
   "the notes", or any source material. Write as if the question exists in a textbook.
2. Questions must be answerable by someone who understands the subject —
   not someone who watched a specific video.
3. Every question tests either (a) conceptual understanding or (b) practical application.
4. Only ONE option should be clearly correct.
5. The "description" must explain WHY the answer is correct AND why the others are wrong,
   in plain language.
6. Output ONLY the raw JSON — no markdown fences, no text outside the object.
"""

_QUIZ_USR = """\
Generate between 10 and 15 MCQs that test deep understanding of the subject below.
Write questions as if for a textbook chapter on this topic — standalone, not tied to any specific source.

Mix of question types:
- Concept questions (6-8): test definitions, explanations, relationships between ideas
- Application questions (4-7): test ability to use or reason with the concepts in a scenario

IMPORTANT: Frame every question as universal knowledge —
"What is X?", "Which best describes Y?", "In situation Z, what would you do?"
Do NOT ask "According to..." or "What did the author say about..."

Source transcript (study this and write questions that test understanding of the subject):
{content}

Output ONLY this JSON:
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
"""

_REPAIR_SYS = """\
Your previous response failed JSON schema validation.
Output ONLY the raw JSON — no markdown, no explanation.
The schema requires: top-level key "quiz", array of 10 to 15 objects,
each with: question (str), options (A/B/C/D), answer (A|B|C|D), description (≥20 chars).
All questions must be distinct. All 4 options per question must be distinct.
Do NOT reference "the video", "the transcript", or "the speaker" in any question.
"""


def _swap_letter(text: str, a: str, b: str) -> str:
    """Swap standalone letter tokens a ↔ b (word boundaries)."""
    return re.sub(
        rf'\b([{a}{b}])\b',
        lambda m: b if m.group(1) == a else a,
        text,
    )


def _shuffle_answer(item: MCQItem) -> MCQItem:
    """Randomly reposition the correct answer across A/B/C/D."""
    letters = ["A", "B", "C", "D"]
    target = random.choice(letters)
    if target == item.answer:
        return item

    old = item.answer
    opts = {k: getattr(item.options, k) for k in letters}
    opts[old], opts[target] = opts[target], opts[old]

    return MCQItem(
        question=item.question,
        options=MCQOptions(**opts),
        answer=target,
        description=_swap_letter(item.description, old, target),
    )


async def _call(system: str, user: str) -> str:
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.25,
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def _validate(raw: str) -> QuizPayload:
    return QuizPayload(**json.loads(_strip(raw)))


async def generate_quiz(content: str) -> list[MCQItem]:
    log = logger.bind(step="quiz")
    user_prompt = _QUIZ_USR.format(content=content)

    # Attempt 1
    log.info("quiz_attempt_1")
    raw1 = await _call(_QUIZ_SYS, user_prompt)
    try:
        items = _validate(raw1).quiz
        return [_shuffle_answer(i) for i in items]
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        log.warning("quiz_attempt_1_failed", error=str(e))

    # Attempt 2 — strict repair with failed response as context
    log.info("quiz_attempt_2")
    repair_user = (
        f"Your previous response:\n\n{raw1}\n\n"
        f"Failed validation. Fix and output corrected JSON with 10 to 15 questions."
    )
    raw2 = await _call(_REPAIR_SYS, repair_user)
    try:
        items = _validate(raw2).quiz
        return [_shuffle_answer(i) for i in items]
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        log.error("quiz_both_failed", error=str(e))
        raise ValueError(f"Quiz generation failed after 2 attempts: {e}") from e
