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
they must stand alone as knowledge tests about the subject domain.

ABSOLUTE RULES:
1. NEVER reference "the video", "the transcript", "the speaker", "the article",
   "the notes", "the story", "the parable", "the text", or any source material.
2. NEVER ask what a specific character said, what a parable symbolizes, or what
   lesson a particular narrative conveys. Extract the underlying concept and test THAT.
3. Questions must be answerable by a subject-matter expert who has NEVER heard
   this particular story or watched this video — they should be answerable from
   general knowledge of the domain.
4. Every question tests either (a) conceptual understanding of a domain idea or
   (b) practical application of a domain principle in a scenario.
5. FORBIDDEN question stems:
   - "What does [X] symbolize in..."
   - "What is the lesson of..."
   - "What does the story/parable/teaching convey..."
   - "What broader principle does [X] illustrate..."
   - "According to [character]..."
   - "What did [character] teach about..."
6. Distractors must be DEFINITIVELY WRONG — each should represent a specific,
   named misconception. Never write a distractor that is merely "a bit off" or
   "partially correct." A learner who truly understands should instantly rule it out.
7. Only ONE option should be clearly correct.
8. The "description" must explain WHY the answer is correct AND why each wrong option
   is wrong, in plain language.
9. Output ONLY the raw JSON — no markdown fences, no text outside the object.
"""

_QUIZ_USR = """\
Use the transcript below as source material to identify the subject domain and its
core concepts. Then write 10-15 MCQs that test understanding of THAT DOMAIN — not
comprehension of this particular transcript.

Approach:
1. Read the transcript and extract the underlying domain (e.g., "karma and spiritual
   effort in yogic philosophy", "machine learning fundamentals", "fiscal policy").
2. Identify 10-15 specific concepts, principles, or applied scenarios from that domain.
3. For each concept, write a question a textbook on this subject might ask — one that
   is answerable without this specific transcript.

Mix of question types:
- Concept questions (6-8): test definitions, cause-effect relationships, distinctions
  between related ideas — framed in domain terms, not story terms.
- Application questions (4-7): present a novel scenario and ask which principle applies
  or what outcome follows — no references to characters or events from the transcript.

Distractor quality requirement:
- Each wrong option must be CLEARLY wrong and represent a specific, common misconception
  about the domain. Label the misconception mentally as you write it.
- Avoid writing "partially true" or "adjacent" options that a thoughtful person could
  argue for.

Source transcript (extract domain knowledge from this; do NOT quote it in questions):
{content}

Output ONLY this JSON:
{{
  "quiz": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "description": "A is correct because... B is wrong because... C is wrong because... D is wrong because..."
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
