"""
quiz_service.py
────────────────
Generates 10-15 MCQs from video content (transcript or merged concept summary).
Questions are framed as universal knowledge tests — no reference to "the video" or
"the speaker". Attempt 1 → standard prompt → Pydantic validate.
Attempt 2 → strict repair prompt with failed response as context → Pydantic validate.
Raises ValueError if both fail.
Temperature: 0.35
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
You are an expert quiz writer calibrated to CBSE/ICSE Class 10-12 HOTS (Higher Order
Thinking Skills) standard. Every question you write must target Bloom's Taxonomy
Level 4 (Analysis), Level 5 (Evaluation), or Level 6 (Synthesis) — never Level 1-3.

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
4. EVERY question must connect at least TWO distinct domain concepts. Questions
   that test only a single isolated fact or definition are forbidden.
5. FORBIDDEN question stems — any question beginning with these patterns is rejected:
   - "What is [X]?" (bare definition recall)
   - "What does [X] mean?"
   - "Define [X]."
   - "Which of the following is true about [X]?" (fact-list recall)
   - "What is the definition of..."
   - "What does [X] symbolize in..."
   - "What is the lesson of..."
   - "According to [character]..."
   - "What did [character] teach about..."
6. REQUIRED question stems — every question must use an analytical or evaluative frame:
   - "Why does [X] lead to [Y]?" (cause-effect analysis)
   - "What distinguishes [X] from [Y] in terms of [criterion]?" (analytical distinction)
   - "If [condition], what would happen to [outcome] and why?" (hypothetical evaluation)
   - "A student argues [claim]. What is the flaw in this reasoning?" (critical evaluation)
   - "Under which condition would [principle A] apply rather than [principle B]?" (synthesis)
   - "Given [scenario], which [concept] best explains [observation] and why?" (application + analysis)
7. Distractors must represent PLAUSIBLE conceptual errors — the mistake a student with
   surface memorization (but not deep understanding) would make. Each distractor should
   correspond to a specific named misconception. They must NOT be obviously false or
   trivially distinguishable.
8. Only ONE option should be clearly correct to someone with genuine understanding.
9. The "description" must explain WHY the correct answer is right by naming the
   underlying principle and explaining the reasoning path — minimum 2 sentences.
10. Output ONLY the raw JSON — no markdown fences, no text outside the object.
"""

_QUIZ_USR = """\
Use the transcript below as source material to identify the subject domain and its
core concepts. Then write 10-15 MCQs at CBSE/ICSE Class 10-12 HOTS difficulty —
questions that require analysis, evaluation, or synthesis, not surface recall.

Approach:
1. Read the transcript and extract the underlying domain (e.g., "electrochemistry",
   "macroeconomic policy", "Mendelian genetics", "Newtonian mechanics").
2. Identify 12-16 specific concepts, principles, distinctions, and causal relationships
   from that domain. Do NOT identify definitions — identify relationships and mechanisms.
3. For each question slot, select TWO or more related concepts and write a question that
   requires the student to reason about the relationship between them. A CBSE HOTS
   examiner would include this question in a Class 12 board paper.

Required question distribution (total 10-15):
- TYPE A — Analytical Distinction (2-3 questions):
    Frame: "What distinguishes [concept X] from [concept Y] specifically in the context
    of [criterion or condition]?"
    Tests: ability to separate closely related ideas by a meaningful criterion.

- TYPE B — Cause-Effect / Mechanism Analysis (5-6 questions):
    Frame: "Why does [X] lead to [Y]?", "What would happen to [outcome] if [condition
    changed], and through what mechanism?", "Which step in [process] is rate-limiting
    and why?"
    Tests: understanding of mechanisms, not just outcomes.

- TYPE C — Evaluation / Synthesis (3-4 questions, hardest):
    Frame: "Given [scenario], which principle from [domain] best accounts for
    [observation] and why?", "A student claims [X]. Under what conditions is this claim
    valid, and where does it break down?", "If you were to [design/choose/predict],
    which approach is correct and on what basis?"
    Tests: applying the right principle to a novel or edge-case scenario.

BANNED question patterns — do not produce any question that:
- Begins with "What is [X]?" or "What does [X] mean?" or "Define [X]."
- Begins with "Which of the following is true about [X]?" (fact-list recall)
- Tests only a single concept in isolation without requiring reasoning about a relationship
- Has a correct answer that could be retrieved by memorizing a single textbook sentence

Distractor quality requirement:
- Each wrong option must represent a specific conceptual error a student with
  surface-level memorization would plausibly make. Name the misconception mentally
  as you write it (e.g., "confusing rate with equilibrium constant",
  "conflating correlation with causation", "reversing cause and effect").
- Distractors must NOT be obviously false, trivially distinguishable, or
  "partially true" in a way a careful student could argue for.
- Each distractor should be the kind of answer a student who studied definitions
  but not mechanisms would choose.

Source transcript (extract domain concepts and relationships from this;
do NOT quote it in questions):
{content}

Output ONLY this JSON:
{{
  "quiz": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "description": "Correct because [name the principle and explain the reasoning
      path — minimum 2 sentences]."
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
        temperature=0.35,
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
