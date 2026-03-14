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
import logging
import random
import re

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import get_settings
from app.schemas import MCQItem, MCQOptions, QuizPayload
from app.utils import get_sarvam_client, logger, strip_think

settings = get_settings()
_client = get_sarvam_client()

def _api_retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=20),
        before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
        reraise=True,
    )


def _strip(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    # Find the first JSON structure opener: object '{' or array '['
    obj = text.find("{")
    arr = text.find("[")
    if obj < 0:
        start = arr
    elif arr < 0:
        start = obj
    else:
        start = min(obj, arr)
    if start > 0:
        text = text[start:]
    # Trim trailing text after the closing bracket/brace
    if text.startswith("["):
        rbracket = text.rfind("]")
        if rbracket >= 0 and rbracket < len(text) - 1:
            text = text[:rbracket + 1]
    else:
        rbrace = text.rfind("}")
        if rbrace >= 0 and rbrace < len(text) - 1:
            text = text[:rbrace + 1]
    return text.strip()


_RECALL_PATTERNS = re.compile(
    r"^(how many|at which stage|which (structure|enzyme|organelle|cell|stage|type)|"
    r"what is (the|a|an)|what are the|define |name the|"
    r"which of the following is (true|correct|false))",
    re.IGNORECASE,
)


def _is_recall_question(q: "MCQItem") -> bool:
    return bool(_RECALL_PATTERNS.match(q.question.strip()))


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

CONCRETE EXAMPLES — pattern to follow:
  BAD (Level 1 — rejected): "What is the role of callase?"
  GOOD (Level 4 — required): "Why does premature callase activity before pollen maturity
    lead to male sterility, and which agricultural application exploits this mechanism?"

  BAD (Level 1 — rejected): "How many microspores are produced per meiosis?"
  GOOD (Level 4 — required): "A mutation causes a PMC to complete only meiosis I before
    arrest. What is the immediate consequence for pollen count and ploidy, and why?"

  BAD (Level 1 — rejected): "At which stage are pollen grains shed?"
  GOOD (Level 5 — required): "What distinguishes a bicellular pollen grain from a
    tricellular one in terms of developmental timing, and which condition determines
    which type a species produces?"
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
{concept_section}
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

_CONCEPTS_SYS = """\
You are a domain-analysis assistant. Your only task is to extract concept-pairs and
their causal / mechanistic relationships from the provided educational transcript.
Output ONLY a raw JSON array — no markdown fences, no explanation.
"""

_CONCEPTS_USR = """\
Read the transcript below and extract 12-16 concept-pairs that capture the most
important mechanisms, causal chains, or analytical distinctions in the domain.

For each pair output:
  "concept_a"    — first concept or process
  "concept_b"    — second concept or process
  "relationship" — one sentence describing the causal, functional, or contrastive link
                   between them (e.g. "X triggers Y by …", "X differs from Y in that …")

Only relationships — no isolated definitions.

Transcript:
{content}

Output ONLY this JSON:
[
  {{"concept_a": "...", "concept_b": "...", "relationship": "..."}},
  ...
]
"""

_FACTUAL_QUIZ_SYS = """\
You are a quiz writer for factual, data-driven content (reports, timelines, statistics).
Write 10-15 MCQs that test precise recall and basic interpretation of dates, figures,
and facts from the source material.

RULES:
1. Questions must be directly answerable from the source data — no inference beyond the text.
2. Each question tests ONE specific fact: a number, date, percentage, ranking, or named entity.
3. The correct answer must appear verbatim (or near-verbatim) in the source.
4. Distractors must be plausible: adjacent years, similar percentages, related entities —
   NOT obviously wrong values. A student who remembers the approximate figure but not
   the exact value should find distractors challenging.
5. Allowed question stems:
   - "What was [X] in [year/period]?"
   - "Which figure represents [X] as of [date]?"
   - "In which year did [event/milestone] occur?"
   - "By what percentage did [X] change between [Y1] and [Y2]?"
   - "Which country/entity ranked [position] in [metric] during [period]?"
6. NEVER reference "the document", "the report", "the text", or "the source".
   Ask as if testing general knowledge of the domain.
7. Output ONLY raw JSON — no markdown fences.
"""

_FACTUAL_QUIZ_USR = """\
Source material:
{content}

Write 10-15 MCQs testing exact recall of dates, figures, and facts.

Output ONLY this JSON:
{{
  "quiz": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "description": "Correct because [state the exact figure/date from source and why the distractors are wrong]."
    }}
  ]
}}
"""

_REPAIR_SYS = """\
Your previous response either failed JSON schema validation or contained low-quality
recall questions. Fix both issues and output corrected JSON.
Output ONLY the raw JSON — no markdown, no explanation.
The schema requires a top-level "quiz" key containing an array of 10 to 15 objects.
Each object must have EXACTLY this structure:
{
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A",
  "description": "Correct because ... [minimum 2 sentences]"
}
Rules: all questions distinct, all 4 options per question distinct, answer is one of A/B/C/D.
Do NOT reference "the video", "the transcript", or "the speaker" in any question.
Replace every rejected recall question with a Why / What-would-happen /
What-distinguishes stem that requires reasoning about a relationship between two concepts.
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


@_api_retry()
async def _call(system: str, user: str) -> str:
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.35,
        max_tokens=8192,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content = (resp.choices[0].message.content or "").strip()
    raw = strip_think(content)
    if not raw:
        raw = content  # fallback: model put JSON inside <think> block
    return raw


@_api_retry()
async def _call_factual(system: str, user: str) -> str:
    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.2,
        max_tokens=8192,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content = (resp.choices[0].message.content or "").strip()
    raw = strip_think(content)
    if not raw:
        raw = content
    return raw


def _validate(raw: str) -> QuizPayload:
    if not raw:
        raise ValueError("Empty response from model")
    return QuizPayload(**json.loads(_strip(raw)))


async def _extract_concepts(content: str, language: str = "English") -> str:
    """Pass 1: extract concept-pairs from the transcript for chain-of-thought seeding."""
    sys_prompt = _CONCEPTS_SYS
    if language != "English":
        sys_prompt += f"\nWrite all concept names and relationship descriptions in {language}."
    try:
        raw = await _call(sys_prompt, _CONCEPTS_USR.format(content=content))
        pairs = json.loads(_strip(raw))
        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("empty concept list")
        return json.dumps(pairs, indent=2)
    except Exception as exc:
        logger.bind(step="quiz").warning("concept_extract_failed", error=str(exc))
        return ""  # fall through gracefully; question prompt still has transcript


async def generate_quiz(
    content: str, language: str = "English", is_factual: bool = False
) -> list[MCQItem]:
    log = logger.bind(step="quiz")

    # ── Factual path: skip concept extraction; use precision recall prompts ───
    if is_factual:
        log.info("quiz_factual_path")
        factual_sys = _FACTUAL_QUIZ_SYS
        if language != "English":
            factual_sys += f"\nWrite all questions, answer options, and descriptions in {language}."
        raw1 = await _call_factual(factual_sys, _FACTUAL_QUIZ_USR.format(content=content))
        try:
            items = _validate(raw1).quiz
            return [_shuffle_answer(i) for i in items]
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            log.warning("quiz_factual_attempt_1_failed", error=str(e))
            # Repair attempt
            _MAX_RAW1_CHARS = 3200
            raw1_excerpt = raw1 if len(raw1) <= _MAX_RAW1_CHARS else raw1[:_MAX_RAW1_CHARS] + "\n... [truncated]"
            repair_user = (
                f"Your previous response:\n\n{raw1_excerpt}\n\n"
                f"Failed validation. Fix and output corrected JSON with 10 to 15 questions."
            )
            repair_sys = _REPAIR_SYS
            if language != "English":
                repair_sys += f"\nWrite all questions, answer options, and descriptions in {language}."
            raw2 = await _call_factual(repair_sys, repair_user)
            try:
                items2 = _validate(raw2).quiz
                return [_shuffle_answer(i) for i in items2]
            except (json.JSONDecodeError, ValidationError, KeyError) as e2:
                # Non-English factual: same lenient fallback as the HOTS path
                if language != "English":
                    try:
                        raw_parsed = json.loads(_strip(raw2))
                        raw_list = raw_parsed.get("quiz", raw_parsed) if isinstance(raw_parsed, dict) else raw_parsed
                        valid = []
                        for q in raw_list:
                            if isinstance(q, dict):
                                try:
                                    valid.append(MCQItem(**q))
                                except Exception:
                                    pass
                        if len(valid) >= 10:
                            log.warning("quiz_factual_non_english_partial", count=len(valid), language=language)
                            return [_shuffle_answer(i) for i in valid]
                    except Exception:
                        pass
                log.error("quiz_factual_both_failed", error=str(e2))
                raise ValueError(f"Factual quiz generation failed after 2 attempts: {e2}") from e2

    # ── Pass 1: chain-of-thought concept extraction ──────────────────────────
    log.info("quiz_concept_extract")
    concept_json = await _extract_concepts(content, language=language)
    if concept_json:
        concept_section = (
            "\nPre-extracted concept-pairs (use these as seed material for your "
            "questions — every question must be grounded in one of these relationships):\n"
            f"{concept_json}\n"
        )
    else:
        concept_section = ""

    user_prompt = _QUIZ_USR.format(content=content, concept_section=concept_section)

    quiz_sys = _QUIZ_SYS
    if language != "English":
        quiz_sys += (
            f"\nWrite all questions, answer options, and descriptions in {language}."
        )

    # ── Pass 2: question generation ──────────────────────────────────────────
    log.info("quiz_attempt_1")
    raw1 = await _call(quiz_sys, user_prompt)
    try:
        items = _validate(raw1).quiz
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        log.warning("quiz_attempt_1_failed", error=str(e))
        items = None

    # ── Recall filter (English-only regex — skip for other languages) ────────
    if items is not None:
        if language == "English":
            recall_items = [q for q in items if _is_recall_question(q)]
            if recall_items:
                log.warning(
                    "recall_questions_detected",
                    count=len(recall_items),
                    questions=[q.question for q in recall_items],
                )
            if len(recall_items) <= 2:
                # Tolerable — drop the offenders and return the rest
                good_items = [q for q in items if not _is_recall_question(q)]
                if good_items:
                    return [_shuffle_answer(i) for i in good_items]
            # More than 2 recall questions — fall through to repair
        else:
            # Non-English: regex can't detect recall questions — trust the model
            return [_shuffle_answer(i) for i in items]

    # ── Pass 3: repair (schema failure OR too many English recall questions) ──
    log.info("quiz_attempt_2")
    rejected_lines = ""
    if items is not None:
        recall_items = [q for q in items if _is_recall_question(q)]
        if recall_items:
            rejected_lines = (
                "\n\nThe following questions were rejected for being surface recall "
                "(Level 1-2). DO NOT repeat these patterns:\n"
                + "\n".join(f"  - {q.question}" for q in recall_items)
                + "\n\nReplace each with a Why / What-would-happen / "
                  "What-distinguishes question that requires reasoning about a "
                  "relationship between two concepts.\n"
            )

    # Truncate raw1 to ~3200 chars (~800 tokens) so the repair prompt's input
    # doesn't balloon and leave insufficient room for the model's response.
    _MAX_RAW1_CHARS = 3200
    raw1_excerpt = raw1 if len(raw1) <= _MAX_RAW1_CHARS else raw1[:_MAX_RAW1_CHARS] + "\n... [truncated]"
    repair_user = (
        f"Your previous response:\n\n{raw1_excerpt}\n\n"
        f"Failed validation or contained too many recall questions. "
        f"Fix and output corrected JSON with 10 to 15 questions."
        f"{rejected_lines}"
    )
    repair_sys = _REPAIR_SYS
    if language != "English":
        repair_sys += f"\nWrite all questions, answer options, and descriptions in {language}."
    raw2 = await _call(repair_sys, repair_user)
    try:
        items2 = _validate(raw2).quiz
        return [_shuffle_answer(i) for i in items2]
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        # Non-English: HOTS + Indic scripts is harder; accept ≥8 well-formed questions
        # rather than returning 500 to the user.
        if language != "English":
            try:
                raw_parsed = json.loads(_strip(raw2))
                raw_list = raw_parsed.get("quiz", raw_parsed) if isinstance(raw_parsed, dict) else raw_parsed
                valid = []
                for q in raw_list:
                    if isinstance(q, dict):
                        try:
                            valid.append(MCQItem(**q))
                        except Exception:
                            pass
                if len(valid) >= 10:
                    log.warning("quiz_non_english_partial", count=len(valid), language=language)
                    return [_shuffle_answer(i) for i in valid]
            except Exception:
                pass
        log.error("quiz_both_failed", error=str(e))
        raise ValueError(f"Quiz generation failed after 2 attempts: {e}") from e
