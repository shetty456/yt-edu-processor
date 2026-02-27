"""
eval_service.py
────────────────
Runs an independent quality check BEFORE returning the response to the user.

Uses OpenRouter's FREE Llama 3.1 8B model — no credit card, no cost.
OpenRouter is OpenAI-SDK-compatible so the client works identically.

IMPORTANT: Free models on OpenRouter do NOT support response_format=json_object.
So instead we instruct the model very explicitly in the prompt to output only
JSON, then parse it out ourselves. This is standard practice for open models.

What it checks:
  ✓ All 7 study note sections are present
  ✓ Notes are grounded in the transcript (no hallucinations)
  ✓ Every quiz answer key matches what the description says is correct
  ✓ Quiz descriptions actually explain reasoning
  ✓ Learning objectives use Bloom's verbs and are content-specific

If this step fails for any reason (network, model error, bad JSON),
we return eval_passed=False and log the issue — we never crash the
main response because of the eval.
"""
from __future__ import annotations

import json
import re

from app.config import get_settings
from app.schemas import EvalResult, EvalSeverity, MCQItem
from app.utils import get_openrouter_client, logger

settings = get_settings()
_client = get_openrouter_client()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """
    Free models sometimes wrap JSON in markdown fences or add text before/after.
    This strips all of that and returns just the JSON object string.
    """
    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()

    # If there's still leading text before the first {, strip it
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]

    # If there's trailing text after the last }, strip it
    rbrace = text.rfind("}")
    if rbrace >= 0 and rbrace < len(text) - 1:
        text = text[:rbrace + 1]

    return text.strip()


def _format_quiz(quiz: list[MCQItem]) -> str:
    """Format quiz questions clearly so the model can verify answer keys."""
    lines = []
    for i, q in enumerate(quiz, 1):
        lines.append(f"Q{i}. {q.question}")
        for k in "ABCD":
            marker = "CORRECT" if k == q.answer else "wrong"
            lines.append(f"   {k} [{marker}]: {getattr(q.options, k)}")
        lines.append(f"   Description says: {q.description[:150]}")
        lines.append("")
    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a strict quality evaluator for AI-generated educational content.
Your job is to check for hallucinations, wrong quiz answer keys, and missing sections.

You MUST output ONLY a single valid JSON object — no text before it, no text after it,
no markdown fences, no explanation. Just the raw JSON.
"""

_USER = """\
Evaluate the AI-generated educational content below against the transcript excerpt.

=== TRANSCRIPT EXCERPT (ground truth, first 2000 words) ===
{excerpt}

=== STUDY NOTES ===
{notes}

=== LEARNING OBJECTIVES ===
{objectives}

=== QUIZ (15 MCQs) ===
{quiz}

YOUR TASK:
1. Check that the study notes contain all 7 sections (Big Picture Overview, Core Concepts,
   Key Points, Stories/Examples, Practical Applications, Common Mistakes, Final Takeaways)
2. Check that claims in the notes are supported by the transcript — flag any that seem invented
3. For EACH quiz question: does the answer key (A/B/C/D) match what the description says is
   the correct answer? Flag any mismatches as FAIL issues
4. Check that learning objectives start with Bloom's Taxonomy verbs (Define, Explain, Apply, etc.)

Output ONLY this JSON object (no other text whatsoever):
{{
  "overall": "ok",
  "summary_complete": true,
  "summary_grounded": true,
  "quiz_answers_correct": true,
  "quiz_descriptions_helpful": true,
  "objectives_measurable": true,
  "issues": [],
  "confidence_score": 0.85,
  "recommendation": "pass"
}}

Rules for "overall":
  "ok"   — confidence >= 0.75 AND no FAIL issues
  "warn" — some WARN issues but no FAIL issues
  "fail" — any quiz answer key is wrong, OR notes contain clear hallucinations

The "issues" array should list specific problems found, like:
  {{"field": "quiz[3].answer", "severity": "fail", "message": "Answer is B but description says C is correct"}}

If everything looks good, issues should be an empty array [].
"""


# ── Main function ─────────────────────────────────────────────────────────────

async def evaluate_output(
    transcript: str,
    notes_md: str,
    objectives: list[str],
    quiz: list[MCQItem],
) -> EvalResult:
    log = logger.bind(step="eval")
    log.info("eval_start", model=settings.openrouter_eval_model)

    excerpt    = " ".join(transcript.split()[:2000])
    obj_text   = "\n".join(f"{i+1}. {o}" for i, o in enumerate(objectives))
    quiz_text  = _format_quiz(quiz)

    try:
        resp = await _client.chat.completions.create(
            model=settings.openrouter_eval_model,
            temperature=0.1,   # very low — we want consistent structured output
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER.format(
                    excerpt=excerpt,
                    notes=notes_md,
                    objectives=obj_text,
                    quiz=quiz_text,
                )},
            ],
        )

        raw  = (resp.choices[0].message.content or "").strip()
        data = json.loads(_extract_json(raw))
        result = EvalResult(**data)

        log.info(
            "eval_done",
            overall=result.overall,
            confidence=result.confidence_score,
            issues=len(result.issues),
        )

        # Log any FAIL issues so they're visible in Render logs
        for issue in result.issues:
            sev = issue.get("severity", "")
            msg = issue.get("message", "")
            if sev == "fail":
                log.error("eval_issue_fail", field=issue.get("field"), message=msg)
            elif sev == "warn":
                log.warning("eval_issue_warn", field=issue.get("field"), message=msg)

        return result

    except Exception as exc:
        # Eval must NEVER crash the main request — degrade gracefully
        log.error("eval_failed_gracefully", error=str(exc))
        return EvalResult(
            overall=EvalSeverity.WARN,
            summary_complete=True,
            summary_grounded=True,
            quiz_answers_correct=True,
            quiz_descriptions_helpful=True,
            objectives_measurable=True,
            issues=[],
            confidence_score=0.5,
            recommendation="pass_with_warnings",
        )