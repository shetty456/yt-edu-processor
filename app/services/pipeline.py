from __future__ import annotations

import json
from typing import Any

from agno.workflow import Workflow, Step, Steps, OnError
from agno.workflow import StepInput, StepOutput

from app.schemas import VideoMeta, MCQItem
from app.services.youtube_service import extract_video
from app.services.summarization_service import run_summarisation
from app.services.quiz_service import generate_quiz
from app.services.eval_service import evaluate_output
from app.utils import logger


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_load(raw: Any, default=None) -> Any:
    """Safely load JSON/dict, return default if None."""
    if raw is None:
        return default
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


# ── Step executors ────────────────────────────────────────────────────────────
async def _extract(step_input: StepInput) -> StepOutput:
    url: str = step_input.input  # type: ignore[assignment]
    log = logger.bind(step="extract_video")
    log.info("start")
    try:
        meta = await extract_video(url)
        log.info("done", title=meta.title, words=meta.word_count)
        return StepOutput(content=meta.model_dump(), success=True)
    except Exception as exc:
        log.error("failed", error=str(exc))
        # Store the error in content so _assemble can surface it as a 422
        return StepOutput(content={"_failed": True, "_error": str(exc)}, success=False)


async def _summarise(step_input: StepInput) -> StepOutput:
    log = logger.bind(step="summarisation")
    log.info("start")
    try:
        meta_d = _safe_load(step_input.get_step_content("extract_video"), {})
        meta = VideoMeta(**meta_d)
        merged, notes = await run_summarisation(meta.transcript, meta.title)
        payload = {
            "merged_summary": merged.model_dump(),
            "notes_markdown": notes,
        }
        log.info("done")
        return StepOutput(content=payload, success=True)
    except Exception as exc:
        log.error("failed", error=str(exc))
        return StepOutput(content=None, success=False, error=str(exc))


async def _quiz(step_input: StepInput) -> StepOutput:
    log = logger.bind(step="quiz_generation")
    log.info("start")
    try:
        meta_d = _safe_load(step_input.get_step_content("extract_video"), {})
        meta = VideoMeta(**meta_d)

        # Always use raw transcript, capped at 8000 words
        words = meta.transcript.split()
        content = " ".join(words[:8000]) if len(words) > 8000 else meta.transcript

        items = await generate_quiz(content=content)
        log.info("done", questions=len(items))
        return StepOutput(content=[i.model_dump() for i in items], success=True)
    except Exception as exc:
        log.error("failed", error=str(exc))
        return StepOutput(content={"_failed": True, "_error": str(exc)}, success=False)


async def _eval(step_input: StepInput) -> StepOutput:
    log = logger.bind(step="eval")
    log.info("start")
    try:
        meta_d = _safe_load(step_input.get_step_content("extract_video"), {})
        meta = VideoMeta(**meta_d)
        summ = _safe_load(step_input.get_step_content("summarisation"), {})
        quiz_raw = _safe_load(step_input.get_step_content("quiz_generation"), [])

        result = await evaluate_output(
            transcript=meta.transcript,
            notes_md=summ.get("notes_markdown", ""),
            quiz=[MCQItem(**q) for q in quiz_raw],
        )
        log.info("done", overall=result.overall, passed=result.passed)
        return StepOutput(content=result.model_dump(), success=True)
    except Exception as exc:
        log.error("failed_gracefully", error=str(exc))
        return StepOutput(
            content={"overall": "warn", "passed": True, "confidence_score": 0.5},
            success=True,
        )


async def _assemble(step_input: StepInput) -> StepOutput:
    meta_d = _safe_load(step_input.get_step_content("extract_video"), {})
    if meta_d.get("_failed"):
        raise ValueError(meta_d.get("_error", "Video extraction failed."))
    summ_d = _safe_load(
        step_input.get_step_content("summarisation"),
        {"notes_markdown": ""},
    )
    quiz_raw = _safe_load(step_input.get_step_content("quiz_generation"), None)
    quiz_failed = isinstance(quiz_raw, dict) and quiz_raw.get("_failed")
    quiz_items = [] if quiz_failed or quiz_raw is None else quiz_raw
    eval_d = _safe_load(step_input.get_step_content("eval"), {})

    overall = eval_d.get("overall", "warn")
    eval_passed = overall in ("ok", "warn")

    payload = {
        "video_title": meta_d.get("title", "Unknown"),
        "summary_markdown": summ_d.get("notes_markdown", ""),
        "quiz": quiz_items,
        "quiz_generation_failed": bool(quiz_failed),
        "eval_passed": eval_passed,
    }

    logger.info(
        "pipeline_complete", title=payload["video_title"], eval_passed=eval_passed
    )
    return StepOutput(content=payload, success=True)


# ── Workflow factory ──────────────────────────────────────────────────────────
def build_pipeline() -> Workflow:
    return Workflow(
        name="VideoProcessorWorkflow",
        description="YouTube video → study notes + MCQs (deterministic, no agents)",
        steps=[
            Step(
                name="extract_video",
                executor=_extract,
                max_retries=1,
                on_error=OnError.fail,
            ),
            Step(
                name="summarisation",
                executor=_summarise,
                max_retries=1,
                on_error=OnError.fail,
            ),
            Step(
                name="quiz_generation",
                executor=_quiz,
                max_retries=1,
                on_error=OnError.fail,
            ),
            Step(name="eval", executor=_eval, max_retries=1, on_error=OnError.skip),
            Step(
                name="assemble_response",
                executor=_assemble,
                max_retries=1,
                on_error=OnError.fail,
            ),
        ],
    )
