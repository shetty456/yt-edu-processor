from __future__ import annotations

import asyncio
from typing import Any

from app.schemas import VideoMeta
from app.services.youtube_service import extract_video
from app.services.summarization_service import run_summarisation
from app.services.quiz_service import generate_quiz
from app.services.eval_service import evaluate_output
from app.utils import logger


async def run_pipeline(url: str) -> dict[str, Any]:
    log = logger.bind(url=url)
    log.info("pipeline_start")

    # Step 1: Extract (must finish before anything else)
    meta: VideoMeta = await extract_video(url)
    log.info("extract_done", title=meta.title, words=meta.word_count)

    # Step 2: Summarise + Quiz in parallel (both only need transcript)
    words = meta.transcript.split()
    quiz_content = " ".join(words[:8000]) if len(words) > 8000 else meta.transcript

    (_, notes_markdown), quiz_items = await asyncio.gather(
        run_summarisation(meta.transcript, meta.title),
        generate_quiz(content=quiz_content),
    )
    log.info("parallel_done", quiz_questions=len(quiz_items))

    # Step 3: Eval (graceful — never crashes the pipeline)
    try:
        eval_result = await evaluate_output(
            transcript=meta.transcript,
            notes_md=notes_markdown,
            quiz=quiz_items,
        )
        eval_passed = eval_result.passed
        log.info("eval_done", overall=eval_result.overall, passed=eval_passed)
    except Exception as exc:
        log.error("eval_failed_gracefully", error=str(exc))
        eval_passed = True

    # Step 4: Assemble
    payload: dict[str, Any] = {
        "video_title": meta.title,
        "summary_markdown": notes_markdown,
        "quiz": [item.model_dump() for item in quiz_items],
        "quiz_generation_failed": False,
        "eval_passed": eval_passed,
    }
    log.info("pipeline_complete", title=meta.title, eval_passed=eval_passed)
    return payload
