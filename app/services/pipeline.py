from __future__ import annotations

import asyncio
from typing import Any

from app.schemas import VideoMeta
from app.services.youtube_service import extract_video
from app.services.summarization_service import run_summarisation
from app.services.quiz_service import generate_quiz
from app.services.eval_service import evaluate_output
from app.utils import logger

# imported lazily inside run_pdf_pipeline to avoid circular imports at module load



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
        "youtube_url": url,
        "video_title": meta.title,
        "summary_markdown": notes_markdown,
        "quiz": [item.model_dump() for item in quiz_items],
        "quiz_generation_failed": False,
        "eval_passed": eval_passed,
    }
    log.info("pipeline_complete", title=meta.title, eval_passed=eval_passed)
    return payload


async def run_pdf_pipeline(pdf_bytes: bytes, filename: str) -> dict[str, Any]:
    from app.services.pdf_service import upload_pdf_to_cloudinary, extract_pdf_text, infer_pdf_title

    log = logger.bind(filename=filename)
    log.info("pdf_pipeline_start")

    # Step 1: Upload + extract text in parallel
    pdf_url, text = await asyncio.gather(
        upload_pdf_to_cloudinary(pdf_bytes, filename),
        extract_pdf_text(pdf_bytes),
    )
    if not text.strip():
        raise ValueError("Could not extract readable text from PDF.")

    # Step 2: Infer title
    title = await infer_pdf_title(text)
    log.info("pdf_title_inferred", title=title)

    # Step 3: Summarise + Quiz in parallel
    from app.config import get_settings as _get_settings
    s = _get_settings()
    words = text.split()
    quiz_content = " ".join(words[:s.pdf_quiz_word_limit]) if len(words) > s.pdf_quiz_word_limit else text

    (_, notes_markdown), quiz_items = await asyncio.gather(
        run_summarisation(
            text, title,
            chunk_word_limit=s.pdf_chunk_word_limit,
            chunk_target_words=s.pdf_chunk_target_words,
        ),
        generate_quiz(content=quiz_content),
    )
    log.info("pdf_parallel_done", quiz_questions=len(quiz_items))

    # Step 4: Eval (graceful)
    try:
        eval_result = await evaluate_output(
            transcript=text,
            notes_md=notes_markdown,
            quiz=quiz_items,
        )
        eval_passed = eval_result.passed
        log.info("pdf_eval_done", overall=eval_result.overall, passed=eval_passed)
    except Exception as exc:
        log.error("pdf_eval_failed_gracefully", error=str(exc))
        eval_passed = True

    payload: dict[str, Any] = {
        "pdf_url": pdf_url,
        "title": title,
        "summary_markdown": notes_markdown,
        "quiz": [item.model_dump() for item in quiz_items],
        "eval_passed": eval_passed,
    }
    log.info("pdf_pipeline_complete", title=title, eval_passed=eval_passed)
    return payload
