from __future__ import annotations

import asyncio
from typing import Any

from app.schemas import VideoMeta
from app.services.youtube_service import extract_video
from app.services.summarization_service import run_summarisation
from app.services.quiz_service import generate_quiz
from app.services.eval_service import evaluate_output
from app.utils import logger
from app.cache import get_or_acquire, _set, _locks, yt_key, pdf_key, web_key
from app.config import get_settings

# imported lazily inside run_pdf_pipeline to avoid circular imports at module load


async def _run_pipeline_inner(url: str) -> dict[str, Any]:
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


async def run_pipeline(url: str) -> dict[str, Any]:
    s = get_settings()
    key = yt_key(url) if s.cache_ttl_seconds else None

    if key:
        cached, lock = await get_or_acquire(key)
        if cached is not None:
            logger.info("cache_hit", key=key)
            return cached
    else:
        lock = None

    try:
        result = await _run_pipeline_inner(url)
        if key and lock:
            _set(key, result, s.cache_ttl_seconds)
        return result
    finally:
        if lock:
            lock.release()
            if _locks.get(key) is lock:
                _locks.pop(key, None)


async def _run_pdf_pipeline_inner(pdf_bytes: bytes, filename: str) -> dict[str, Any]:
    from app.services.pdf_service import upload_pdf_to_cloudinary, extract_pdf_text, infer_pdf_title, detect_language

    log = logger.bind(filename=filename)
    log.info("pdf_pipeline_start")

    # Step 1: Upload + extract text in parallel
    pdf_url, text = await asyncio.gather(
        upload_pdf_to_cloudinary(pdf_bytes, filename),
        extract_pdf_text(pdf_bytes),
    )
    if not text.strip():
        raise ValueError("Could not extract readable text from PDF.")

    # Step 1b: Detect language of the extracted text (run in thread — langdetect is sync/CPU)
    detected_language = await asyncio.to_thread(detect_language, text)
    log.info("pdf_language_detected", language=detected_language)

    # Step 2: Infer title (falls back to filename stem on API error)
    fallback_title = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()
    title = await infer_pdf_title(text, fallback=fallback_title, language=detected_language)
    log.info("pdf_title_inferred", title=title)

    # Step 3: Summarise + Quiz in parallel
    s = get_settings()
    words = text.split()
    quiz_content = " ".join(words[:s.pdf_quiz_word_limit]) if len(words) > s.pdf_quiz_word_limit else text

    (_, notes_markdown), quiz_items = await asyncio.gather(
        run_summarisation(
            text, title,
            chunk_word_limit=s.pdf_chunk_word_limit,
            chunk_target_words=s.pdf_chunk_target_words,
            language=detected_language,
        ),
        generate_quiz(content=quiz_content, language=detected_language),
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


async def run_pdf_pipeline(pdf_bytes: bytes, filename: str) -> dict[str, Any]:
    s = get_settings()
    key = pdf_key(pdf_bytes) if s.cache_ttl_seconds else None

    if key:
        cached, lock = await get_or_acquire(key)
        if cached is not None:
            logger.info("cache_hit", key=key)
            return cached
    else:
        lock = None

    try:
        result = await _run_pdf_pipeline_inner(pdf_bytes, filename)
        if key and lock:
            _set(key, result, s.cache_ttl_seconds)
        return result
    finally:
        if lock:
            lock.release()
            if _locks.get(key) is lock:
                _locks.pop(key, None)


async def _run_web_pipeline_inner(url: str) -> dict[str, Any]:
    from app.services.web_service import fetch_and_extract
    from app.services.pdf_service import detect_language

    log = logger.bind(url=url)
    log.info("web_pipeline_start")

    # Step 1: Fetch and extract article text
    title, text = await fetch_and_extract(url)  # raises ValueError on failure

    # Step 2: Detect language
    detected_language = await asyncio.to_thread(detect_language, text)
    log.info("web_language_detected", language=detected_language)

    # Step 3: Summarise + Quiz in parallel
    s = get_settings()
    words = text.split()
    quiz_content = " ".join(words[:s.web_quiz_word_limit]) if len(words) > s.web_quiz_word_limit else text

    (_, notes_markdown), quiz_items = await asyncio.gather(
        run_summarisation(text, title, language=detected_language),
        generate_quiz(content=quiz_content, language=detected_language),
    )
    log.info("web_parallel_done", quiz_questions=len(quiz_items))

    # Step 4: Eval (graceful — never crashes the pipeline)
    try:
        eval_result = await evaluate_output(
            transcript=text,
            notes_md=notes_markdown,
            quiz=quiz_items,
        )
        eval_passed = eval_result.passed
        log.info("web_eval_done", overall=eval_result.overall, passed=eval_passed)
    except Exception as exc:
        log.error("web_eval_failed_gracefully", error=str(exc))
        eval_passed = True

    payload: dict[str, Any] = {
        "source_url": url,
        "title": title,
        "summary_markdown": notes_markdown,
        "quiz": [item.model_dump() for item in quiz_items],
        "eval_passed": eval_passed,
    }
    log.info("web_pipeline_complete", title=title, eval_passed=eval_passed)
    return payload


async def run_web_pipeline(url: str) -> dict[str, Any]:
    s = get_settings()
    key = web_key(url) if s.cache_ttl_seconds else None

    if key:
        cached, lock = await get_or_acquire(key)
        if cached is not None:
            logger.info("cache_hit", key=key)
            return cached
    else:
        lock = None

    try:
        result = await _run_web_pipeline_inner(url)
        if key and lock:
            _set(key, result, s.cache_ttl_seconds)
        return result
    finally:
        if lock:
            lock.release()
            if _locks.get(key) is lock:
                _locks.pop(key, None)
