from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas import ProcessRequest, ProcessResponse, ProcessPDFResponse
from app.services.pipeline import run_pipeline, run_pdf_pipeline
from app.services.pdf_service import validate_pdf
from app.utils import get_semaphore, logger

settings = get_settings()


# -----------------------------
# Lifespan (startup / shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "startup",
        model=settings.sarvam_model,
        max_concurrent=settings.max_concurrent_jobs,
    )
    yield
    logger.info("shutdown")


app = FastAPI(
    title="YouTube Educational Content Processor",
    version="1.0.0",
    lifespan=lifespan,
)


# -----------------------------
# Request logging middleware
# -----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed = round(time.perf_counter() - start_time, 2)

    logger.info(
        "http",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed=elapsed,
    )

    return response


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# -----------------------------
# Main Processing Endpoint
# -----------------------------
@app.post("/process", response_model=ProcessResponse)
async def process_video(body: ProcessRequest) -> JSONResponse:
    sem = get_semaphore()
    log = logger.bind(url=body.youtube_url)

    # Basic concurrency guard
    if sem.locked():
        raise HTTPException(
            status_code=429,
            detail="Server busy — retry in 30 seconds.",
        )

    async with sem:
        log.info("processing_started")

        try:
            payload = await asyncio.wait_for(
                run_pipeline(
                    body.youtube_url,
                    transcript=body.transcript,
                    duration_seconds=body.duration_seconds,
                ),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Processing timed out. Try a shorter video.",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            )
        except Exception as exc:
            log.error("unexpected_error", error=str(exc))
            raise HTTPException(
                status_code=500,
                detail="Internal error. Please try again.",
            )

        log.info(
            "processing_completed",
            title=payload.get("video_title"),
            eval_passed=payload.get("eval_passed"),
        )

        return JSONResponse(content=payload)


# -----------------------------
# PDF Processing Endpoint
# -----------------------------
@app.post("/process-pdf", response_model=ProcessPDFResponse)
async def process_pdf(file: UploadFile = File(...)) -> JSONResponse:
    sem = get_semaphore()

    if file.content_type != "application/pdf" and not (file.filename or "").endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    try:
        await validate_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if sem.locked():
        raise HTTPException(status_code=429, detail="Server busy — retry in 30 seconds.")

    async with sem:
        log = logger.bind(filename=file.filename)
        log.info("pdf_processing_started")

        try:
            payload = await asyncio.wait_for(
                run_pdf_pipeline(pdf_bytes, file.filename or "document.pdf"),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Processing timed out.")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            log.error("pdf_unexpected_error", error=str(exc))
            raise HTTPException(status_code=500, detail="Internal error. Please try again.")

        log.info("pdf_processing_completed", title=payload.get("title"))
        return JSONResponse(content=payload)