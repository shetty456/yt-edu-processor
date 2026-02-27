from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas import ProcessRequest, ProcessResponse
from app.services.pipeline import build_pipeline
from app.utils import get_semaphore, logger

settings = get_settings()


# -----------------------------
# Lifespan (startup / shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "startup",
        sarvam=settings.sarvam_model,
        eval=settings.openai_eval_model,
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
    if sem._value == 0:  # type: ignore[attr-defined]
        raise HTTPException(
            status_code=429,
            detail="Server busy — retry in 30 seconds.",
        )

    async with sem:
        log.info("processing_started")

        workflow = build_pipeline()

        try:
            run_output = await asyncio.wait_for(
                workflow.arun(input=body.youtube_url),
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

        # -----------------------------
        # Extract Final Output (NEW Agno Compatible)
        # -----------------------------
        if not run_output or not isinstance(run_output.content, dict):
            raise HTTPException(
                status_code=500,
                detail="Workflow returned empty output.",
            )

        payload: dict[str, Any] = run_output.content

        log.info(
            "processing_completed",
            title=payload.get("video_title"),
            eval_passed=payload.get("eval_passed"),
        )

        return JSONResponse(content=payload)