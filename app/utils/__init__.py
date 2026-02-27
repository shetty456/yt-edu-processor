from __future__ import annotations
import asyncio
import sys
import logging
import structlog
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()

# ── Structured logger ─────────────────────────────────────────────────────────
_LOG_LEVEL = getattr(logging, settings.log_level.upper(), logging.INFO)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        (
            structlog.dev.ConsoleRenderer()
            if settings.log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer()
        ),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(_LOG_LEVEL),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
)

logger = structlog.get_logger()

# ── Concurrency semaphore ─────────────────────────────────────────────────────
_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
    return _semaphore


# ── LLM clients ───────────────────────────────────────────────────────────────


def get_sarvam_client() -> AsyncOpenAI:
    """Sarvam-m via its own endpoint — used for summarisation + quiz."""
    return AsyncOpenAI(
        api_key=settings.sarvam_api_key,
        base_url=settings.sarvam_base_url,
    )


def get_openrouter_client() -> AsyncOpenAI:
    """
    OpenRouter via OpenAI-compatible SDK.
    Used only for the eval pass — free model, no billing.
    Base URL is always https://openrouter.ai/api/v1
    """
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            # OpenRouter strongly recommends these headers for routing
            "HTTP-Referer": "https://github.com/yt-edu-processor",
            "X-Title": "YT Educational Processor",
        },
    )
