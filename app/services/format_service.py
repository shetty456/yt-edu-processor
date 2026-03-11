"""
format_service.py
──────────────────
Calls Sarvam-m to reformat raw quiz text into standard Q1:/Q2: structure.
Returns the formatted text string.
"""
from __future__ import annotations
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from app.config import get_settings
from app.utils import get_sarvam_client, logger, strip_think

settings = get_settings()
_client = get_sarvam_client()

_SYSTEM_PROMPT = """\
You are a quiz formatter. The user will give you raw quiz content that may be
messy, out of order, or in a non-standard format.

Reformat it into this EXACT structure — one question per block, separated by blank lines:

Q1: <question text>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Answer: <A/B/C/D>
Explanation: <explanation text, if present>

Rules:
- Number questions sequentially: Q1, Q2, Q3...
- Always use A. B. C. D. (capital letter + period + space) for options
- Always write "Answer: X" on its own line where X is A, B, C, or D
- If an explanation or description is present, include it as "Explanation: ..."
- Do NOT add, invent, or remove any content — only restructure
- Output ONLY the formatted questions — no intro text, no commentary, no markdown fences
- For multi-statement questions (i., ii., iii. or Statement I:), include all statements
  inside the question text before the options
- For Assertion-Reasoning format, include "Assertion (A): ..." and "Reason (R): ..."
  inside the question text before the options
"""


def _retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
        before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
        reraise=True,
    )


@_retry()
async def format_questions(text: str) -> str:
    log = logger.bind(service="format_questions")
    log.info("format_start", chars=len(text))

    resp = await _client.chat.completions.create(
        model=settings.sarvam_model,
        temperature=0.1,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text.strip()},
        ],
    )

    formatted = strip_think(resp.choices[0].message.content or "").strip()
    log.info("format_done", chars=len(formatted))
    return formatted
