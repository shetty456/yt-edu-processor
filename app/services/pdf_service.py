"""PDF validation, extraction, Cloudinary upload, and title inference."""
from __future__ import annotations

import asyncio
import io
import re

import cloudinary
import cloudinary.uploader
from pypdf import PdfReader

from app.config import get_settings
from app.utils import get_sarvam_client, logger, strip_think

_LANG_CODE_TO_NAME: dict[str, str] = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "ar": "Arabic",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


def detect_language(text: str) -> str:
    """Detect the primary language of *text*; returns a human-readable name.

    Uses the first ~1000 characters for speed.  Falls back to ``"English"``
    if detection fails, the sample is too short, or the code is not in the
    mapping.
    """
    sample = text.strip()[:1000]
    # langdetect is unreliable on very short or non-linguistic samples
    if len(sample) < 20:
        return "English"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42  # make detection deterministic
        code = detect(sample)
        return _LANG_CODE_TO_NAME.get(code, "English")
    except Exception as exc:
        logger.warning("language_detection_failed", error=str(exc))
        return "English"

_TITLE_PROMPT = """\
Read the text below (first ~400 words of a document) and respond with ONLY a concise,
descriptive title for this document — 5 to 10 words, title-case, no punctuation at the end.
Do not explain, do not quote. Just the title.{lang_instruction}

Text:
{snippet}
"""

_TITLE_LANG_INSTRUCTION = "\nRespond in {language}."


def _init_cloudinary() -> None:
    s = get_settings()
    cloudinary.config(
        cloud_name=s.cloudinary_cloud_name,
        api_key=s.cloudinary_api_key,
        api_secret=s.cloudinary_api_secret,
        secure=True,
    )


def _validate_and_count_pages(pdf_bytes: bytes) -> int:
    """Raises ValueError if invalid; returns page count."""
    s = get_settings()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > s.max_pdf_size_mb:
        raise ValueError(f"PDF exceeds {s.max_pdf_size_mb} MB limit ({size_mb:.1f} MB).")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = len(reader.pages)
    if pages > s.max_pdf_pages:
        raise ValueError(f"PDF has {pages} pages; maximum allowed is {s.max_pdf_pages}.")
    return pages


_OCR_FALLBACK_THRESHOLD = 50  # chars; below this we assume scanned


def _extract_text_sync(pdf_bytes: bytes) -> str:
    s = get_settings()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(parts)

    if s.ocr_enabled and len(text.strip()) < _OCR_FALLBACK_THRESHOLD:
        logger.info("ocr_fallback_triggered", extracted_chars=len(text.strip()))
        text = _extract_text_with_ocr_sync(pdf_bytes)

    return text


def _ocr_quality_ok(text: str) -> bool:
    """Returns False if OCR output looks like garbage (< 40% alphanumeric + space chars)."""
    if not text.strip():
        return False
    alnum_or_space = sum(c.isalnum() or c.isspace() for c in text)
    return (alnum_or_space / len(text)) > 0.40


def _extract_text_with_ocr_sync(pdf_bytes: bytes) -> str:
    """OCR fallback for image-based/scanned PDFs.

    Returns empty string if OCR libraries are not installed or all pages fail.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError as exc:
        logger.warning("ocr_libraries_not_installed", error=str(exc))
        return ""

    s = get_settings()
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=s.max_pdf_pages)
    except Exception as exc:
        logger.warning("ocr_pdf_render_failed", error=str(exc))
        return ""

    parts = []
    for i, image in enumerate(images):
        try:
            parts.append(pytesseract.image_to_string(image, lang="eng+hin"))
        except Exception as exc:
            logger.warning("ocr_page_failed", page=i + 1, error=str(exc))
    result = "\n\n".join(parts)
    if not _ocr_quality_ok(result):
        logger.warning("ocr_quality_check_failed", chars=len(result))
        return ""
    return result


def _slugify(name: str) -> str:
    """Convert an arbitrary filename stem into a Cloudinary-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)   # strip non-word chars except hyphens
    slug = re.sub(r"[\s_]+", "-", slug)     # spaces/underscores → hyphen
    slug = re.sub(r"-{2,}", "-", slug)      # collapse consecutive hyphens
    return slug.strip("-") or "document"


def _upload_sync(pdf_bytes: bytes, filename: str) -> str:
    _init_cloudinary()
    stem = filename.rsplit(".", 1)[0]
    public_id = f"edu-pdfs/{_slugify(stem)}"
    result = cloudinary.uploader.upload(
        pdf_bytes,
        resource_type="raw",
        public_id=public_id,
        overwrite=False,
        use_filename=True,
    )
    return result["secure_url"]


async def validate_pdf(pdf_bytes: bytes) -> int:
    """Async wrapper — raises ValueError on violation, returns page count."""
    return await asyncio.to_thread(_validate_and_count_pages, pdf_bytes)


async def extract_pdf_text(pdf_bytes: bytes) -> str:
    return await asyncio.to_thread(_extract_text_sync, pdf_bytes)


async def upload_pdf_to_cloudinary(pdf_bytes: bytes, filename: str) -> str:
    try:
        return await asyncio.to_thread(_upload_sync, pdf_bytes, filename)
    except Exception as exc:
        logger.warning("cloudinary_upload_failed", error=str(exc))
        return ""  # pipeline continues; pdf_url will be empty in response


def detect_factual_content(text: str) -> bool:
    """Returns True if text is dominated by dates, numbers, and statistics."""
    words = text.split()
    if not words:
        return False
    numeric_pattern = re.compile(
        r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?%?'     # numbers, percentages
        r'|(?:1[89]\d{2}|20[0-9]{2}|21\d{2})'        # years 1800–2199
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}'  # month dates
        r'|Q[1-4]\s*\d{4})\b',                     # quarters
        re.IGNORECASE
    )
    matches = numeric_pattern.findall(text[:3000])  # check first 3000 chars
    density = len(matches) / max(len(words[:500]), 1)
    return density > 0.06  # >6% of tokens are numeric/date = factual


def detect_question_bank(text: str) -> bool:
    """Return True if the text looks like a question bank rather than study material.

    Heuristic: checks what fraction of non-empty lines start with patterns
    common in question papers (numbered questions, MCQ options, Q. prefixes).
    Requires both a minimum absolute count and a high ratio to avoid false
    positives from textbooks with a few numbered sections or examples.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 10:
        return False

    q_pattern = re.compile(
        r'^(?:'
        r'[1-9]\d{0,2}[\.\)]\s|'  # 1–999 followed by ". " or ") " (question numbers)
        r'Q\.?\s*\d+[\.\):\s]|'   # Q.1. / Q1: / Q 1
        r'[Qq]uestion\s+\d+|'     # Question 1
        r'\([a-dA-D]\)\s'         # (a) / (b) MCQ option
        r')'
    )
    q_lines = sum(1 for l in lines if q_pattern.match(l))
    # Require at least 8 matching lines (absolute) AND >35% ratio.
    # The higher ratio + absolute floor avoids rejecting textbooks that happen
    # to have a handful of numbered section headers or worked examples.
    return q_lines >= 8 and (q_lines / len(lines)) > 0.35


async def infer_pdf_title(
    text: str,
    fallback: str = "Untitled Document",
    language: str = "English",
) -> str:
    snippet = " ".join(text.split()[:400])
    client = get_sarvam_client()
    s = get_settings()
    lang_instruction = (
        _TITLE_LANG_INSTRUCTION.format(language=language)
        if language != "English"
        else ""
    )
    prompt = _TITLE_PROMPT.format(snippet=snippet, lang_instruction=lang_instruction)
    try:
        resp = await client.chat.completions.create(
            model=s.sarvam_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=30,
        )
        raw = strip_think(resp.choices[0].message.content or "").strip('"\'')
        return raw if raw else fallback
    except Exception as exc:
        logger.warning("title_inference_failed", error=str(exc), fallback=fallback)
        return fallback
