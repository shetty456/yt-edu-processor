"""PDF validation, extraction, Cloudinary upload, and title inference."""
from __future__ import annotations

import asyncio
import io
import re

import cloudinary
import cloudinary.uploader
from pypdf import PdfReader

from app.config import get_settings
from app.utils import get_sarvam_client, logger

_TITLE_PROMPT = """\
Read the text below (first ~400 words of a document) and respond with ONLY a concise,
descriptive title for this document — 5 to 10 words, title-case, no punctuation at the end.
Do not explain, do not quote. Just the title.

Text:
{snippet}
"""


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


def _extract_text_sync(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(parts)


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
    return await asyncio.to_thread(_upload_sync, pdf_bytes, filename)


async def infer_pdf_title(text: str) -> str:
    snippet = " ".join(text.split()[:400])
    client = get_sarvam_client()
    s = get_settings()
    resp = await client.chat.completions.create(
        model=s.sarvam_model,
        messages=[{"role": "user", "content": _TITLE_PROMPT.format(snippet=snippet)}],
        temperature=0.2,
        max_tokens=30,
    )
    return resp.choices[0].message.content.strip().strip('"\'')
