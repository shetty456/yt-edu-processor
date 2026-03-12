from __future__ import annotations

import asyncio
import re
import socket
from ipaddress import ip_address
from typing import Any

import httpx
import trafilatura

from app.config import get_settings
from app.utils import logger

# ---------------------------------------------------------------------------
# Blocked domains (adult / gambling — checked before any network call)
# ---------------------------------------------------------------------------

_BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "pornhub.com",
    "xvideos.com",
    "xhamster.com",
    "redtube.com",
    "youporn.com",
    "tube8.com",
    "xtube.com",
    "beeg.com",
    "spankbang.com",
    "xnxx.com",
    "xfantasy.com",
    "chaturbate.com",
    "livejasmin.com",
    "myfreecams.com",
    "onlyfans.com",
    "bet365.com",
    "draftkings.com",
    "fanduel.com",
    "bovada.lv",
    "pokerstars.com",
    "888casino.com",
    "ladbrokes.com",
    "williamhill.com",
    "betway.com",
})

# ---------------------------------------------------------------------------
# Explicit content keywords (minimal, unambiguous — avoids medical FP)
# ---------------------------------------------------------------------------

_EXPLICIT_KEYWORDS: frozenset[str] = frozenset({
    "pornography",
    "hardcore sex",
    "nude photos",
    "naked pictures",
    "sex tape",
    "xxx video",
    "adult film",
    "erotic video",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_from_tag(html: str) -> str | None:
    """Extract page title from <title>…</title> as a fallback."""
    m = re.search(r"<title[^>]*>([^<]{1,300})</title>", html, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _ip_is_private(addr_str: str) -> bool:
    """Return True if the string is a private/loopback/link-local IP."""
    try:
        addr = ip_address(addr_str)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False


def _resolve_and_check(hostname: str) -> bool:
    """Blocking DNS lookup — run via asyncio.to_thread. Returns True if private."""
    # Fast path: literal IP address (no DNS needed)
    try:
        addr = ip_address(hostname)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        pass
    # Slow path: DNS resolution (blocking — must be called in a thread)
    try:
        results = socket.getaddrinfo(hostname, None)
        return any(_ip_is_private(sockaddr[0]) for *_, sockaddr in results)
    except Exception:
        return False


async def _is_private_host(hostname: str) -> bool:
    """Async wrapper: resolves hostname in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(_resolve_and_check, hostname)


def _check_blocked_domain(url: str) -> bool:
    """Return True if the URL's domain is in the blocklist."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        # strip port and www.
        host = host.split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        # check exact match and parent domain suffix
        if host in _BLOCKED_DOMAINS:
            return True
        for blocked in _BLOCKED_DOMAINS:
            if host.endswith("." + blocked):
                return True
    except Exception:
        pass
    return False


def _check_explicit(text: str) -> bool:
    """Return True if text (first 500 words) contains explicit keywords."""
    words = text.split()
    sample = " ".join(words[:500]).lower()
    for kw in _EXPLICIT_KEYWORDS:
        if kw in sample:
            return True
    return False


async def _validate_url(url: str) -> None:
    """Raise ValueError with a user-facing message for disallowed URL types."""
    lower = url.lower()

    # YouTube URLs
    if "youtube.com" in lower or "youtu.be" in lower:
        raise ValueError(
            "This looks like a YouTube URL. Use the /process endpoint for YouTube videos."
        )

    # PDF suffix
    parsed_path = url.split("?")[0].split("#")[0]
    if parsed_path.lower().endswith(".pdf"):
        raise ValueError(
            "This URL points to a PDF file. Use the /process-pdf endpoint to upload PDFs directly."
        )

    # Private/loopback IPs — DNS lookup runs in a thread
    try:
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or ""
        if hostname and await _is_private_host(hostname):
            raise ValueError(
                "URLs pointing to private or local network addresses are not allowed."
            )
    except ValueError:
        raise
    except Exception:
        pass


def _extract_text(html: str, url: str) -> tuple[str, str]:
    """Run trafilatura synchronously; returns (title, text)."""
    text = trafilatura.extract(
        html,
        url=url,
        include_tables=False,
        include_comments=False,
        favor_precision=True,
        deduplicate=True,
    )
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta and meta.title else None) or _title_from_tag(html) or "Untitled"
    return title, text or ""


async def _fetch_html(url: str, timeout: int) -> tuple[str, str]:
    """Fetch URL and return (html_text, final_url). Raises ValueError on failure."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; EduBot/1.0)"},
            )
    except httpx.TimeoutException:
        raise ValueError("The page took too long to load. Try a different URL.")
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.InvalidURL):
        raise ValueError(
            "Could not reach that URL. Check that the address is correct and the site is publicly accessible."
        )
    except Exception as exc:
        logger.warning("web_fetch_error", error=str(exc))
        raise ValueError(
            "Could not reach that URL. Check that the address is correct and the site is publicly accessible."
        )

    # HTTP status checks
    status = response.status_code
    if status in (401, 403):
        raise ValueError(
            "This page requires authentication. Only publicly accessible pages are supported."
        )
    if status == 404:
        raise ValueError(
            "Page not found (404). The URL may be broken or the article may have been removed."
        )
    if 400 <= status < 500:
        raise ValueError(
            f"The server returned an error ({status}). The page may not be publicly accessible."
        )
    if status >= 500:
        raise ValueError("The target website returned a server error. Try again later.")

    # Content-type check
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        if "pdf" in content_type or "application/octet-stream" in content_type:
            raise ValueError(
                "This URL points to a PDF file. Use the /process-pdf endpoint to upload PDFs directly."
            )
        raise ValueError(
            "The URL does not point to a web page. Only HTML articles and blog posts are supported."
        )

    return response.text, str(response.url)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_and_extract(url: str) -> tuple[str, str]:
    """
    Fetch a web URL and extract readable article text.

    Returns:
        (title, text)

    Raises:
        ValueError: user-facing message for every failure mode.
    """
    s = get_settings()
    log = logger.bind(url=url)

    # Pre-flight checks (DNS lookup runs in thread via _validate_url)
    await _validate_url(url)

    if _check_blocked_domain(url):
        raise ValueError("This content is not supported.")

    log.info("web_fetch_start")

    # Fetch HTML
    html, final_url = await _fetch_html(url, timeout=s.web_fetch_timeout_seconds)

    # Re-check domain after redirect — short URLs may redirect to blocked sites
    if final_url != url and _check_blocked_domain(final_url):
        raise ValueError("This content is not supported.")

    # Extract text in thread (trafilatura is CPU-bound)
    title, text = await asyncio.to_thread(_extract_text, html, final_url)

    if not text.strip():
        raise ValueError(
            "No readable article content could be extracted from that page. "
            "The page may be JavaScript-only, paywalled, or mostly non-text."
        )

    word_count = len(text.split())
    if word_count < s.web_min_word_count:
        raise ValueError(
            f"The page doesn't contain enough text to process "
            f"({word_count} words found; minimum is {s.web_min_word_count})."
        )

    if _check_explicit(text):
        raise ValueError("This content is not supported.")

    log.info("web_extract_done", title=title, words=word_count)
    return title, text
