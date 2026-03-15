"""Tests for cache key functions and eviction logic."""
import hashlib
import urllib.parse
import pytest

from app import cache
from app.cache import yt_key, web_key, pdf_key


# ── yt_key ────────────────────────────────────────────────────────────────────

class TestYtKey:
    def test_standard_watch_url(self):
        key = yt_key("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert key == "yt:dQw4w9WgXcQ"

    def test_short_url(self):
        key = yt_key("https://youtu.be/dQw4w9WgXcQ")
        assert key == "yt:dQw4w9WgXcQ"

    def test_embed_url(self):
        key = yt_key("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert key == "yt:dQw4w9WgXcQ"

    def test_shorts_url(self):
        key = yt_key("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        assert key == "yt:dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self):
        assert yt_key("https://example.com/not-youtube") is None

    def test_same_video_different_formats_match(self):
        k1 = yt_key("https://www.youtube.com/watch?v=abcdefghijk")
        k2 = yt_key("https://youtu.be/abcdefghijk")
        assert k1 == k2


# ── web_key ───────────────────────────────────────────────────────────────────

class TestWebKey:
    def test_returns_web_prefix(self):
        key = web_key("https://example.com/article")
        assert key.startswith("web:")

    def test_trailing_slash_normalized(self):
        k1 = web_key("https://example.com/article")
        k2 = web_key("https://example.com/article/")
        assert k1 == k2

    def test_tracking_params_stripped(self):
        clean = web_key("https://example.com/page")
        with_utm = web_key("https://example.com/page?utm_source=newsletter&utm_medium=email")
        assert clean == with_utm

    def test_fbclid_stripped(self):
        clean = web_key("https://example.com/article")
        with_fbclid = web_key("https://example.com/article?fbclid=abc123")
        assert clean == with_fbclid

    def test_meaningful_query_params_preserved(self):
        k1 = web_key("https://example.com/search?q=python")
        k2 = web_key("https://example.com/search?q=javascript")
        assert k1 != k2

    def test_case_insensitive_host(self):
        k1 = web_key("https://Example.COM/page")
        k2 = web_key("https://example.com/page")
        assert k1 == k2


# ── pdf_key ───────────────────────────────────────────────────────────────────

class TestPdfKey:
    def test_returns_pdf_prefix(self):
        key = pdf_key(b"some pdf bytes")
        assert key.startswith("pdf:")

    def test_same_bytes_same_key(self):
        data = b"%PDF-1.4 test content"
        assert pdf_key(data) == pdf_key(data)

    def test_different_bytes_different_key(self):
        assert pdf_key(b"content one") != pdf_key(b"content two")

    def test_key_uses_sha256(self):
        data = b"test"
        expected = "pdf:" + hashlib.sha256(data).hexdigest()
        assert pdf_key(data) == expected


# ── Cache eviction ────────────────────────────────────────────────────────────

class TestCacheEviction:
    def setup_method(self):
        """Clear cache store before each test."""
        cache._store.clear()

    def test_set_and_get(self):
        cache._set("k1", "value1", ttl=60)
        assert cache._get("k1") == "value1"

    def test_expired_entry_returns_none(self):
        import time
        cache._set("k_expire", "val", ttl=0)
        # Manually set to expire immediately
        cache._store["k_expire"] = ("val", time.monotonic() - 1)
        assert cache._get("k_expire") is None

    def test_eviction_on_max_size(self, monkeypatch):
        """When cache is full, oldest entries are evicted."""
        from app.config import get_settings
        s = get_settings()
        # Fill cache to capacity
        for i in range(s.cache_max_size):
            cache._set(f"key_{i}", f"val_{i}", ttl=3600)
        assert len(cache._store) <= s.cache_max_size
        # Add one more — should trigger eviction
        cache._set("overflow_key", "overflow_val", ttl=3600)
        assert len(cache._store) <= s.cache_max_size
