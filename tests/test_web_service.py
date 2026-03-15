"""Tests for pure/sync functions in web_service."""
import pytest

from app.services.web_service import _ip_is_private, _check_blocked_domain, _check_explicit, _is_paywall_html


# ── _ip_is_private ────────────────────────────────────────────────────────────

class TestIpIsPrivate:
    def test_loopback_ipv4(self):
        assert _ip_is_private("127.0.0.1") is True

    def test_private_10_range(self):
        assert _ip_is_private("10.0.0.1") is True

    def test_private_192_168(self):
        assert _ip_is_private("192.168.1.100") is True

    def test_private_172_16(self):
        assert _ip_is_private("172.16.0.1") is True

    def test_link_local(self):
        assert _ip_is_private("169.254.0.1") is True

    def test_loopback_ipv6(self):
        assert _ip_is_private("::1") is True

    def test_public_ip(self):
        assert _ip_is_private("8.8.8.8") is False

    def test_another_public_ip(self):
        assert _ip_is_private("1.1.1.1") is False

    def test_invalid_string_returns_false(self):
        assert _ip_is_private("not-an-ip") is False

    def test_empty_string_returns_false(self):
        assert _ip_is_private("") is False


# ── _check_blocked_domain ─────────────────────────────────────────────────────

class TestCheckBlockedDomain:
    def test_blocked_domain_detected(self):
        assert _check_blocked_domain("https://pornhub.com/video/123") is True

    def test_blocked_with_www(self):
        assert _check_blocked_domain("https://www.onlyfans.com/profile") is True

    def test_blocked_subdomain(self):
        assert _check_blocked_domain("https://sub.xvideos.com/page") is True

    def test_gambling_domain_blocked(self):
        assert _check_blocked_domain("https://bet365.com/sports") is True

    def test_clean_domain_allowed(self):
        assert _check_blocked_domain("https://wikipedia.org/wiki/Python") is False

    def test_educational_domain_allowed(self):
        assert _check_blocked_domain("https://mit.edu/courses") is False

    def test_partial_name_match_not_blocked(self):
        # "xvideos" appearing in a path should not block a clean domain
        assert _check_blocked_domain("https://example.com/review-xvideos-app") is False


# ── _check_explicit ───────────────────────────────────────────────────────────

class TestCheckExplicit:
    def test_explicit_keyword_detected(self):
        text = "This article is about pornography and its effects on society."
        assert _check_explicit(text) is True

    def test_clean_text_passes(self):
        text = (
            "Python is a high-level programming language known for its readability. "
            "It is widely used in data science, web development, and automation."
        )
        assert _check_explicit(text) is False

    def test_medical_context_not_flagged(self):
        text = (
            "The study examined sexual health in adolescents. "
            "Participants reported various symptoms. "
            "The research was conducted by university doctors."
        )
        assert _check_explicit(text) is False

    def test_only_first_1000_words_checked(self):
        # Place explicit keyword at word 1100 — should not be detected
        filler = ("safe word " * 1000)
        explicit_tail = "pornography is discussed here"
        text = filler + explicit_tail
        assert _check_explicit(text) is False

    def test_keyword_within_1000_words_detected(self):
        # Place explicit keyword at word 800 — should be detected
        filler = ("safe word " * 400)
        text = filler + "pornography and its effects"
        assert _check_explicit(text) is True

    def test_empty_text_passes(self):
        assert _check_explicit("") is False


# ── _is_paywall_html ──────────────────────────────────────────────────────────

class TestIsPaywallHtml:
    def test_subscribe_to_continue(self):
        html = "<html><body>Subscribe to continue reading this article.</body></html>"
        assert _is_paywall_html(html) is True

    def test_sign_in_to_read(self):
        html = "<html><body>Sign in to read the full story.</body></html>"
        assert _is_paywall_html(html) is True

    def test_members_only(self):
        html = "<html><body>This is a members-only article.</body></html>"
        assert _is_paywall_html(html) is True

    def test_paywall_keyword(self):
        html = "<html><body>This article is behind a paywall.</body></html>"
        assert _is_paywall_html(html) is True

    def test_clean_article_not_flagged(self):
        html = (
            "<html><body><h1>Introduction to Python</h1>"
            "<p>Python is a high-level programming language.</p></body></html>"
        )
        assert _is_paywall_html(html) is False

    def test_only_first_5000_chars_checked(self):
        # Paywall pattern placed past 5000 chars — should not trigger
        filler = "x" * 5100
        html = filler + "subscribe to continue reading"
        assert _is_paywall_html(html) is False
