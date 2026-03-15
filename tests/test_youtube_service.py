"""Tests for pure functions in youtube_service."""
import pytest

from app.services.youtube_service import _clean, _is_educational, _is_clearly_non_educational


# ── _clean ────────────────────────────────────────────────────────────────────

class TestClean:
    def test_removes_bracket_noise(self):
        raw = "Hello [Music] world [Applause] this is text"
        result = _clean(raw)
        assert "[Music]" not in result
        assert "[Applause]" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_parenthetical_noise(self):
        raw = "Some speech (inaudible) and more speech (background noise)"
        result = _clean(raw)
        assert "(inaudible)" not in result
        assert "Some speech" in result

    def test_collapses_newlines(self):
        raw = "Line one\nLine two\n\nLine three"
        result = _clean(raw)
        assert "\n" not in result

    def test_collapses_extra_spaces(self):
        raw = "word1  word2   word3"
        result = _clean(raw)
        assert "  " not in result

    def test_deduplicates_repeated_ngrams(self):
        """Auto-caption loop artefact: 8-word ngram repeated multiple times."""
        phrase = "the quick brown fox jumps over the lazy dog "
        # Repeat the phrase enough times to trigger dedup
        raw = phrase * 5
        result = _clean(raw)
        result_words = result.split()
        original_words = (phrase * 5).split()
        # Result should be shorter than original due to dedup
        assert len(result_words) < len(original_words)

    def test_punctuation_spacing_fixed(self):
        raw = "Hello , world ! How are you ?"
        result = _clean(raw)
        assert " ," not in result
        assert " !" not in result

    def test_empty_string(self):
        assert _clean("") == ""

    def test_only_noise_returns_empty_or_minimal(self):
        raw = "[Music] [Applause] [Laughter]"
        result = _clean(raw)
        assert result.strip() == ""


# ── _is_educational ───────────────────────────────────────────────────────────

class TestIsEducational:
    def test_tutorial_keyword(self):
        assert _is_educational("Python Tutorial for Beginners", "Tech Channel") is True

    def test_lecture_keyword(self):
        assert _is_educational("MIT OpenCourse Lecture on Algorithms", "MIT") is True

    def test_university_in_author(self):
        assert _is_educational("Introduction to Physics", "Stanford University") is True

    def test_howto_keyword(self):
        assert _is_educational("How to Build a REST API", "Dev School") is True

    def test_non_educational_video(self):
        assert _is_educational("My Summer Vlog Day 5", "Travel Blogger") is False

    def test_case_insensitive(self):
        assert _is_educational("MACHINE LEARNING EXPLAINED", "AI Lab") is True


# ── _is_clearly_non_educational ───────────────────────────────────────────────

class TestIsNonEducational:
    def test_gameplay_detected(self):
        assert _is_clearly_non_educational("Let's Play Minecraft Episode 5", "Gamer") is True

    def test_music_video_detected(self):
        assert _is_clearly_non_educational("My Song - Official Music Video", "Artist") is True

    def test_vlog_detected(self):
        assert _is_clearly_non_educational("Daily Vlog - Day in my life", "Creator") is True

    def test_reaction_video_detected(self):
        assert _is_clearly_non_educational("Reacting to viral videos", "Reactor") is True

    def test_educational_not_flagged(self):
        assert _is_clearly_non_educational("Python Tutorial for Beginners", "Tech") is False

    def test_neutral_title_not_flagged(self):
        assert _is_clearly_non_educational("Documentary on World History", "History Channel") is False
