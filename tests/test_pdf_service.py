"""Tests for pure/deterministic functions in pdf_service."""
import pytest

from app.services.pdf_service import (
    detect_question_bank,
    detect_factual_content,
    _slugify,
    _ocr_quality_ok,
)


# ── detect_question_bank ─────────────────────────────────────────────────────

class TestDetectQuestionBank:
    def _make_qbank(self, n: int = 15) -> str:
        """Generate n numbered question lines."""
        return "\n".join(f"{i}. What is the capital of France?" for i in range(1, n + 1))

    def test_numbered_questions_detected(self):
        assert detect_question_bank(self._make_qbank(15)) is True

    def test_mcq_options_detected(self):
        lines = ["(a) Option one", "(b) Option two", "(c) Option three", "(d) Option four"] * 5
        assert detect_question_bank("\n".join(lines)) is True

    def test_q_prefix_detected(self):
        lines = [f"Q.{i}. Explain the concept." for i in range(1, 15)]
        assert detect_question_bank("\n".join(lines)) is True

    def test_textbook_prose_not_detected(self):
        prose = "\n".join([
            "The French Revolution began in 1789.",
            "It fundamentally transformed French society.",
            "The causes were complex and multifaceted.",
            "Economic hardship played a major role.",
            "Political instability also contributed significantly.",
            "The monarchy was eventually abolished.",
            "A republic was proclaimed in 1792.",
            "Napoleon rose to power shortly after.",
            "He consolidated the revolutionary gains.",
            "His reforms shaped modern France.",
            "The Code Napoleon standardized French law.",
            "It influenced legal systems worldwide.",
        ])
        assert detect_question_bank(prose) is False

    def test_too_few_lines_returns_false(self):
        short = "\n".join(f"{i}. Question here." for i in range(1, 6))
        assert detect_question_bank(short) is False

    def test_empty_string_returns_false(self):
        assert detect_question_bank("") is False

    def test_low_ratio_not_detected(self):
        """A doc with a few numbered lines but mostly prose should not trigger."""
        lines = (
            ["1. First question here."]
            + ["This is a paragraph of normal prose text that explains a concept."] * 20
        )
        assert detect_question_bank("\n".join(lines)) is False


# ── detect_factual_content ───────────────────────────────────────────────────

class TestDetectFactualContent:
    def test_date_heavy_text_detected(self):
        text = (
            "On 1945-05-08 the war ended. In 1917, the revolution occurred. "
            "The 1789 revolution changed France. By 1066 England was conquered. " * 10
        )
        assert detect_factual_content(text) is True

    def test_stats_heavy_text_detected(self):
        text = "GDP grew 3.5% in Q1 2023. Revenue was $1,200,000. Inflation hit 8.2%. " * 15
        assert detect_factual_content(text) is True

    def test_product_ids_not_detected(self):
        """4-digit product codes like 'Part 2024' or 'Table 1234' must not trigger."""
        text = (
            "See Table 2024 for details. Part 1234 is listed in Appendix 5678. "
            "Model 3456 is discontinued. Refer to section 7890 for more." * 10
        )
        assert detect_factual_content(text) is False

    def test_pure_prose_not_detected(self):
        text = (
            "The photosynthesis process converts light energy into chemical energy. "
            "Plants use carbon dioxide and water to produce glucose. "
            "Oxygen is released as a byproduct of this reaction. " * 15
        )
        assert detect_factual_content(text) is False

    def test_empty_string_returns_false(self):
        assert detect_factual_content("") is False

    def test_year_range_1800_detected(self):
        text = "The industrial revolution started around 1800. By 1850 it transformed Europe. " * 10
        assert detect_factual_content(text) is True

    def test_year_range_2100_detected(self):
        text = "Projections for 2100 show significant climate change. By 2150 impacts will be severe. " * 10
        assert detect_factual_content(text) is True


# ── _slugify ─────────────────────────────────────────────────────────────────

class TestSlugify:
    def test_basic_lowercase(self):
        assert _slugify("Hello World") == "hello-world"

    def test_underscores_become_hyphens(self):
        assert _slugify("hello_world") == "hello-world"

    def test_special_chars_stripped(self):
        assert _slugify("Hello! @World#") == "hello-world"

    def test_consecutive_hyphens_collapsed(self):
        assert _slugify("hello---world") == "hello-world"

    def test_empty_fallback(self):
        assert _slugify("!!!") == "document"

    def test_leading_trailing_hyphens_stripped(self):
        assert _slugify("-hello-") == "hello"


# ── _ocr_quality_ok ──────────────────────────────────────────────────────────

class TestOcrQualityOk:
    def test_clean_english_text_passes(self):
        text = "This is a clean English sentence with normal words and punctuation."
        assert _ocr_quality_ok(text) is True

    def test_empty_string_fails(self):
        assert _ocr_quality_ok("") is False

    def test_whitespace_only_fails(self):
        assert _ocr_quality_ok("   \n\t  ") is False

    def test_garbage_noise_fails(self):
        # Mostly non-alphanumeric chars — typical OCR garbage for unsupported script
        garbage = "§§¶¶©©®®±±×÷¿¡" * 20
        assert _ocr_quality_ok(garbage) is False

    def test_mixed_noise_below_threshold_fails(self):
        # Build a string where alnum chars are clearly < 40%
        # 30 noise chars + 10 alnum = 25% alnum — below 40% threshold
        noise = "@#$%^&*()!" * 3 + "abcdefghij"
        assert _ocr_quality_ok(noise) is False

    def test_hindi_text_passes(self):
        # Devanagari is alphanumeric (isalnum() returns True for Unicode letters)
        hindi = "यह एक हिंदी वाक्य है जो पढ़ने में सामान्य है।"
        assert _ocr_quality_ok(hindi) is True
