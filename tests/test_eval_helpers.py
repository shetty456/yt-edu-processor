"""Tests for pure helpers in eval_service."""
import pytest

from app.services.eval_service import _extract_json, _format_quiz
from app.schemas import MCQItem, MCQOptions


def _make_item(q: str, answer: str = "A", desc: str = "Correct because X is the right choice.") -> MCQItem:
    # Pad short questions to satisfy the min-length constraint on MCQItem.question
    if len(q) < 10:
        q = q.ljust(10, "?")
    return MCQItem(
        question=q,
        options=MCQOptions(A="opt A", B="opt B", C="opt C", D="opt D"),
        answer=answer,
        description=desc,
    )


# ── _extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json_unchanged(self):
        s = '{"overall": "ok"}'
        assert _extract_json(s) == '{"overall": "ok"}'

    def test_strips_json_fence(self):
        s = '```json\n{"overall": "ok"}\n```'
        assert _extract_json(s) == '{"overall": "ok"}'

    def test_strips_plain_fence(self):
        s = '```\n{"overall": "ok"}\n```'
        assert _extract_json(s) == '{"overall": "ok"}'

    def test_strips_leading_text(self):
        s = 'Here is the JSON:\n{"overall": "ok"}'
        result = _extract_json(s)
        assert result.startswith("{")
        assert '"overall"' in result

    def test_strips_trailing_text(self):
        s = '{"overall": "ok"} some trailing text'
        result = _extract_json(s)
        assert result.endswith("}")
        assert "trailing" not in result

    def test_empty_string_returns_empty(self):
        result = _extract_json("")
        assert result == ""

    def test_no_brace_returns_as_is(self):
        s = "no json here"
        result = _extract_json(s)
        assert result == "no json here"

    def test_nested_json_preserved(self):
        s = '{"issues": [{"field": "quiz[0].answer", "severity": "fail"}]}'
        result = _extract_json(s)
        assert '"issues"' in result
        assert '"severity"' in result


# ── _format_quiz ──────────────────────────────────────────────────────────────

class TestFormatQuiz:
    def test_empty_quiz_returns_empty_string(self):
        assert _format_quiz([]) == ""

    def test_single_question_contains_q_label(self):
        items = [_make_item("Why does X cause Y?")]
        result = _format_quiz(items)
        assert "Q1." in result
        assert "Why does X cause Y?" in result

    def test_correct_answer_marked(self):
        items = [_make_item("Q?", answer="B")]
        result = _format_quiz(items)
        assert "B [CORRECT]" in result

    def test_wrong_answers_marked(self):
        items = [_make_item("Q?", answer="C")]
        result = _format_quiz(items)
        assert "A [wrong]" in result
        assert "B [wrong]" in result
        assert "D [wrong]" in result

    def test_description_truncated_to_150(self):
        long_desc = "X " * 200  # 400 chars
        items = [_make_item("Q?", desc=long_desc)]
        result = _format_quiz(items)
        # Description line should contain at most 150 chars of desc
        for line in result.splitlines():
            if "Description says" in line:
                assert len(line) < 200

    def test_multiple_questions_numbered(self):
        items = [_make_item(f"Question {i}?") for i in range(1, 4)]
        result = _format_quiz(items)
        assert "Q1." in result
        assert "Q2." in result
        assert "Q3." in result
