"""Tests for pure helpers in quiz_service."""
import pytest

from app.services.quiz_service import _is_recall_question, _shuffle_answer
from app.schemas import MCQItem, MCQOptions


def _make_item(q: str, answer: str = "A") -> MCQItem:
    # Pad short questions to satisfy the min-length constraint on MCQItem.question
    if len(q) < 10:
        q = q.ljust(10, "?")
    return MCQItem(
        question=q,
        options=MCQOptions(A="opt A", B="opt B", C="opt C", D="opt D"),
        answer=answer,
        description="Correct because the answer key is A.",
    )


# ── _is_recall_question ───────────────────────────────────────────────────────

class TestIsRecallQuestion:
    def test_how_many_is_recall(self):
        assert _is_recall_question(_make_item("How many chromosomes does a human cell have?"))

    def test_what_is_is_recall(self):
        assert _is_recall_question(_make_item("What is the role of mitochondria?"))

    def test_define_is_recall(self):
        assert _is_recall_question(_make_item("Define osmosis."))

    def test_which_structure_is_recall(self):
        assert _is_recall_question(_make_item("Which structure carries oxygen in the blood?"))

    def test_analytical_why_not_recall(self):
        assert not _is_recall_question(
            _make_item("Why does premature callase activity lead to male sterility?")
        )

    def test_causal_what_would_happen_not_recall(self):
        assert not _is_recall_question(
            _make_item("What would happen to enzyme activity if temperature doubled?")
        )

    def test_distinguishes_not_recall(self):
        assert not _is_recall_question(
            _make_item("What distinguishes bicellular from tricellular pollen grains?")
        )

    def test_short_question_not_recall(self):
        # A padded short question (no recall prefix) should return False
        assert not _is_recall_question(_make_item("Explain why"))


# ── _shuffle_answer ───────────────────────────────────────────────────────────

class TestShuffleAnswer:
    def test_returns_mcq_item(self):
        item = _make_item("Q?", answer="A")
        result = _shuffle_answer(item)
        assert isinstance(result, MCQItem)

    def test_answer_is_valid_letter(self):
        item = _make_item("Q?", answer="A")
        for _ in range(20):
            result = _shuffle_answer(item)
            assert result.answer in ("A", "B", "C", "D")

    def test_correct_option_text_preserved(self):
        """The text of the originally-correct option must follow the correct answer."""
        item = MCQItem(
            question="Why does X cause Y in the system?",
            options=MCQOptions(A="alpha", B="beta", C="gamma", D="delta"),
            answer="A",
            description="Correct because answer is A and explains mechanism.",
        )
        result = _shuffle_answer(item)
        # whichever letter result.answer points to should hold "alpha"
        assert getattr(result.options, result.answer) == "alpha"

    def test_no_option_text_lost(self):
        item = MCQItem(
            question="What distinguishes process X from process Y?",
            options=MCQOptions(A="alpha", B="beta", C="gamma", D="delta"),
            answer="B",
            description="Correct because the answer is B and covers both concepts.",
        )
        result = _shuffle_answer(item)
        option_texts = {result.options.A, result.options.B, result.options.C, result.options.D}
        assert option_texts == {"alpha", "beta", "gamma", "delta"}
