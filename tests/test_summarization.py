"""Tests for chunk_transcript in summarization_service."""
import pytest

from app.services.summarization_service import chunk_transcript


class TestChunkTranscript:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short text. It has only a few words."
        chunks = chunk_transcript(text, word_limit=100, target_words=50)
        assert chunks == [text]

    def test_english_sentences_split_correctly(self):
        # Build text > word_limit so chunking fires
        sentence = "The quick brown fox jumps over the lazy dog. "
        text = sentence * 30  # ~270 words
        chunks = chunk_transcript(text, word_limit=50, target_words=30)
        assert len(chunks) > 1
        # Every chunk should be non-empty
        assert all(c.strip() for c in chunks)

    def test_all_words_preserved(self):
        sentence = "Hello world this is a test sentence with several words. "
        text = sentence * 20
        chunks = chunk_transcript(text, word_limit=30, target_words=20)
        original_words = text.split()
        rejoined_words = " ".join(chunks).split()
        assert len(original_words) == len(rejoined_words)

    def test_hindi_danda_splits(self):
        """Hindi text using । as sentence terminator must produce multiple chunks."""
        sentence = "यह एक हिंदी वाक्य है जो परीक्षण के लिए लिखा गया है। "
        text = sentence * 30  # ~150+ words
        chunks = chunk_transcript(text, word_limit=30, target_words=20)
        assert len(chunks) > 1

    def test_hindi_no_chunking_without_fix_regression(self):
        """Confirm that a Hindi text with ।  produces >1 chunk (regression guard for Fix 1)."""
        # 60 copies of a ~7-word sentence = ~420 words, well above any limit
        sentence = "राम ने एक सेब खाया। "
        text = sentence * 60
        chunks = chunk_transcript(text, word_limit=50, target_words=30)
        assert len(chunks) > 1, (
            "Hindi text with । should split into multiple chunks. "
            "Regression: chunk_transcript may be missing the danda in its split regex."
        )

    def test_no_sentence_boundaries_falls_back_gracefully(self):
        """Text with no punctuation still produces chunks (word-count split)."""
        words = ["word"] * 200
        text = " ".join(words)
        chunks = chunk_transcript(text, word_limit=50, target_words=30)
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_exact_word_limit_boundary(self):
        """Text exactly at word_limit should NOT be split."""
        text = " ".join(["word"] * 100)
        chunks = chunk_transcript(text, word_limit=100, target_words=50)
        assert len(chunks) == 1

    def test_exclamation_and_question_marks_split(self):
        sentence_e = "What a wonderful day! "
        sentence_q = "Is this working correctly? "
        text = (sentence_e + sentence_q) * 20
        chunks = chunk_transcript(text, word_limit=20, target_words=12)
        assert len(chunks) > 1
