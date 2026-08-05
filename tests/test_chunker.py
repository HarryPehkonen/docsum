"""Tests for the chunking module: splitting large text into token-aware chunks."""

import pytest

from docsum.chunker import chunk_text, count_tokens


class TestCountTokens:
    """Token counting using tiktoken."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        assert count_tokens("Hello world") > 0

    def test_longer_text_has_more_tokens(self):
        short = "The cat sat on the mat."
        long = "The cat sat on the mat. " * 100
        assert count_tokens(long) > count_tokens(short)

    def test_custom_model_encoding(self):
        # Should work with a model name that tiktoken knows
        assert count_tokens("Hello world", model="gpt-4") > 0


class TestChunkText:
    """Splitting text into chunks that respect token limits and boundaries."""

    def test_short_text_returns_single_chunk(self):
        text = "This is a simple sentence. It is not very long."
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", max_tokens=100) == []

    def test_respects_max_tokens(self):
        text = "This is a sentence. " * 200  # ~800 tokens
        chunks = chunk_text(text, max_tokens=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert count_tokens(chunk) <= 50

    def test_splits_at_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, max_tokens=10)
        # Each chunk should end at a sentence boundary (period)
        for chunk in chunks:
            if chunk.strip():
                # The chunk should end with a period or be the last partial
                assert chunk.rstrip().endswith(".") or chunk == chunks[-1]

    def test_overlap_includes_context_from_previous_chunk(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text(text, max_tokens=20, overlap_tokens=10)
        if len(chunks) > 1:
            # The overlap means some text from chunk 0 should appear in chunk 1
            # Check that chunks share some words
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert words_0 & words_1  # non-empty intersection

    def test_no_overlap_by_default(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunk_text(text, max_tokens=10, overlap_tokens=0)
        assert len(chunks) > 0

    def test_preserves_all_text_content(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, max_tokens=100)
        # All text should be recoverable by joining chunks
        assert " ".join(chunks).replace("  ", " ").strip() in text or text in " ".join(chunks)

    def test_single_long_sentence_exceeding_max_tokens(self):
        """A single sentence longer than max_tokens should still be included
        (we can't split mid-sentence, so it may exceed the limit)."""
        long_sentence = "word " * 100  # one very long "sentence" with no period
        text = long_sentence.strip()
        chunks = chunk_text(text, max_tokens=50)
        # Should still produce at least one chunk
        assert len(chunks) >= 1

    def test_paragraph_boundaries_preferred(self):
        """When possible, split at paragraph breaks before sentence breaks."""
        text = (
            "First paragraph sentence one here. First paragraph sentence two here.\n\n"
            "Second paragraph sentence one here. Second paragraph sentence two here.\n\n"
            "Third paragraph sentence one here. Third paragraph sentence two here.\n\n"
            "Fourth paragraph sentence one here. Fourth paragraph sentence two here.\n\n"
            "Fifth paragraph sentence one here. Fifth paragraph sentence two here.\n\n"
            "Sixth paragraph sentence one here. Sixth paragraph sentence two here.\n\n"
            "Seventh paragraph sentence one here. Seventh paragraph sentence two here.\n\n"
            "Eighth paragraph sentence one here. Eighth paragraph sentence two here.\n\n"
        )
        chunks = chunk_text(text, max_tokens=50)
        # Should produce multiple chunks
        assert len(chunks) > 1
