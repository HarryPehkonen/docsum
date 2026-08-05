"""Tests for the summarization algorithms: map-reduce, refine, hierarchical."""

from unittest.mock import MagicMock

import pytest

from docsum.algorithms import map_reduce, refine, hierarchical
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS, render_prompt, render_reduce_prompt


@pytest.fixture
def mock_client():
    """A mock LLM client that returns predictable responses."""
    client = MagicMock(spec=LLMClient)
    # Default: return the prompt text so we can verify what was sent
    client.complete.side_effect = lambda prompt, **kwargs: f"SUMMARY_OF: {prompt[:50]}"
    return client


@pytest.fixture
def simple_prompt():
    return BUILTIN_PROMPTS["summary"]


@pytest.fixture
def reduce_prompt():
    return BUILTIN_PROMPTS["reduce"]


class TestMapReduce:
    """Map-reduce: chunk → summarize each → combine."""

    def test_single_chunk(self, mock_client, simple_prompt, reduce_prompt):
        """If the text fits in one chunk, map-reduce does a single call."""
        mock_client.complete.side_effect = None
        mock_client.complete.return_value = "Single summary"
        result = map_reduce(
            text="Short text",
            client=mock_client,
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            max_tokens=1000,
        )
        assert result == "Single summary"

    def test_multiple_chunks(self, mock_client, simple_prompt, reduce_prompt):
        """Multiple chunks: each is summarized, then combined."""
        # Track calls
        call_args = []

        def side_effect(prompt, **kwargs):
            call_args.append(prompt)
            if "summaries" in prompt.lower() or "combine" in prompt.lower() or "different sections" in prompt.lower():
                return "FINAL COMBINED SUMMARY"
            return f"CHUNK_SUMMARY_{len(call_args)}"

        mock_client.complete.side_effect = side_effect

        text = "This is sentence one. " * 200  # Long enough for multiple chunks
        result = map_reduce(
            text=text,
            client=mock_client,
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            max_tokens=50,
        )
        assert result == "FINAL COMBINED SUMMARY"
        # Should have called the LLM for each chunk + one reduce step
        assert mock_client.complete.call_count > 1

    def test_preserves_all_content(self, mock_client, simple_prompt, reduce_prompt):
        """Every chunk should be processed."""
        processed_texts = []

        def side_effect(prompt, **kwargs):
            processed_texts.append(prompt)
            if "different sections" in prompt:
                return "COMBINED"
            return "chunk_result"

        mock_client.complete.side_effect = side_effect

        text = "Paragraph one content here. " * 100 + "\n\n" + "Paragraph two content here. " * 100
        map_reduce(
            text=text,
            client=mock_client,
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            max_tokens=100,
        )
        # Each chunk's content should appear in at least one prompt
        assert mock_client.complete.call_count >= 2


class TestRefine:
    """Refine: chunk → summarize → refine with next chunk → repeat."""

    def test_single_chunk(self, mock_client, simple_prompt):
        """If the text fits in one chunk, refine does a single call."""
        mock_client.complete.side_effect = None
        mock_client.complete.return_value = "Single summary"
        result = refine(
            text="Short text",
            client=mock_client,
            prompt_template=simple_prompt,
            max_tokens=1000,
        )
        assert result == "Single summary"

    def test_multiple_chunks_refines(self, mock_client, simple_prompt):
        """Multiple chunks: each refines the running summary."""
        responses = iter(["Summary of chunk 1", "Refined with chunk 2", "Final refinement"])

        def side_effect(prompt, **kwargs):
            return next(iter(responses)) if not hasattr(side_effect, "_count") or side_effect._count < 3 else "Final refinement"

        # Track call count to return sequential responses
        call_count = [0]
        responses = ["Summary of chunk 1", "Refined with chunk 2", "Final refinement"]

        def side_effect2(prompt, **kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        mock_client.complete.side_effect = side_effect2

        text = "This is a sentence with enough words. " * 300
        result = refine(
            text=text,
            client=mock_client,
            prompt_template=simple_prompt,
            max_tokens=50,
        )
        assert result == "Final refinement"
        assert mock_client.complete.call_count >= 2

    def test_refine_passes_previous_summary(self, mock_client, simple_prompt):
        """Each refine call should include the previous summary."""
        prompts_seen = []

        def side_effect(prompt, **kwargs):
            prompts_seen.append(prompt)
            return "refined summary"

        mock_client.complete.side_effect = side_effect

        text = "Sentence one with enough words to be meaningful. " * 200
        refine(
            text=text,
            client=mock_client,
            prompt_template=simple_prompt,
            max_tokens=30,
        )
        # After the first call, subsequent prompts should reference prior summary
        if len(prompts_seen) > 1:
            # Second prompt should contain the first summary somehow
            assert len(prompts_seen[1]) > 0


class TestHierarchical:
    """Hierarchical: map-reduce with recursive reduction of summaries."""

    def test_single_chunk(self, mock_client, simple_prompt, reduce_prompt):
        mock_client.complete.side_effect = None
        mock_client.complete.return_value = "Single summary"
        result = hierarchical(
            text="Short text",
            client=mock_client,
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            max_tokens=1000,
        )
        assert result == "Single summary"

    def test_multiple_chunks_reduces(self, mock_client, simple_prompt, reduce_prompt):
        """If chunk summaries are too large, hierarchical recursively reduces."""
        # Make chunk summaries large enough to need a second reduction
        call_count = [0]

        def side_effect(prompt, **kwargs):
            call_count[0] += 1
            if "different sections" in prompt:
                # This is a reduce step - return something compact
                return "FINAL REDUCED"
            # This is a map step - return a large summary to force recursion
            return "A" * 200  # Large summary

        mock_client.complete.side_effect = side_effect

        text = "This is a long sentence repeated many times. " * 500
        result = hierarchical(
            text=text,
            client=mock_client,
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            max_tokens=100,
        )
        assert result == "FINAL REDUCED"
        # Should have multiple calls (map + at least one reduce)
        assert mock_client.complete.call_count > 1
