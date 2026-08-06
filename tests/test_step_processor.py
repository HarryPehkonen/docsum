"""Tests for the step-by-step processing: prepare, step, finalize."""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from docsum.step_processor import prepare, step, finalize, get_status
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS


@pytest.fixture
def tmp_state_path():
    path = tempfile.mktemp(suffix=".json")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete.side_effect = lambda prompt, **kwargs: f"RESULT: {prompt[:30]}"
    return client


@pytest.fixture
def simple_prompt():
    return BUILTIN_PROMPTS["summary"]


@pytest.fixture
def reduce_prompt():
    return BUILTIN_PROMPTS["reduce"]


class TestPrepare:
    """Prepare: chunk the input and initialize state."""

    def test_prepare_creates_state_file(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        text = "This is a test sentence. " * 100
        state = prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        assert os.path.exists(tmp_state_path)
        assert state.total_chunks > 1
        assert state.chunks_done == 0

    def test_prepare_single_chunk(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        text = "Short text."
        state = prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
        )
        assert state.total_chunks == 1
        assert state.is_complete() is False  # not until step processes it

    def test_prepare_empty_text(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        state = prepare(
            text="",
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=100,
        )
        assert state.total_chunks == 0
        assert state.is_complete() is True

    def test_prepare_preserves_settings(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        text = "Some text."
        state = prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="refine",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="z-ai/glm-5.2",
            base_url="http://127.0.0.1:8645/v1",
            max_tokens=500,
            overlap_tokens=100,
            max_output_tokens=16384,
        )
        assert state.mode == "refine"
        assert state.model == "z-ai/glm-5.2"
        assert state.max_tokens == 500
        assert state.overlap_tokens == 100
        assert state.max_output_tokens == 16384


class TestStep:
    """Step: process one chunk and save the result."""

    def test_step_processes_first_chunk(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        text = "This is a long sentence repeated many times. " * 100
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        result = step(state_path=tmp_state_path, client=mock_client)
        assert result["chunk_index"] == 0
        assert "result" in result
        assert result["chunks_done"] == 1
        assert result["is_complete"] is False

    def test_step_processes_all_chunks_sequentially(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        text = "This is a long sentence repeated many times. " * 200
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=30,
        )
        total = None
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if total is None:
                total = result["total_chunks"]
            if result["is_complete"]:
                break
        assert result["chunks_done"] == total
        assert result["is_complete"] is True

    def test_step_when_already_complete(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Calling step when all chunks are done returns 'already complete'."""
        text = "Short text."
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
        )
        # Process the single chunk
        step(state_path=tmp_state_path, client=mock_client)
        # Try to step again
        result = step(state_path=tmp_state_path, client=mock_client)
        assert result["is_complete"] is True
        assert "already" in result.get("message", "").lower() or result["chunks_done"] == result["total_chunks"]

    def test_step_refine_mode(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Refine mode: each step refines the running summary."""
        text = "This is a long sentence repeated many times. " * 100
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="refine",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        # First chunk: simple summary
        result = step(state_path=tmp_state_path, client=mock_client)
        assert result["chunk_index"] == 0
        assert "result" in result

        # Second chunk: should include running summary from first
        result2 = step(state_path=tmp_state_path, client=mock_client)
        assert result2["chunk_index"] == 1
        assert "result" in result2

    def test_step_resume_after_partial(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Resuming: already-processed chunks are skipped."""
        text = "This is a long sentence repeated many times. " * 200
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=30,
        )
        # Process first 2 chunks
        step(state_path=tmp_state_path, client=mock_client)
        step(state_path=tmp_state_path, client=mock_client)
        # Next step should be chunk 2, not chunk 0
        result = step(state_path=tmp_state_path, client=mock_client)
        assert result["chunk_index"] == 2
        assert result["chunks_done"] == 3


class TestFinalize:
    """Finalize: combine all chunk results into a final output."""

    def test_finalize_map_reduce(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Finalize combines all chunk summaries with the reduce prompt."""
        text = "This is a long sentence repeated many times. " * 100
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        # Process all chunks
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break

        # Finalize
        final = finalize(state_path=tmp_state_path, client=mock_client)
        assert "result" in final
        assert final["is_complete"] is True

    def test_finalize_single_chunk(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Single chunk: finalize returns the chunk result directly (no reduce needed)."""
        text = "Short text."
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
        )
        step(state_path=tmp_state_path, client=mock_client)
        final = finalize(state_path=tmp_state_path, client=mock_client)
        assert "result" in final
        # Single chunk: no reduce call needed
        mock_client.complete.assert_called_once()

    def test_finalize_refine_mode(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Refine mode: finalize returns the running summary directly."""
        text = "This is a long sentence repeated many times. " * 100
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="refine",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        # Process all chunks
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break

        # Finalize: for refine, the running summary IS the final result
        final = finalize(state_path=tmp_state_path, client=mock_client)
        assert "result" in final

    def test_finalize_not_all_chunks_done(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Finalize before all chunks are done returns an error."""
        text = "This is a long sentence repeated many times. " * 200
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=30,
        )
        # Process only one chunk
        step(state_path=tmp_state_path, client=mock_client)
        # Try to finalize
        final = finalize(state_path=tmp_state_path, client=mock_client)
        assert "error" in final
        assert "not all" in final["error"].lower() or "incomplete" in final["error"].lower()

    def test_finalize_hierarchical_mode(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Hierarchical mode: finalize recursively reduces if needed."""
        # Make summaries large to force recursion
        mock_client.complete.side_effect = lambda prompt, **kwargs: "A" * 200 if "different sections" not in prompt else "FINAL REDUCED"
        text = "This is a long sentence repeated many times. " * 500
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="hierarchical",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break
        final = finalize(state_path=tmp_state_path, client=mock_client)
        assert "result" in final


class TestGetStatus:
    """Status: report progress without processing."""

    def test_status_fresh(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        text = "This is a long sentence. " * 100
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        status = get_status(tmp_state_path)
        assert status["total_chunks"] > 0
        assert status["chunks_done"] == 0
        assert status["is_complete"] is False

    def test_status_after_steps(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        text = "This is a long sentence. " * 100
        prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=reduce_prompt,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )
        step(state_path=tmp_state_path, client=mock_client)
        status = get_status(tmp_state_path)
        assert status["chunks_done"] == 1

    def test_status_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            get_status("/nonexistent/state.json")
