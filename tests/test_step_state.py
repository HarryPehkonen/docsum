"""Tests for the step-by-step state management.

The state file tracks:
- which chunks exist and which are done
- the chunk results so far
- the algorithm settings (mode, prompt, model, etc.)
so a long document can be processed one chunk at a time across
multiple invocations of the CLI.
"""

import json
import os
import tempfile

import pytest

from docsum.step_state import StepState, load_state, save_state


@pytest.fixture
def tmp_state_path():
    path = tempfile.mktemp(suffix=".json")
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestStepState:
    """Creating and managing step-by-step processing state."""

    def test_create_new_state(self, tmp_state_path):
        state = StepState(
            state_path=tmp_state_path,
            chunks=["chunk one text", "chunk two text", "chunk three text"],
            mode="map-reduce",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="z-ai/glm-5.2",
            base_url="http://127.0.0.1:8645/v1",
            max_tokens=2000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        assert state.total_chunks == 3
        assert state.chunks_done == 0
        assert state.is_complete() is False

    def test_save_and_load(self, tmp_state_path):
        state = StepState(
            state_path=tmp_state_path,
            chunks=["a", "b"],
            mode="refine",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=50,
            max_output_tokens=4096,
            tokenizer_model="gpt-4",
        )
        save_state(state)

        loaded = load_state(tmp_state_path)
        assert loaded.total_chunks == 2
        assert loaded.mode == "refine"
        assert loaded.model == "test-model"
        assert loaded.max_tokens == 1000
        assert loaded.overlap_tokens == 50

    def test_record_chunk_result(self, tmp_state_path):
        state = StepState(
            state_path=tmp_state_path,
            chunks=["chunk one", "chunk two", "chunk three"],
            mode="map-reduce",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        state.record_result(0, "summary of chunk one")
        assert state.chunks_done == 1
        assert state.results[0] == "summary of chunk one"
        assert state.is_complete() is False

        state.record_result(1, "summary of chunk two")
        assert state.chunks_done == 2

        state.record_result(2, "summary of chunk three")
        assert state.chunks_done == 3
        assert state.is_complete() is True

    def test_next_chunk_index(self, tmp_state_path):
        state = StepState(
            state_path=tmp_state_path,
            chunks=["a", "b", "c"],
            mode="map-reduce",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        assert state.next_chunk_index() == 0

        state.record_result(0, "result a")
        assert state.next_chunk_index() == 1

        state.record_result(1, "result b")
        assert state.next_chunk_index() == 2

        state.record_result(2, "result c")
        assert state.next_chunk_index() is None  # all done

    def test_get_results(self, tmp_state_path):
        state = StepState(
            state_path=tmp_state_path,
            chunks=["a", "b"],
            mode="map-reduce",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        state.record_result(0, "result a")
        state.record_result(1, "result b")
        results = state.get_results()
        assert results == ["result a", "result b"]

    def test_resume_after_partial_completion(self, tmp_state_path):
        """If we saved state after 2 of 3 chunks, loading should show 2 done."""
        state = StepState(
            state_path=tmp_state_path,
            chunks=["a", "b", "c"],
            mode="map-reduce",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        state.record_result(0, "result a")
        state.record_result(1, "result b")
        save_state(state)

        loaded = load_state(tmp_state_path)
        assert loaded.chunks_done == 2
        assert loaded.next_chunk_index() == 2
        assert loaded.is_complete() is False

    def test_load_nonexistent_state(self):
        with pytest.raises(FileNotFoundError):
            load_state("/nonexistent/path/state.json")

    def test_save_persist_chunk_texts(self, tmp_state_path):
        """Chunk texts must be saved so step can process them without re-reading the input file."""
        state = StepState(
            state_path=tmp_state_path,
            chunks=["first chunk text", "second chunk text"],
            mode="refine",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        save_state(state)
        loaded = load_state(tmp_state_path)
        assert loaded.chunks == ["first chunk text", "second chunk text"]

    def test_refine_running_summary(self, tmp_state_path):
        """Refine mode tracks a running summary that each step builds on."""
        state = StepState(
            state_path=tmp_state_path,
            chunks=["a", "b", "c"],
            mode="refine",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        # In refine, each result IS the running summary
        state.record_result(0, "running summary after chunk 1")
        assert state.running_summary == "running summary after chunk 1"

        state.record_result(1, "running summary after chunk 2")
        assert state.running_summary == "running summary after chunk 2"

    def test_status_output(self, tmp_state_path):
        """Status gives a summary of progress."""
        state = StepState(
            state_path=tmp_state_path,
            chunks=["a", "b", "c", "d"],
            mode="map-reduce",
            prompt_template="Summarize: {text}",
            reduce_template="Combine: {summaries}",
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=1000,
            overlap_tokens=0,
            max_output_tokens=8192,
            tokenizer_model="gpt-4",
        )
        state.record_result(0, "result a")
        state.record_result(1, "result b")
        status = state.status()
        assert status["total_chunks"] == 4
        assert status["chunks_done"] == 2
        assert status["chunks_remaining"] == 2
        assert status["is_complete"] is False
        assert status["mode"] == "map-reduce"
