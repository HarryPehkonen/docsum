"""Tests for the fixes: custom reduce prompt, output cleanup, retry backoff."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from docsum.cli import main
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS, get_builtin_prompt
from docsum.step_processor import prepare, step, finalize
from docsum.step_state import StepState, load_state, save_state


@pytest.fixture
def tmp_state_path():
    path = tempfile.mktemp(suffix=".json")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete.side_effect = lambda prompt, **kwargs: f"RESULT: {prompt[:50]}"
    return client


@pytest.fixture
def simple_prompt():
    return BUILTIN_PROMPTS["summary"]


@pytest.fixture
def reduce_prompt():
    return BUILTIN_PROMPTS["reduce"]


class TestCustomReducePrompt:
    """Issue 1: --reduce-prompt-file for JSON-aware reduce."""

    def test_reduce_prompt_file_in_prepare(self, tmp_state_path, mock_client, simple_prompt):
        """prepare saves the custom reduce prompt to state."""
        custom_reduce = "Merge these JSON results into one. Return ONLY JSON.\n\n{summaries}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as rf:
            rf.write(custom_reduce)
            reduce_path = rf.name

        try:
            text = "Short text."
            state = prepare(
                text=text,
                state_path=tmp_state_path,
                client=mock_client,
                mode="map-reduce",
                prompt_template=simple_prompt,
                reduce_template=custom_reduce,
                model="test-model",
                base_url="http://localhost:8645/v1",
                max_tokens=1000,
            )
            assert state.reduce_template == custom_reduce

            # Verify it was saved to state
            loaded = load_state(tmp_state_path)
            assert loaded.reduce_template == custom_reduce
        finally:
            os.unlink(reduce_path)

    def test_finalize_uses_custom_reduce_prompt(self, tmp_state_path, mock_client, simple_prompt):
        """finalize should use the custom reduce prompt from state, not the default."""
        custom_reduce = "Merge JSON results. Return ONLY valid JSON.\n\n{summaries}"

        # Set up state with 2 chunks
        text = "Sentence one. " * 200
        state = prepare(
            text=text,
            state_path=tmp_state_path,
            client=mock_client,
            mode="map-reduce",
            prompt_template=simple_prompt,
            reduce_template=custom_reduce,
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=50,
        )

        # Process all chunks
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break

        # Track what reduce prompt was used
        reduce_call_prompt = None
        original_complete = mock_client.complete

        def track_call(prompt, **kwargs):
            nonlocal reduce_call_prompt
            reduce_call_prompt = prompt
            if "Merge JSON" in prompt:
                return '{"merged": true}'
            return "chunk result"

        mock_client.complete.side_effect = track_call

        result = finalize(state_path=tmp_state_path, client=mock_client)
        assert "result" in result
        assert "Merge JSON" in reduce_call_prompt
        assert "Combine" not in reduce_call_prompt  # default prompt not used

    def test_json_merge_reduce_prompt_builtin(self):
        """There should be a built-in 'json_reduce' prompt for merging JSON results."""
        assert "json_reduce" in BUILTIN_PROMPTS
        prompt = get_builtin_prompt("json_reduce")
        assert "{summaries}" in prompt
        assert "json" in prompt.lower()

    def test_cli_prepare_with_reduce_prompt_file(self, tmp_state_path):
        """CLI prepare accepts --reduce-prompt-file."""
        custom_reduce = "Merge JSON: {summaries}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as rf:
            rf.write(custom_reduce)
            reduce_path = rf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as inf:
            inf.write("Test text. " * 100)
            input_path = inf.name

        try:
            with patch("docsum.cli.LLMClient") as mock_cls:
                mock_cls.return_value = MagicMock()

                main([
                    "prepare",
                    "--input", input_path,
                    "--state", tmp_state_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--max-tokens", "100",
                    "--reduce-prompt-file", reduce_path,
                ])

                loaded = load_state(tmp_state_path)
                assert loaded.reduce_template == custom_reduce
        finally:
            os.unlink(reduce_path)
            os.unlink(input_path)
            if os.path.exists(tmp_state_path):
                os.unlink(tmp_state_path)


class TestOutputCleanup:
    """Issue 2 & 3: strip leading whitespace and trailing braces from output."""

    def test_finalize_strips_leading_whitespace(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Finalize output should not have leading whitespace."""
        text = "Sentence one. " * 200
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=simple_prompt,
            reduce_template=reduce_prompt, model="test",
            base_url="http://localhost:8645/v1", max_tokens=50,
        )
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break

        mock_client.complete.side_effect = lambda prompt, **kwargs: "\n\n  {\"result\": \"data\"}  "
        result = finalize(state_path=tmp_state_path, client=mock_client)
        assert result["result"] == '{"result": "data"}'
        assert not result["result"].startswith("\n")
        assert not result["result"].startswith(" ")

    def test_finalize_strips_markdown_fences(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Strip ```json ... ``` fences from output."""
        text = "Sentence one. " * 200
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=simple_prompt,
            reduce_template=reduce_prompt, model="test",
            base_url="http://localhost:8645/v1", max_tokens=50,
        )
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break

        mock_client.complete.side_effect = lambda prompt, **kwargs: '```json\n{"result": "data"}\n```'
        result = finalize(state_path=tmp_state_path, client=mock_client)
        # Should strip the markdown fences
        stripped = result["result"].strip()
        assert not stripped.startswith("```")

    def test_finalize_strips_trailing_extra_braces(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Strip extra trailing braces that the LLM sometimes appends."""
        text = "Sentence one. " * 200
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=simple_prompt,
            reduce_template=reduce_prompt, model="test",
            base_url="http://localhost:8645/v1", max_tokens=50,
        )
        while True:
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break

        # LLM returns valid JSON plus an extra brace
        mock_client.complete.side_effect = lambda prompt, **kwargs: '{"result": "data"}}'
        result = finalize(state_path=tmp_state_path, client=mock_client)
        # The extra trailing brace should be stripped
        stripped = result["result"].strip()
        # Should be valid JSON (no extra brace)
        # Count opening and closing braces
        assert stripped.count("{") == stripped.count("}")


class TestRetryBackoff:
    """Issue 4: --retry-backoff flag for step command."""

    def test_step_with_retry_succeeds_on_second_attempt(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """step with retry: first call fails, second succeeds."""
        text = "Sentence one. " * 200
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=simple_prompt,
            reduce_template=reduce_prompt, model="test",
            base_url="http://localhost:8645/v1", max_tokens=50,
        )

        call_count = [0]

        def flaky_complete(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("524 timeout")
            return "result after retry"

        mock_client.complete.side_effect = flaky_complete

        # With retry, should succeed on second attempt
        result = step(state_path=tmp_state_path, client=mock_client, retry_backoff=0, max_retries=2)
        assert result["chunk_index"] == 0
        assert result["result"] == "result after retry"
        assert call_count[0] == 2  # first failed, second succeeded

    def test_step_without_retry_fails(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """Without retry, step should propagate the exception."""
        text = "Sentence one. " * 200
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=simple_prompt,
            reduce_template=reduce_prompt, model="test",
            base_url="http://localhost:8645/v1", max_tokens=50,
        )

        mock_client.complete.side_effect = Exception("524 timeout")

        with pytest.raises(Exception, match="524"):
            step(state_path=tmp_state_path, client=mock_client)

    def test_step_retry_max_attempts(self, tmp_state_path, mock_client, simple_prompt, reduce_prompt):
        """After max retries, step should raise the exception."""
        text = "Sentence one. " * 200
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=simple_prompt,
            reduce_template=reduce_prompt, model="test",
            base_url="http://localhost:8645/v1", max_tokens=50,
        )

        mock_client.complete.side_effect = Exception("524 timeout")

        with pytest.raises(Exception, match="524"):
            step(state_path=tmp_state_path, client=mock_client, retry_backoff=0, max_retries=2)

        # Should have tried 3 times (1 initial + 2 retries)
        assert mock_client.complete.call_count == 3
