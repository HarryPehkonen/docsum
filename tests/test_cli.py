"""Tests for the CLI entry point."""

import json
import sys
import tempfile
import os
from unittest.mock import patch, MagicMock

import pytest

from docsum.cli import main


@pytest.fixture
def temp_text_file():
    """Create a temporary text file with sample content."""
    content = "This is a test sentence. " * 100
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


class TestCLIMonolithic:
    """Monolithic mode (run subcommand)."""

    def test_help_exits_clean(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_subcommand_shows_help(self):
        exit_code = main([])
        assert exit_code == 1

    def test_basic_invocation(self, temp_text_file):
        """Basic run with all required args."""
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Summary result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "run",
                    "--input", temp_text_file,
                    "--output", out_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--mode", "map-reduce",
                    "--max-tokens", "500",
                ])

                with open(out_path) as f:
                    result = f.read()
                assert result == "Summary result"
            finally:
                os.unlink(out_path)

    def test_mode_refine(self, temp_text_file):
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Refined result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "run", "--input", temp_text_file, "--output", out_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--mode", "refine",
                ])

                with open(out_path) as f:
                    assert "Refined" in f.read()
            finally:
                os.unlink(out_path)

    def test_mode_hierarchical(self, temp_text_file):
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Hierarchical result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "run", "--input", temp_text_file, "--output", out_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--mode", "hierarchical",
                ])

                with open(out_path) as f:
                    assert "Hierarchical" in f.read()
            finally:
                os.unlink(out_path)

    def test_builtin_prompt_selection(self, temp_text_file):
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Theme analysis"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "run", "--input", temp_text_file, "--output", out_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--prompt", "themes",
                ])

                first_call_args = mock_client.complete.call_args_list[0]
                prompt_text = first_call_args.args[0]
                assert "theme" in prompt_text.lower()
            finally:
                os.unlink(out_path)

    def test_custom_prompt(self, temp_text_file):
        custom_prompt = "Extract all dates from: {text}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
            pf.write(custom_prompt)
            prompt_path = pf.name

        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Date result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "run", "--input", temp_text_file, "--output", out_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--prompt-file", prompt_path,
                ])

                first_call = mock_client.complete.call_args_list[0]
                assert "Extract all dates" in first_call.args[0]
            finally:
                os.unlink(out_path)
                os.unlink(prompt_path)

    def test_max_output_tokens_flag(self, temp_text_file):
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "run", "--input", temp_text_file, "--output", out_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--max-output-tokens", "16384",
                ])

                call_kwargs = mock_client.complete.call_args
                assert call_kwargs.kwargs.get("max_tokens") == 16384
            finally:
                os.unlink(out_path)

    def test_quiet_flag(self, temp_text_file):
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "run", "--input", temp_text_file, "--output", out_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--quiet",
                ])

                with open(out_path) as f:
                    assert f.read() == "Result"
            finally:
                os.unlink(out_path)


class TestCLIStepByStep:
    """Step-by-step mode (prepare/step/finalize/status subcommands)."""

    def test_prepare_creates_state(self, temp_text_file):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sf:
            state_path = sf.name
        os.unlink(state_path)  # should not exist yet

        try:
            with patch("docsum.cli.LLMClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client

                exit_code = main([
                    "prepare",
                    "--input", temp_text_file,
                    "--state", state_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--max-tokens", "100",
                ])

                assert exit_code == 0
                assert os.path.exists(state_path)

                with open(state_path) as f:
                    state = json.load(f)
                assert state["model"] == "test-model"
                assert len(state["chunks"]) > 0
                assert state["mode"] == "map-reduce"
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_prepare_custom_prompt(self, temp_text_file):
        custom_prompt = "Extract dates: {text}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
            pf.write(custom_prompt)
            prompt_path = pf.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sf:
            state_path = sf.name
        os.unlink(state_path)

        try:
            with patch("docsum.cli.LLMClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client

                main([
                    "prepare", "--input", temp_text_file, "--state", state_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--prompt-file", prompt_path,
                ])

                with open(state_path) as f:
                    state = json.load(f)
                assert "Extract dates" in state["prompt_template"]
        finally:
            os.unlink(prompt_path)
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_step_processes_chunk(self, temp_text_file):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sf:
            state_path = sf.name
        os.unlink(state_path)

        try:
            with patch("docsum.cli.LLMClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.complete.return_value = "Chunk summary"

                main([
                    "prepare", "--input", temp_text_file, "--state", state_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--max-tokens", "100",
                ])

                exit_code = main(["step", "--state", state_path])
                assert exit_code == 0

                # Verify state was updated
                with open(state_path) as f:
                    state = json.load(f)
                assert len(state["results"]) >= 1
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_status_shows_progress(self, temp_text_file):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sf:
            state_path = sf.name
        os.unlink(state_path)

        try:
            with patch("docsum.cli.LLMClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.complete.return_value = "Result"

                main([
                    "prepare", "--input", temp_text_file, "--state", state_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--max-tokens", "100",
                ])

                # Step once
                main(["step", "--state", state_path])

                # Check status
                exit_code = main(["status", "--state", state_path])
                assert exit_code == 0
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_full_step_by_step_flow(self, temp_text_file):
        """Full prepare → step × N → finalize flow."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sf:
            state_path = sf.name
        os.unlink(state_path)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
            out_path = out.name

        try:
            with patch("docsum.cli.LLMClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.complete.return_value = "Summary"

                # Prepare
                main([
                    "prepare", "--input", temp_text_file, "--state", state_path,
                    "--model", "test-model", "--base-url", "http://localhost:8645/v1",
                    "--max-tokens", "50",
                ])

                # Step until complete
                while True:
                    exit_code = main(["step", "--state", state_path])
                    assert exit_code == 0
                    # Check if done by reading state
                    with open(state_path) as f:
                        state = json.load(f)
                    if len(state["results"]) == len(state["chunks"]):
                        break

                # Finalize
                exit_code = main(["finalize", "--state", state_path, "--output", out_path])
                assert exit_code == 0

                with open(out_path) as f:
                    result = f.read()
                assert len(result) > 0
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)
            os.unlink(out_path)

    def test_status_nonexistent_state(self):
        exit_code = main(["status", "--state", "/nonexistent/state.json"])
        assert exit_code == 1
