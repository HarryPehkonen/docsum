"""Tests for the CLI entry point."""

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


class TestCLI:
    """CLI argument parsing and execution."""

    def test_help_exits_clean(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_input_file(self):
        with pytest.raises(SystemExit):
            main(["--input", "/nonexistent/file.txt"])

    def test_basic_invocation(self, temp_text_file):
        """Basic invocation with all required args."""
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Summary result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
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
        """Refine mode is accepted."""
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Refined result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "--input", temp_text_file,
                    "--output", out_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--mode", "refine",
                ])

                with open(out_path) as f:
                    result = f.read()
                assert "Refined" in result
            finally:
                os.unlink(out_path)

    def test_mode_hierarchical(self, temp_text_file):
        """Hierarchical mode is accepted."""
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Hierarchical result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "--input", temp_text_file,
                    "--output", out_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--mode", "hierarchical",
                ])

                with open(out_path) as f:
                    result = f.read()
                assert "Hierarchical" in result
            finally:
                os.unlink(out_path)

    def test_builtin_prompt_selection(self, temp_text_file):
        """--prompt themes uses the themes built-in."""
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Theme analysis"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "--input", temp_text_file,
                    "--output", out_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--prompt", "themes",
                ])

                # Check the first call used the themes prompt
                first_call_args = mock_client.complete.call_args_list[0]
                prompt_text = first_call_args.args[0]
                assert "theme" in prompt_text.lower()
            finally:
                os.unlink(out_path)

    def test_custom_prompt(self, temp_text_file):
        """--prompt-file loads a custom prompt template."""
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
                    "--input", temp_text_file,
                    "--output", out_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--prompt-file", prompt_path,
                ])

                first_call = mock_client.complete.call_args_list[0]
                assert "Extract all dates" in first_call.args[0]
            finally:
                os.unlink(out_path)
                os.unlink(prompt_path)

    def test_invalid_mode(self, temp_text_file):
        """Invalid mode raises an error."""
        with pytest.raises(SystemExit):
            main([
                "--input", temp_text_file,
                "--model", "test-model",
                "--base-url", "http://localhost:8645/v1",
                "--mode", "bogus",
            ])

    def test_max_output_tokens_flag(self, temp_text_file):
        """--max-output-tokens is passed through to the LLM."""
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "--input", temp_text_file,
                    "--output", out_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--max-output-tokens", "16384",
                ])

                call_kwargs = mock_client.complete.call_args
                assert call_kwargs.kwargs.get("max_tokens") == 16384
            finally:
                os.unlink(out_path)

    def test_quiet_flag_suppresses_progress(self, temp_text_file):
        """--quiet runs without progress bar."""
        with patch("docsum.cli.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.complete.return_value = "Result"

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
                out_path = out.name

            try:
                main([
                    "--input", temp_text_file,
                    "--output", out_path,
                    "--model", "test-model",
                    "--base-url", "http://localhost:8645/v1",
                    "--quiet",
                ])

                # Should still produce output
                with open(out_path) as f:
                    assert f.read() == "Result"
            finally:
                os.unlink(out_path)
