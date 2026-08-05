"""Tests for the LLM client that calls OpenAI-compatible endpoints."""

from unittest.mock import patch, MagicMock

import pytest

from docsum.llm_client import LLMClient


class TestLLMClient:
    """The LLM client wraps an OpenAI-compatible API call."""

    def test_init(self):
        client = LLMClient(
            base_url="http://localhost:8645/v1",
            model="z-ai/glm-5.2",
            api_key="test-key",
        )
        assert client.model == "z-ai/glm-5.2"
        assert client.base_url == "http://localhost:8645/v1"

    def test_default_api_key(self):
        """If no API key is given, a default placeholder is used."""
        client = LLMClient(base_url="http://localhost:8645/v1", model="test-model")
        assert client.api_key is not None

    @patch("docsum.llm_client.OpenAI")
    def test_complete_calls_api(self, mock_openai_cls):
        """complete() should call the OpenAI client with the right parameters."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary result"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:8645/v1", model="test-model")
        result = client.complete("Summarize: hello")

        assert result == "Summary result"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "test-model"
        assert call_args.kwargs["messages"][0]["content"] == "Summarize: hello"

    @patch("docsum.llm_client.OpenAI")
    def test_complete_with_system_prompt(self, mock_openai_cls):
        """If a system prompt is provided, it's sent as a system message."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:8645/v1", model="test-model")
        result = client.complete("Hello", system_prompt="You are a summarizer.")

        assert result == "OK"
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a summarizer."
        assert messages[1]["role"] == "user"

    @patch("docsum.llm_client.OpenAI")
    def test_complete_empty_response(self, mock_openai_cls):
        """Handle an empty response gracefully."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=""))]
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:8645/v1", model="test-model")
        result = client.complete("Hello")
        assert result == ""

    @patch("docsum.llm_client.OpenAI")
    def test_complete_with_temperature(self, mock_openai_cls):
        """Temperature parameter is passed through."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:8645/v1", model="test-model")
        client.complete("Hello", temperature=0.3)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.3
