"""Tests for streaming mode and no-max-output-tokens."""

from unittest.mock import patch, MagicMock, PropertyMock
import pytest

from docsum.llm_client import LLMClient


class TestStreamingMode:
    """Streaming mode: tokens flow incrementally instead of one big response."""

    @patch("docsum.llm_client.OpenAI")
    def test_stream_false_by_default(self, mock_openai_cls):
        """Non-streaming by default — single response, no iteration."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Result"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:8645/v1", model="test")
        result = client.complete("Hello")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("stream") is not True or "stream" not in call_kwargs

    @patch("docsum.llm_client.OpenAI")
    def test_stream_true_when_enabled(self, mock_openai_cls):
        """When stream=True, the API call includes stream=True."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock streaming response: iterator of chunks with content deltas
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock(delta=MagicMock(content=None))]
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        client = LLMClient(base_url="http://localhost:8645/v1", model="test")
        result = client.complete("Hello", stream=True)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("stream") is True
        assert result == "Hello world"

    @patch("docsum.llm_client.OpenAI")
    def test_stream_collects_all_chunks(self, mock_openai_cls):
        """Streaming collects all delta chunks into the final text."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunks = []
        for text in ["The ", "quick ", "brown ", "fox"]:
            ch = MagicMock()
            ch.choices = [MagicMock(delta=MagicMock(content=text))]
            chunks.append(ch)
        # Final chunk with None content (finish)
        final = MagicMock()
        final.choices = [MagicMock(delta=MagicMock(content=None), finish_reason="stop")]
        chunks.append(final)

        mock_client.chat.completions.create.return_value = iter(chunks)

        client = LLMClient(base_url="http://localhost:8645/v1", model="test")
        result = client.complete("Test", stream=True)

        assert result == "The quick brown fox"

    @patch("docsum.llm_client.OpenAI")
    def test_stream_empty_response(self, mock_openai_cls):
        """Streaming with no content returns empty string."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        final = MagicMock()
        final.choices = [MagicMock(delta=MagicMock(content=None), finish_reason="stop")]
        mock_client.chat.completions.create.return_value = iter([final])

        client = LLMClient(base_url="http://localhost:8645/v1", model="test")
        result = client.complete("Test", stream=True)
        assert result == ""


class TestNoMaxOutputTokens:
    """--no-max-output-tokens: omit max_tokens from the API call entirely."""

    @patch("docsum.llm_client.OpenAI")
    def test_max_tokens_sent_by_default(self, mock_openai_cls):
        """By default, max_tokens is included in the API call."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:8645/v1", model="test")
        client.complete("Hello")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 8192

    @patch("docsum.llm_client.OpenAI")
    def test_max_tokens_none_omits_from_call(self, mock_openai_cls):
        """When max_tokens=None, the parameter is not sent to the API."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:8645/v1", model="test")
        client.complete("Hello", max_tokens=None)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs or call_kwargs["max_tokens"] is None

    @patch("docsum.llm_client.OpenAI")
    def test_stream_and_no_max_tokens_combined(self, mock_openai_cls):
        """Streaming + no max_tokens can be used together."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="streamed result"))]
        final = MagicMock()
        final.choices = [MagicMock(delta=MagicMock(content=None), finish_reason="stop")]
        mock_client.chat.completions.create.return_value = iter([chunk, final])

        client = LLMClient(base_url="http://localhost:8645/v1", model="test")
        result = client.complete("Hello", stream=True, max_tokens=None)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("stream") is True
        assert "max_tokens" not in call_kwargs or call_kwargs["max_tokens"] is None
        assert result == "streamed result"
