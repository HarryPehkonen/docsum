"""Tests for streaming mode and no-max-output-tokens."""

from unittest.mock import MagicMock, patch

from docsum.algorithms import hierarchical, map_reduce, refine
from docsum.cli import main
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS


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
        client.complete("Hello")

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
        mock_client.chat.completions.create.return_value = iter(
            [chunk1, chunk2, chunk3]
        )

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


class TestStreamFlagReachesTheApiCall:
    """`--stream` must change the API call, from every entry point.

    Regression pin (2026-09-19): `docsum run` parsed the flag, copied it into a
    local at cli.py:264, and never passed it to the algorithms, so the flag was
    documented, accepted, and silently ignored. `prepare` did honour it, so the
    lie was invisible from the help text — this is the test that says the whole
    flag is real.
    """

    def _client(self):
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = lambda prompt, **kwargs: "ok"
        return client

    def _long_text(self):
        # Long enough to force several chunks at max_tokens=100
        return "Sentence one about Picard and Data. " * 200

    def _streams(self, client):
        return [call.kwargs.get("stream") for call in client.complete.call_args_list]

    def _call(self, algorithm, client, **extra):
        """Call an algorithm the way cli.py does — `refine` has no reduce template."""
        kwargs = {
            "text": self._long_text(),
            "client": client,
            "prompt_template": BUILTIN_PROMPTS["summary"],
            "max_tokens": 100,
        }
        if algorithm is not refine:
            kwargs["reduce_template"] = BUILTIN_PROMPTS["reduce"]
        return algorithm(**kwargs, **extra)

    def test_map_reduce_forwards_stream(self):
        client = self._client()
        self._call(map_reduce, client, stream=True)
        assert self._streams(client)
        assert all(s is True for s in self._streams(client))

    def test_refine_forwards_stream(self):
        client = self._client()
        self._call(refine, client, stream=True)
        assert self._streams(client)
        assert all(s is True for s in self._streams(client))

    def test_hierarchical_forwards_stream(self):
        client = self._client()
        self._call(hierarchical, client, stream=True)
        assert self._streams(client)
        assert all(s is True for s in self._streams(client))

    def test_algorithms_do_not_stream_by_default(self):
        """No `--stream` means stream=False — not "whatever the default is"."""
        for algorithm in (map_reduce, refine, hierarchical):
            client = self._client()
            self._call(algorithm, client)
            streams = self._streams(client)
            assert streams, algorithm.__name__
            assert all(s is False for s in streams), algorithm.__name__

    def _run_cli(self, tmp_path, *extra):
        src = tmp_path / "input.txt"
        src.write_text(self._long_text(), encoding="utf-8")
        with patch("docsum.cli.LLMClient") as mock_cls:
            client = self._client()
            mock_cls.return_value = client
            rc = main(
                [
                    "run",
                    "--input",
                    str(src),
                    "--model",
                    "test-model",
                    "--max-tokens",
                    "100",
                    "--quiet",
                    *extra,
                ]
            )
        assert rc == 0
        return client

    def test_cli_run_stream_flag_reaches_the_client(self, tmp_path):
        client = self._run_cli(tmp_path, "--stream")
        assert self._streams(client)
        assert all(s is True for s in self._streams(client))

    def test_cli_run_without_stream_flag_does_not_stream(self, tmp_path):
        client = self._run_cli(tmp_path)
        assert self._streams(client)
        assert all(s is False for s in self._streams(client))
