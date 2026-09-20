"""Tests for streaming mode and no-max-output-tokens."""

import pytest
from unittest.mock import MagicMock, patch

from docsum.algorithms import hierarchical, map_reduce, refine
from docsum.cli import main
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS
from docsum.step_processor import finalize, prepare, step


def _stub_client():
    """An LLMClient stand-in that records every call and answers "ok"."""
    client = MagicMock(spec=LLMClient)
    client.complete.side_effect = lambda prompt, **kwargs: "ok"
    return client


def _long_text():
    """Long enough to force several chunks at max_tokens=100."""
    return "Sentence one about Picard and Data. " * 200


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


class TestStepPathHonoursTheRecordedFlags:
    """`prepare` -> `step` -> `finalize` must honour BOTH recorded flags.

    Regression pin (2026-09-20, card t_01e401ae): `step()` passed `stream=state.stream`
    and handled `no_max_output_tokens`, but `finalize()` built its LLM calls with
    neither. So `prepare --stream` streamed every chunk call and then sent the longest
    request in the whole flow — the reduce, which carries every chunk summary at once —
    unstreamed, and `prepare --no-max-output-tokens` re-imposed max_tokens on that same
    final reduce. `finalize` is the most 524-prone call there is, which is exactly what
    the flag exists to protect.
    """

    def _run_step_path(self, tmp_path, mode, **prepare_kwargs):
        """Drive prepare -> step* -> finalize; return (client, calls before finalize)."""
        client = _stub_client()
        state = tmp_path / "state.json"
        prepare(
            text=_long_text(),
            state_path=str(state),
            client=client,
            mode=mode,
            prompt_template=BUILTIN_PROMPTS["summary"],
            reduce_template=BUILTIN_PROMPTS["reduce"],
            model="test-model",
            base_url="http://localhost:8645/v1",
            max_tokens=100,
            **prepare_kwargs,
        )
        for _ in range(50):
            if step(state_path=str(state), client=client)["is_complete"]:
                break
        else:
            pytest.fail("step never completed")
        calls_before_finalize = len(client.complete.call_args_list)
        finalize(state_path=str(state), client=client)
        return client, calls_before_finalize

    @pytest.mark.parametrize("mode", ["map-reduce", "hierarchical"])
    def test_finalize_streams_when_the_state_recorded_it(self, tmp_path, mode):
        client, calls_before_finalize = self._run_step_path(tmp_path, mode, stream=True)
        calls = client.complete.call_args_list
        assert len(calls) > calls_before_finalize, "finalize made no LLM call"
        assert all(c.kwargs.get("stream") is True for c in calls)

    def test_refine_step_path_streams_every_call(self, tmp_path):
        """Refine has no finalize LLM call; its chunk calls still must stream."""
        client, _ = self._run_step_path(tmp_path, "refine", stream=True)
        calls = client.complete.call_args_list
        assert calls
        assert all(c.kwargs.get("stream") is True for c in calls)

    def test_step_path_does_not_stream_without_the_flag(self, tmp_path):
        client, _ = self._run_step_path(tmp_path, "map-reduce")
        calls = client.complete.call_args_list
        assert len(calls) > 1
        assert all(c.kwargs.get("stream") is False for c in calls)

    @pytest.mark.parametrize("mode", ["map-reduce", "hierarchical"])
    def test_finalize_omits_max_tokens_when_the_state_recorded_it(self, tmp_path, mode):
        client, calls_before_finalize = self._run_step_path(
            tmp_path, mode, no_max_output_tokens=True
        )
        calls = client.complete.call_args_list
        assert len(calls) > calls_before_finalize, "finalize made no LLM call"
        assert all(c.kwargs.get("max_tokens") is None for c in calls)

    def test_step_path_sends_the_recorded_max_output_tokens(self, tmp_path):
        client, _ = self._run_step_path(tmp_path, "map-reduce", max_output_tokens=1234)
        calls = client.complete.call_args_list
        assert len(calls) > 1
        assert all(c.kwargs.get("max_tokens") == 1234 for c in calls)


class TestRunPathNoMaxOutputTokens:
    """`run --no-max-output-tokens` must actually reach the API call — in every mode.

    Regression pin (2026-09-20, card t_01e401ae): `_cmd_run` computed
    `run_max_tokens = None` for the flag and then passed
    `max_output_tokens=run_max_tokens if run_max_tokens is not None else 8192` at all
    three call sites — the `else 8192` flattened the None straight back, so the flag
    was accepted and ignored (and it silently discarded an explicit
    `--max-output-tokens` value too). `LLMClient.complete` omits the field only when it
    receives None, so the algorithms could never ask for omission. Every test here runs
    all three modes, because each mode had its own flattened call site.
    """

    _MODES = ["map-reduce", "refine", "hierarchical"]

    def _run_cli(self, tmp_path, *extra):
        src = tmp_path / "input.txt"
        src.write_text(_long_text(), encoding="utf-8")
        with patch("docsum.cli.LLMClient") as mock_cls:
            client = _stub_client()
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

    def _max_tokens(self, client):
        return [c.kwargs.get("max_tokens") for c in client.complete.call_args_list]

    @pytest.mark.parametrize("mode", _MODES)
    def test_run_sends_the_default_max_output_tokens(self, tmp_path, mode):
        client = self._run_cli(tmp_path, "--mode", mode)
        sent = self._max_tokens(client)
        assert sent
        assert all(m == 8192 for m in sent)

    @pytest.mark.parametrize("mode", _MODES)
    def test_run_sends_an_explicit_max_output_tokens(self, tmp_path, mode):
        client = self._run_cli(tmp_path, "--mode", mode, "--max-output-tokens", "16384")
        sent = self._max_tokens(client)
        assert sent
        assert all(m == 16384 for m in sent)

    @pytest.mark.parametrize("mode", _MODES)
    def test_run_no_max_output_tokens_omits_the_field(self, tmp_path, mode):
        client = self._run_cli(tmp_path, "--mode", mode, "--no-max-output-tokens")
        sent = self._max_tokens(client)
        assert sent
        assert all(m is None for m in sent)

    @pytest.mark.parametrize("mode", _MODES)
    def test_run_no_max_output_tokens_wins_over_an_explicit_value(self, tmp_path, mode):
        """The flag says "omit the field", so it must not silently send 8192 instead."""
        client = self._run_cli(
            tmp_path,
            "--mode",
            mode,
            "--max-output-tokens",
            "100",
            "--no-max-output-tokens",
        )
        sent = self._max_tokens(client)
        assert sent
        assert all(m is None for m in sent)
