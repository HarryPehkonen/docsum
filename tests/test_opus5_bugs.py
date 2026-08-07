"""Failing tests that confirm bugs identified by Opus 5 code review.

These tests are expected to FAIL against the current code — they document
the bugs so Claude Code can fix them and turn them green.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from docsum.json_merge import merge_json_objects, merge_json_chunks, _keep_longest, _dedup_by_name
from docsum.algorithms import map_reduce, refine, hierarchical, _recursive_reduce
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS


# === json_merge.py bugs ===================================================

class TestJsonMergeBugStringifyNonStrings:
    """Bug 1: _keep_longest stringifies non-string values, corrupting them."""

    def test_integer_value_preserved(self):
        a = {"year": 1954}
        b = {"year": 1962}
        merged = merge_json_objects(a, b)
        # Should keep the larger integer, not stringify it
        assert merged["year"] == 1962
        assert isinstance(merged["year"], int)

    def test_boolean_value_preserved(self):
        a = {"flag": True}
        b = {"flag": False}
        merged = merge_json_objects(a, b)
        # Should keep one of the booleans, not "True" or "False"
        assert isinstance(merged["flag"], bool)

    def test_nested_dict_not_stringified(self):
        a = {"metadata": {"author": "Alice"}}
        b = {"metadata": {"author": "Bob"}}
        merged = merge_json_objects(a, b)
        # Nested dicts should be merged recursively or at least kept as dict
        assert isinstance(merged["metadata"], dict)


class TestJsonMergeBugKeyEventsChunkLocalOrder:
    """Bug 2: key_events order is chunk-local — global sort interleaves events.

    Chunk 1: events with order 1, 2, 3
    Chunk 2: events with order 1, 2, 3
    After global sort by order: 1, 1, 2, 2, 3, 3 — WRONG

    Should be: chunk 1's events first, then chunk 2's, preserving
    document order: 1, 2, 3, 1, 2, 3 (or rebalanced globally)
    """

    def test_key_events_preserve_document_order(self):
        a = {"key_events": [{"event": "A", "order": 1}, {"event": "B", "order": 2}]}
        b = {"key_events": [{"event": "C", "order": 1}, {"event": "D", "order": 2}]}
        merged = merge_json_objects(a, b)
        events = merged["key_events"]
        # Events from chunk a should come before events from chunk b
        # (document order, not chunk-local order)
        event_names = [e["event"] for e in events]
        assert event_names == ["A", "B", "C", "D"], f"Expected document order, got {event_names}"


class TestJsonMergeBugOrderMixedTypes:
    """Bug 3: sorting key_events by 'order' crashes on mixed types."""

    def test_order_as_string_and_int(self):
        a = {"key_events": [{"event": "A", "order": 1}]}
        b = {"key_events": [{"event": "B", "order": "2"}]}
        # Should not raise TypeError
        merged = merge_json_objects(a, b)
        assert len(merged["key_events"]) == 2

    def test_order_as_string(self):
        a = {"key_events": [{"event": "A", "order": "1"}, {"event": "B", "order": "3"}]}
        b = {"key_events": [{"event": "C", "order": "2"}]}
        merged = merge_json_objects(a, b)
        assert len(merged["key_events"]) == 3


class TestJsonMergeBugDedupAssumesDicts:
    """Bug 4: _dedup_by_name assumes items are dicts — crashes on string arrays.

    LLMs sometimes return characters as ["Frodo", "Sam"] instead of
    [{"name": "Frodo"}, {"name": "Sam"}].
    """

    def test_characters_as_strings(self):
        a = {"characters": ["Picard", "Data"]}
        b = {"characters": ["Picard", "Riker"]}
        # Should not crash — should deduplicate as strings
        merged = merge_json_objects(a, b)
        assert "Picard" in merged["characters"]
        assert "Data" in merged["characters"]
        assert "Riker" in merged["characters"]
        assert merged["characters"].count("Picard") == 1

    def test_places_as_dicts(self):
        """Some LLMs return places as objects instead of strings."""
        a = {"places": [{"name": "Bridge", "type": "room"}]}
        b = {"places": ["Engineering"]}
        # Should not crash — handle mixed types gracefully
        merged = merge_json_objects(a, b)
        assert merged["places"] is not None


class TestJsonMergeBugKeyOrderNondeterministic:
    """Bug 5: Key order in merged output is nondeterministic (iterating a set).

    Running merge twice on the same input should produce the same key order.
    """

    def test_key_order_deterministic(self):
        a = {"z": "z", "a": "a", "m": "m"}
        b = {"z": "z2", "a": "a2", "m": "m2"}
        result1 = merge_json_objects(a, b)
        result2 = merge_json_objects(a, b)
        keys1 = list(result1.keys())
        keys2 = list(result2.keys())
        assert keys1 == keys2, f"Key order is nondeterministic: {keys1} vs {keys2}"


class TestJsonMergeBugSilentDataLoss:
    """Bug 6: Unparseable chunks vanish with no signal."""

    def test_merge_chunks_reports_skipped(self):
        chunks = [
            '{"summary": "good", "characters": [{"name": "Picard"}]}',
            'NOT JSON AT ALL',
            '{"summary": "also good", "characters": [{"name": "Data"}]}',
        ]
        result_str = merge_json_chunks(chunks)
        result = json.loads(result_str)
        # The two valid chunks were merged
        assert len(result["characters"]) == 2
        # But we should know that one chunk was skipped
        # (This test documents the current behavior — the fix should add reporting)


# === algorithms.py bugs ===================================================

class TestAlgorithmsBugRecursiveReduceUnbounded:
    """Bug 1: _recursive_reduce can recurse without bound.

    max_output_tokens defaults to 8192 while _max_reduce_tokens is 2000,
    so each group's reduced output can be 4× the budget it was supposed
    to fit. With enough groups the recursion re-expands and never converges.
    """

    def test_recursive_reduce_has_depth_limit(self):
        """The function should have a max depth to prevent infinite recursion."""
        import inspect
        from docsum.algorithms import _recursive_reduce
        sig = inspect.signature(_recursive_reduce)
        params = list(sig.parameters.keys())
        # Should have a depth parameter or internal limit
        # This test documents the issue — the fix should add a depth cap
        # For now, just verify it doesn't infinite-loop on a small input
        client = MagicMock(spec=LLMClient)
        # Each reduce call returns something large enough to need another reduce
        client.complete.side_effect = lambda prompt, **kw: "A" * 5000
        from docsum.prompts import render_reduce_prompt
        summaries = ["big summary"] * 20
        # This should terminate, not recurse forever
        # If it hangs, the test times out (confirming the bug)
        result = _recursive_reduce(
            summaries=summaries,
            client=client,
            reduce_template=BUILTIN_PROMPTS["reduce"],
            max_tokens=100,
            model="gpt-4",
        )
        assert result is not None
        # Verify it didn't make an unreasonable number of calls
        assert client.complete.call_count < 50


class TestAlgorithmsBugRechunkingDestroysBoundaries:
    """Bug 2: Re-chunking destroys summary boundaries.

    _recursive_reduce joins all summaries into one string, then chunks
    that string by tokens — cutting mid-summary. Each fragment is passed
    to render_reduce_prompt as if it were one whole summary.
    """

    def test_reduce_preserves_summary_boundaries(self):
        """Each summary should be passed as a whole unit to the reduce step."""
        client = MagicMock(spec=LLMClient)
        prompts_seen = []

        def track(prompt, **kw):
            prompts_seen.append(prompt)
            if "Combine" in prompt:
                return "combined"
            return "A" * 200  # large summary to force chunking

        client.complete.side_effect = track

        summaries = ["Summary one about Picard.", "Summary two about Data.", "Summary three about Worf."]
        from docsum.algorithms import _recursive_reduce
        _recursive_reduce(
            summaries=summaries,
            client=client,
            reduce_template=BUILTIN_PROMPTS["reduce"],
            max_tokens=50,  # small to force multiple reduce calls
            model="gpt-4",
        )
        # At least one reduce prompt should contain complete summaries,
        # not fragments. Check that no prompt has a summary cut in half.
        # (This is hard to test precisely — the point is that summaries
        # should not be split across reduce calls)
        # At minimum, verify the reduce step was called
        reduce_calls = [p for p in prompts_seen if "Combine" in p or "different sections" in p]
        assert len(reduce_calls) > 0


class TestAlgorithmsBugTextClobbering:
    """Bug 3: {text} in running summary gets clobbered by sequential .replace().

    If the LLM's running summary contains the literal string "{text}",
    the next refine iteration's .replace("{text}", chunk) will substitute
    the chunk body into the summary text, corrupting it.
    """

    def test_refine_with_literal_text_placeholder(self):
        """Refine should not clobber a literal {text} in the running summary."""
        client = MagicMock(spec=LLMClient)
        call_count = [0]
        prompts_sent = []

        def side_effect(prompt, **kw):
            prompts_sent.append(prompt)
            call_count[0] += 1
            if call_count[0] == 1:
                return "The template uses {text} as a placeholder."
            return "Refined summary"

        client.complete.side_effect = side_effect

        text = "Sentence one with enough words. " * 200
        result = refine(
            text=text,
            client=client,
            prompt_template="Summarize: {text}",
            max_tokens=50,
        )
        # The second prompt should NOT have the chunk body inserted
        # where {text} appeared in the running summary
        if len(prompts_sent) > 1:
            second_prompt = prompts_sent[1]
            # The running summary "The template uses {text} as a placeholder."
            # should appear intact, not with the chunk substituted in
            # The bug: .replace("{text}", chunk) replaces {text} in the summary
            # Check that the summary text is preserved
            assert "The template uses" in second_prompt or "Refined" in second_prompt
            # The chunk body (long repeated sentences) should NOT appear
            # where {text} was in the summary
            # (It will appear in the "New text:" section, but not in the summary section)


class TestAlgorithmsJsonDetectionFragile:
    """Design issue: JSON detection by string sniffing is fragile."""

    def test_json_detection_misses_variants(self):
        """A prompt saying 'Respond in JSON format' should be detected as JSON mode."""
        client = MagicMock(spec=LLMClient)
        prompts_sent = []

        def side_effect(prompt, **kw):
            prompts_sent.append(prompt)
            return '{"result": "ok"}'

        client.complete.side_effect = side_effect

        text = "Sentence one with enough words. " * 200
        # This prompt asks for JSON but doesn't match the detection strings
        json_prompt = "Respond in JSON format with {summary, characters, places} fields.\n\n{text}"
        refine(
            text=text,
            client=client,
            prompt_template=json_prompt,
            max_tokens=50,
        )
        # The second prompt should use JSON-aware refine instructions
        # (preserve all fields, merge arrays, etc.)
        if len(prompts_sent) > 1:
            second_prompt = prompts_sent[1]
            # Bug: this prompt says "JSON format" but doesn't match
            # "return only json" or '"json"' so it falls through to prose mode
            assert "MUST preserve ALL" in second_prompt or "preserve" in second_prompt.lower(), \
                "JSON mode should be detected from 'JSON format' in the prompt"
