"""Tests for programmatic JSON merge — no LLM call for the reduce step.

When the reduce prompt is 'json_reduce', finalize merges JSON chunk
results programmatically instead of calling the LLM. This:
- Never times out (no API call)
- Preserves all fields (deterministic merge)
- Deduplicates by name/id
- Concatenates key_events in order
- Unions array fields
- Keeps the longest moral_dilemma / summary
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from docsum.json_merge import merge_json_objects, merge_json_chunks
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS
from docsum.step_processor import prepare, step, finalize
from docsum.step_state import load_state


@pytest.fixture
def tmp_state_path():
    path = tempfile.mktemp(suffix=".json")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    return client


@pytest.fixture
def json_prompt():
    """A prompt that asks for JSON output."""
    return (
        "Analyze the following text and return a JSON object with these fields:\n"
        "- \"summary\": concise prose summary\n"
        "- \"characters\": array of {\"name\", \"role\", \"description\"}\n"
        "- \"places\": array of location names\n"
        "- \"themes\": array of theme names\n"
        "- \"key_events\": array of {\"event\", \"order\"}\n"
        "- \"entities\": array of named things\n"
        "- \"moral_dilemma\": the central ethical question\n"
        "Return ONLY valid JSON.\n\n{text}"
    )


@pytest.fixture
def json_reduce_prompt():
    return BUILTIN_PROMPTS["json_reduce"]


class TestMergeJsonObjects:
    """Unit tests for the programmatic JSON merge logic."""

    def test_merge_simple_objects(self):
        a = {"summary": "Part one", "characters": [{"name": "Picard", "role": "Captain"}]}
        b = {"summary": "Part two", "characters": [{"name": "Data", "role": "Android"}]}
        merged = merge_json_objects(a, b)
        assert "Picard" in [c["name"] for c in merged["characters"]]
        assert "Data" in [c["name"] for c in merged["characters"]]

    def test_merge_deduplicates_characters_by_name(self):
        a = {"characters": [{"name": "Picard", "role": "Captain", "description": "version A"}]}
        b = {"characters": [{"name": "Picard", "role": "Captain", "description": "version B"}]}
        merged = merge_json_objects(a, b)
        picards = [c for c in merged["characters"] if c["name"] == "Picard"]
        assert len(picards) == 1  # deduplicated

    def test_merge_concatenates_key_events_in_order(self):
        a = {"key_events": [{"event": "first", "order": 1}, {"event": "third", "order": 3}]}
        b = {"key_events": [{"event": "second", "order": 2}]}
        merged = merge_json_objects(a, b)
        events = merged["key_events"]
        assert len(events) == 3
        # Should be sorted by order
        orders = [e["order"] for e in events]
        assert orders == [1, 2, 3]

    def test_merge_unions_string_arrays(self):
        a = {"places": ["Bridge", "Engineering"], "themes": ["duty"]}
        b = {"places": ["Bridge", "Sickbay"], "themes": ["loyalty", "duty"]}
        merged = merge_json_objects(a, b)
        assert "Bridge" in merged["places"]
        assert "Engineering" in merged["places"]
        assert "Sickbay" in merged["places"]
        # No duplicates
        assert merged["places"].count("Bridge") == 1
        assert "duty" in merged["themes"]
        assert "loyalty" in merged["themes"]
        assert merged["themes"].count("duty") == 1

    def test_merge_keeps_longest_summary(self):
        a = {"summary": "short"}
        b = {"summary": "a much longer summary that has more detail"}
        merged = merge_json_objects(a, b)
        assert merged["summary"] == b["summary"]

    def test_merge_keeps_longest_moral_dilemma(self):
        a = {"moral_dilemma": "short question"}
        b = {"moral_dilemma": "a much longer and more detailed moral dilemma description"}
        merged = merge_json_objects(a, b)
        assert len(merged["moral_dilemma"]) > len(a["moral_dilemma"])

    def test_merge_handles_missing_fields(self):
        """If one chunk is missing a field, the other's values are kept."""
        a = {"characters": [{"name": "Picard"}], "places": ["Bridge"]}
        b = {"characters": [{"name": "Data"}]}  # no places
        merged = merge_json_objects(a, b)
        assert "Bridge" in merged["places"]
        assert len(merged["characters"]) == 2

    def test_merge_handles_empty_objects(self):
        merged = merge_json_objects({}, {"characters": [{"name": "Picard"}]})
        assert len(merged["characters"]) == 1

    def test_merge_preserves_unknown_fields(self):
        """Fields not in the known schema are unioned as arrays or kept as-is."""
        a = {"custom_field": ["x"], "scalar": "a"}
        b = {"custom_field": ["y"], "scalar": "b"}
        merged = merge_json_objects(a, b)
        # Unknown arrays are unioned
        assert "x" in merged["custom_field"]
        assert "y" in merged["custom_field"]

    def test_merge_three_objects(self):
        a = {"places": ["A"], "characters": [{"name": "X"}]}
        b = {"places": ["B"], "characters": [{"name": "Y"}]}
        c = {"places": ["A", "C"], "characters": [{"name": "Z"}]}
        merged = merge_json_objects(a, b, c)
        assert merged["places"] == ["A", "B", "C"]
        assert len(merged["characters"]) == 3


class TestMergeJsonChunks:
    """Merging a list of JSON string results (as from chunk summaries)."""

    def test_merge_valid_json_strings(self):
        chunks = [
            '{"summary": "part 1", "characters": [{"name": "Picard"}], "places": ["Bridge"]}',
            '{"summary": "part 2", "characters": [{"name": "Data"}], "places": ["Engineering"]}',
        ]
        result = merge_json_chunks(chunks)
        parsed = json.loads(result)
        assert len(parsed["characters"]) == 2
        assert "Bridge" in parsed["places"]
        assert "Engineering" in parsed["places"]

    def test_merge_strips_markdown_fences_from_chunks(self):
        """Chunks may come wrapped in ```json fences."""
        chunks = [
            '```json\n{"summary": "part 1", "characters": []}\n```',
            '```json\n{"summary": "part 2", "characters": [{"name": "Data"}]}\n```',
        ]
        result = merge_json_chunks(chunks)
        parsed = json.loads(result)
        assert parsed["summary"] in ["part 1", "part 2"]
        assert len(parsed["characters"]) == 1

    def test_merge_handles_invalid_json_chunk(self):
        """If one chunk is not valid JSON, merge the rest and note the failure."""
        chunks = [
            '{"summary": "good", "characters": [{"name": "Picard"}]}',
            'NOT JSON AT ALL',
            '{"summary": "also good", "characters": [{"name": "Data"}]}',
        ]
        result = merge_json_chunks(chunks)
        parsed = json.loads(result)
        # The two valid chunks were merged; the bad one was skipped
        assert len(parsed["characters"]) == 2

    def test_merge_empty_list(self):
        result = merge_json_chunks([])
        parsed = json.loads(result)
        assert parsed == {}

    def test_merge_single_chunk(self):
        chunks = ['{"summary": "only one", "characters": [{"name": "Picard"}]}']
        result = merge_json_chunks(chunks)
        parsed = json.loads(result)
        assert parsed["summary"] == "only one"
        assert len(parsed["characters"]) == 1


class TestFinalizeWithProgrammaticMerge:
    """finalize uses programmatic merge when reduce_template is json_reduce."""

    def test_finalize_no_llm_call_for_json_reduce(self, tmp_state_path, mock_client, json_prompt, json_reduce_prompt):
        """When reduce prompt is json_reduce, finalize makes NO LLM call for reduce."""
        # Prepare with multiple chunks
        text = "Sentence one with enough words. " * 300
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=json_prompt,
            reduce_template=json_reduce_prompt,
            model="test-model", base_url="http://localhost:8645/v1",
            max_tokens=50,
        )

        # Step: return JSON for each chunk
        chunk_results = [
            json.dumps({"summary": f"part {i}", "characters": [{"name": f"Char{i}"}],
                       "places": [f"Place{i}"], "themes": [f"theme{i}"],
                       "key_events": [{"event": f"event {i}", "order": i}],
                       "entities": [f"entity{i}"], "moral_dilemma": f"dilemma {i}"})
            for i in range(10)
        ]
        call_idx = [0]
        def chunk_side_effect(prompt, **kw):
            idx = call_idx[0]
            call_idx[0] += 1
            return chunk_results[idx] if idx < len(chunk_results) else chunk_results[-1]

        mock_client.complete.side_effect = chunk_side_effect
        for _ in range(100):
            step(state_path=tmp_state_path, client=mock_client)
            state = load_state(tmp_state_path)
            if state.is_complete():
                break

        # Finalize — should NOT call the LLM
        mock_client.complete.reset_mock()
        result = finalize(state_path=tmp_state_path, client=mock_client)
        assert "result" in result
        assert mock_client.complete.call_count == 0  # no LLM call!

        # Result should be valid JSON
        parsed = json.loads(result["result"])
        assert len(parsed["characters"]) == 10
        assert len(parsed["places"]) == 10

    def test_finalize_falls_back_to_llm_for_non_json_reduce(self, tmp_state_path, mock_client, json_prompt):
        """When reduce prompt is the default 'reduce', finalize uses the LLM."""
        text = "Sentence one with enough words. " * 300
        prepare(
            text=text, state_path=tmp_state_path, client=mock_client,
            mode="map-reduce", prompt_template=json_prompt,
            reduce_template=BUILTIN_PROMPTS["reduce"],
            model="test-model", base_url="http://localhost:8645/v1",
            max_tokens=50,
        )

        # Step
        mock_client.complete.side_effect = lambda prompt, **kw: json.dumps({"summary": "chunk"})
        for _ in range(100):
            result = step(state_path=tmp_state_path, client=mock_client)
            if result["is_complete"]:
                break

        # Finalize — should call LLM
        mock_client.complete.reset_mock()
        mock_client.complete.return_value = "combined result"
        result = finalize(state_path=tmp_state_path, client=mock_client)
        assert mock_client.complete.call_count > 0
