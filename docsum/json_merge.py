"""Programmatic JSON merge for chunk results — no LLM call needed.

When chunk results are JSON objects (e.g., from a JSON-extraction prompt),
the reduce step can merge them programmatically instead of calling the LLM.
This is faster, free, deterministic, and never times out.

Merge rules:
- characters: deduplicate by name, merge descriptions (keep longest)
- places, themes, entities: union as string arrays, deduplicate
- key_events: concatenate all, sort by "order" field
- summary, moral_dilemma: keep the longest (most detailed)
- unknown array fields: union and deduplicate
- unknown scalar fields: keep the longest string value
"""

import json
import re


def _strip_fences(text: str) -> str:
    """Strip markdown ```json ... ``` fences from a string."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove ```json or ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json_safely(text: str) -> dict | None:
    """Try to parse a JSON string, stripping fences if needed. Returns None on failure."""
    text = _strip_fences(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _dedup_by_name(items: list[dict]) -> list[dict]:
    """Deduplicate a list of dicts by their 'name' field, merging descriptions."""
    seen: dict[str, dict] = {}
    for item in items:
        name = item.get("name", "")
        if name in seen:
            # Merge: keep the longest description, fill in missing fields
            existing = seen[name]
            for key, val in item.items():
                if key == "name":
                    continue
                if key not in existing or not existing[key]:
                    existing[key] = val
                elif isinstance(val, str) and len(val) > len(str(existing[key])):
                    existing[key] = val
        else:
            seen[name] = dict(item)
    return list(seen.values())


def _union_strings(*lists: list[str]) -> list[str]:
    """Union multiple string lists, preserving order, deduplicating."""
    seen: set[str] = set()
    result: list[str] = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _merge_key_events(*lists: list[dict]) -> list[dict]:
    """Concatenate key_events from all chunks and sort by 'order' field."""
    all_events: list[dict] = []
    for lst in lists:
        all_events.extend(lst)
    # Sort by order; events without order go last
    all_events.sort(key=lambda e: e.get("order", 9999))
    return all_events


def _keep_longest(*values: str) -> str:
    """Return the longest non-empty string from the arguments."""
    result = ""
    for v in values:
        if v and len(str(v)) > len(str(result)):
            result = v
    return result


def merge_json_objects(*objects: dict) -> dict:
    """Merge multiple JSON objects programmatically.

    Args:
        *objects: Two or more dict objects to merge.

    Returns:
        A single merged dict.
    """
    if not objects:
        return {}

    # Known field merge strategies
    merged: dict = {}

    # Collect all keys across all objects
    all_keys = set()
    for obj in objects:
        all_keys.update(obj.keys())

    for key in all_keys:
        values = [obj.get(key) for obj in objects if key in obj]
        if not values:
            continue

        # Characters: deduplicate by name
        if key == "characters":
            all_chars: list[dict] = []
            for v in values:
                if isinstance(v, list):
                    all_chars.extend(v)
            merged[key] = _dedup_by_name(all_chars)

        # Key events: concatenate and sort by order
        elif key == "key_events":
            all_events: list[dict] = []
            for v in values:
                if isinstance(v, list):
                    all_events.extend(v)
            merged[key] = _merge_key_events(all_events)

        # String arrays: union and deduplicate
        elif key in ("places", "themes", "entities", "species", "technology"):
            lists = [v for v in values if isinstance(v, list)]
            merged[key] = _union_strings(*lists)

        # Scalar fields: keep the longest
        elif key in ("summary", "moral_dilemma", "title"):
            merged[key] = _keep_longest(*[str(v) for v in values if v])

        # Unknown list fields: union
        elif isinstance(values[0], list):
            all_items: list = []
            for v in values:
                if isinstance(v, list):
                    all_items.extend(v)
            # Deduplicate if items are hashable (strings, numbers)
            try:
                seen: set = set()
                result: list = []
                for item in all_items:
                    if item not in seen:
                        seen.add(item)
                        result.append(item)
                merged[key] = result
            except TypeError:
                merged[key] = all_items

        # Unknown scalar fields: keep the longest
        else:
            merged[key] = _keep_longest(*[str(v) for v in values if v])

    return merged


def merge_json_chunks(chunk_results: list[str]) -> str:
    """Merge a list of JSON string results into a single JSON string.

    Each chunk result is expected to be a JSON object (possibly wrapped in
    markdown fences). Invalid chunks are skipped — the rest are merged.

    Args:
        chunk_results: List of JSON strings from chunk processing.

    Returns:
        A single JSON string with all results merged.
    """
    if not chunk_results:
        return "{}"

    objects: list[dict] = []
    for chunk_str in chunk_results:
        obj = _parse_json_safely(chunk_str)
        if obj is not None:
            objects.append(obj)

    if not objects:
        return "{}"

    merged = merge_json_objects(*objects)
    return json.dumps(merged, ensure_ascii=False, indent=2)
