"""Programmatic JSON merge for chunk results — no LLM call needed.

When chunk results are JSON objects (e.g., from a JSON-extraction prompt),
the reduce step can merge them programmatically instead of calling the LLM.
This is faster, free, deterministic, and never times out.

Merge rules:
- characters: deduplicate by name, merge descriptions (keep longest)
- places, themes, entities: union as string arrays, deduplicate
- key_events: concatenate all in document order
- summary, moral_dilemma: keep the longest (most detailed)
- unknown array fields: union and deduplicate
- unknown object fields: merge recursively
- unknown scalar fields: keep the most detailed value, preserving its type
"""

import json
import re

# Sort position for events with no usable "order" field — they go last.
_NO_ORDER = 9999


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


def _identity(item) -> tuple:
    """A hashable identity for a list item, including dicts and lists."""
    try:
        hash(item)
    except TypeError:
        return ("json", json.dumps(item, sort_keys=True, default=str))
    return ("value", item)


def _dedup_by_name(items: list) -> list:
    """Deduplicate a list of items by name, merging their other fields.

    Items are usually dicts with a 'name' field, but models sometimes return
    bare strings instead (["Frodo", "Sam"]). A string and a dict with the same
    name collapse into the dict, which carries more detail.
    """
    seen: dict = {}
    for item in items:
        if isinstance(item, dict):
            key = item.get("name", "")
        elif isinstance(item, str):
            key = item
        else:
            key = _identity(item)

        if key not in seen:
            seen[key] = dict(item) if isinstance(item, dict) else item
            continue

        existing = seen[key]
        if not isinstance(item, dict):
            continue  # a bare string adds nothing to what we already have
        if not isinstance(existing, dict):
            seen[key] = dict(item)  # the dict carries more detail than the string
            continue
        # Merge: keep the longest description, fill in missing fields
        for field, val in item.items():
            if field == "name":
                continue
            if field not in existing or not existing[field]:
                existing[field] = val
            elif isinstance(val, str) and len(val) > len(str(existing[field])):
                existing[field] = val
    return list(seen.values())


def _union_strings(*lists: list) -> list:
    """Union multiple lists, preserving order, deduplicating.

    Items are usually strings, but models sometimes return objects instead
    ({"name": "Bridge"}), so unhashable items are keyed by a canonical
    encoding rather than crashing the merge.
    """
    seen: set = set()
    result: list = []
    for lst in lists:
        for item in lst:
            key = _identity(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def _event_order(event) -> int:
    """The 'order' field of an event as an int; models sometimes return "3"."""
    if not isinstance(event, dict):
        return _NO_ORDER
    try:
        return int(event.get("order", _NO_ORDER))
    except (TypeError, ValueError):
        return _NO_ORDER


def _merge_key_events(*lists: list[dict]) -> list[dict]:
    """Concatenate key_events from every chunk, in document order.

    Each chunk usually numbers its own events from 1, so a global sort by
    "order" interleaves the chunks (1, 1, 2, 2, ...). Sort globally only when
    the order values are unique across all chunks — the one case where they
    describe a single document-wide sequence. Otherwise keep chunk order,
    which is document order.
    """
    numbered = [
        (chunk_index, _event_order(event), event)
        for chunk_index, lst in enumerate(lists)
        for event in lst
    ]
    orders = [order for _, order, _ in numbered if order != _NO_ORDER]
    if len(orders) == len(set(orders)):
        numbered.sort(key=lambda item: item[1])
    else:
        numbered.sort(key=lambda item: (item[0], item[1]))
    return [event for _, _, event in numbered]


def _keep_longest(*values):
    """Return the most detailed value, preserving its original type.

    Numbers are compared numerically (the larger wins); everything else is
    compared by the length of its string form, so the most detailed text
    survives. Values are returned as-is — an int stays an int, a bool a bool.
    """
    candidates = [v for v in values if v is not None and v != ""]
    if not candidates:
        return ""
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in candidates):
        return max(candidates)
    result = candidates[0]
    for v in candidates[1:]:
        if len(str(v)) > len(str(result)):
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

    # Collect all keys across all objects, in first-seen order so the merged
    # output is deterministic
    all_keys: dict = {}
    for obj in objects:
        for key in obj:
            all_keys[key] = None

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

        # Key events: concatenate in document order (one list per chunk, so
        # _merge_key_events can tell chunk-local numbering from a global one)
        elif key == "key_events":
            merged[key] = _merge_key_events(*[v for v in values if isinstance(v, list)])

        # String arrays: union and deduplicate
        elif key in ("places", "themes", "entities", "species", "technology"):
            lists = [v for v in values if isinstance(v, list)]
            merged[key] = _union_strings(*lists)

        # Scalar fields: keep the longest
        elif key in ("summary", "moral_dilemma", "title"):
            merged[key] = _keep_longest(*values)

        # Unknown list fields: union
        elif isinstance(values[0], list):
            merged[key] = _union_strings(*[v for v in values if isinstance(v, list)])

        # Unknown object fields: merge recursively
        elif all(isinstance(v, dict) for v in values):
            merged[key] = merge_json_objects(*values)

        # Unknown scalar fields: keep the most detailed value, as-is
        else:
            merged[key] = _keep_longest(*values)

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
