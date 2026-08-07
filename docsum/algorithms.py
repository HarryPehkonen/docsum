"""Summarization algorithms for documents larger than the LLM context window.

Three algorithms are provided:
- map_reduce: chunk → summarize each → combine (parallelizable)
- refine: chunk → summarize → refine with next chunk → repeat (sequential)
- hierarchical: map-reduce with recursive reduction when summaries are still too large

All algorithms accept an optional `progress` callback for reporting progress:
    def progress(phase: str, current: int, total: int): ...
"""

import re
from typing import Callable, Optional

from docsum.chunker import chunk_text, count_tokens
from docsum.llm_client import LLMClient
from docsum.prompts import render_prompt, render_reduce_prompt

ProgressCallback = Optional[Callable[[str, int, int], None]]

_REFINE_PLACEHOLDER_RE = re.compile(r"\{(summary|text)\}")

JSON_REFINE_INSTRUCTION = (
    "Here is a current JSON analysis of a document, followed by the next section of that document. "
    "Update the JSON to incorporate the new information. "
    "You MUST preserve ALL existing fields — do not drop any field even if the new section doesn't mention it. "
    "Merge new entries into existing arrays (characters, places, themes, key_events, entities, species, technology). "
    "Deduplicate by name where applicable. Keep key_events sorted by order. "
    "Return ONLY valid JSON — no prose, no markdown fences.\n\n"
    "Current JSON:\n{summary}\n\n"
    "New text:\n{text}"
)

PROSE_REFINE_INSTRUCTION = (
    "Here is a current summary of a document, followed by the next section of that document. "
    "Update the summary to incorporate the new information. Keep the summary concise and "
    "well-organized. Preserve important details from both the existing summary and the new text.\n\n"
    "Current summary:\n{summary}\n\n"
    "New text:\n{text}"
)


def _noop_progress(phase: str, current: int, total: int) -> None:
    pass


def wants_json(prompt_template: str) -> bool:
    """Whether a chunk prompt asks for JSON output.

    Any mention of JSON counts: prompts phrase the request many ways ("return
    only valid JSON", "respond in JSON format", "a JSON object with..."), and
    falling through to prose mode drops fields on every refine step.
    """
    return "json" in prompt_template.lower()


def refine_instruction_for(prompt_template: str) -> str:
    """The refine instruction matching the chunk prompt's output format."""
    return JSON_REFINE_INSTRUCTION if wants_json(prompt_template) else PROSE_REFINE_INSTRUCTION


def fill_refine_instruction(instruction: str, summary: str, chunk: str) -> str:
    """Substitute {summary} and {text} in a single pass.

    Sequential .replace() calls would let a literal "{text}" in the running
    summary be replaced by the chunk body on the next iteration, corrupting
    the summary.
    """
    values = {"summary": summary, "text": chunk}
    return _REFINE_PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], instruction)


def map_reduce(
    text: str,
    client: LLMClient,
    prompt_template: str,
    reduce_template: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    model: str = "gpt-4",
    max_output_tokens: int = 8192,
    progress: ProgressCallback = None,
) -> str:
    """Summarize text using the map-reduce algorithm.

    1. Split text into chunks that fit the context window.
    2. Summarize each chunk independently (map).
    3. Combine all summaries into a final summary (reduce).

    Args:
        text: The text to summarize.
        client: LLM client for making completion calls.
        prompt_template: Template with {text} for chunk summarization.
        reduce_template: Template with {summaries} for combining summaries.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks.
        model: Model name for tokenization.
        max_output_tokens: Maximum tokens for LLM response per call.
        progress: Optional callback(phase, current, total) for progress reporting.

    Returns:
        The final combined summary.
    """
    progress = progress or _noop_progress
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model=model)

    if len(chunks) == 0:
        return ""
    if len(chunks) == 1:
        progress("map", 1, 1)
        prompt = render_prompt(prompt_template, chunks[0])
        return client.complete(prompt, max_tokens=max_output_tokens)

    # Map: summarize each chunk
    chunk_summaries = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        progress("map", i + 1, total)
        prompt = render_prompt(prompt_template, chunk)
        summary = client.complete(prompt, max_tokens=max_output_tokens)
        chunk_summaries.append(summary)

    # Reduce: combine all summaries
    progress("reduce", 1, 1)
    reduce_prompt = render_reduce_prompt(reduce_template, chunk_summaries)
    return client.complete(reduce_prompt, max_tokens=max_output_tokens)


def refine(
    text: str,
    client: LLMClient,
    prompt_template: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    model: str = "gpt-4",
    max_output_tokens: int = 8192,
    progress: ProgressCallback = None,
) -> str:
    """Summarize text using the iterative refinement algorithm.

    1. Split text into chunks.
    2. Summarize the first chunk.
    3. For each subsequent chunk, pass the running summary + new chunk to the LLM,
       asking it to refine the summary with the new content.
    4. The final running summary is the result.

    Args:
        text: The text to summarize.
        client: LLM client for making completion calls.
        prompt_template: Template with {text} for chunk summarization.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks.
        model: Model name for tokenization.
        max_output_tokens: Maximum tokens for LLM response per call.
        progress: Optional callback(phase, current, total) for progress reporting.

    Returns:
        The refined summary.
    """
    progress = progress or _noop_progress
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model=model)

    if len(chunks) == 0:
        return ""
    if len(chunks) == 1:
        progress("refine", 1, 1)
        prompt = render_prompt(prompt_template, chunks[0])
        return client.complete(prompt, max_tokens=max_output_tokens)

    total = len(chunks)

    # First chunk: simple summary
    progress("refine", 1, total)
    first_prompt = render_prompt(prompt_template, chunks[0])
    running_summary = client.complete(first_prompt, max_tokens=max_output_tokens)

    # Subsequent chunks: refine the running summary
    # Detect JSON output and reinforce the schema to prevent field loss
    refine_instruction = refine_instruction_for(prompt_template)

    for i, chunk in enumerate(chunks[1:], start=2):
        progress("refine", i, total)
        prompt = fill_refine_instruction(refine_instruction, running_summary, chunk)
        running_summary = client.complete(prompt, max_tokens=max_output_tokens)

    return running_summary


def hierarchical(
    text: str,
    client: LLMClient,
    prompt_template: str,
    reduce_template: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    model: str = "gpt-4",
    max_output_tokens: int = 8192,
    progress: ProgressCallback = None,
    _max_reduce_tokens: int = 2000,
) -> str:
    """Summarize text using the hierarchical map-reduce algorithm.

    Like map-reduce, but if the combined chunk summaries are too large to fit
    in a single context window, recursively applies map-reduce to the summaries
    themselves until the result fits.

    Args:
        text: The text to summarize.
        client: LLM client for making completion calls.
        prompt_template: Template with {text} for chunk summarization.
        reduce_template: Template with {summaries} for combining summaries.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks.
        model: Model name for tokenization.
        max_output_tokens: Maximum tokens for LLM response per call.
        progress: Optional callback(phase, current, total) for progress reporting.
        _max_reduce_tokens: Token limit for the reduce step (internal recursion).

    Returns:
        The final hierarchical summary.
    """
    progress = progress or _noop_progress
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model=model)

    if len(chunks) == 0:
        return ""
    if len(chunks) == 1:
        progress("map", 1, 1)
        prompt = render_prompt(prompt_template, chunks[0])
        return client.complete(prompt, max_tokens=max_output_tokens)

    # Map: summarize each chunk
    chunk_summaries = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        progress("map", i + 1, total)
        prompt = render_prompt(prompt_template, chunk)
        summary = client.complete(prompt, max_tokens=max_output_tokens)
        chunk_summaries.append(summary)

    # Reduce: combine summaries, recursively if needed
    progress("reduce", 1, 1)
    return _recursive_reduce(
        summaries=chunk_summaries,
        client=client,
        reduce_template=reduce_template,
        max_tokens=_max_reduce_tokens,
        model=model,
        max_output_tokens=max_output_tokens,
    )


def _recursive_reduce(
    summaries: list[str],
    client: LLMClient,
    reduce_template: str,
    max_tokens: int,
    model: str = "gpt-4",
    max_output_tokens: int = 8192,
) -> str:
    """Recursively reduce summaries until they fit in a single LLM call.

    If the combined summaries fit within max_tokens, do a single reduce call.
    Otherwise, chunk the summaries, reduce each group, and recurse.
    """
    # Combine all summaries with separators
    combined = "\n\n---\n\n".join(summaries)

    if count_tokens(combined, model) <= max_tokens:
        # Fits in one call — do the final reduce
        reduce_prompt = render_reduce_prompt(reduce_template, summaries)
        return client.complete(reduce_prompt, max_tokens=max_output_tokens)

    # Too large — chunk the summaries and reduce each group
    summary_chunks = chunk_text(combined, max_tokens=max_tokens, model=model)

    if len(summary_chunks) <= 1:
        # Edge case: a single summary chunk that's still too large
        # Just send it — the LLM may truncate, but we can't split further
        reduce_prompt = render_reduce_prompt(reduce_template, summaries)
        return client.complete(reduce_prompt, max_tokens=max_output_tokens)

    # Reduce each group
    reduced_summaries = []
    for chunk in summary_chunks:
        prompt = render_reduce_prompt(reduce_template, [chunk])
        reduced = client.complete(prompt, max_tokens=max_output_tokens)
        reduced_summaries.append(reduced)

    # Recurse on the reduced summaries
    return _recursive_reduce(
        summaries=reduced_summaries,
        client=client,
        reduce_template=reduce_template,
        max_tokens=max_tokens,
        model=model,
        max_output_tokens=max_output_tokens,
    )
