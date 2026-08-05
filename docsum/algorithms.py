"""Summarization algorithms for documents larger than the LLM context window.

Three algorithms are provided:
- map_reduce: chunk → summarize each → combine (parallelizable)
- refine: chunk → summarize → refine with next chunk → repeat (sequential)
- hierarchical: map-reduce with recursive reduction when summaries are still too large
"""

from docsum.chunker import chunk_text, count_tokens
from docsum.llm_client import LLMClient
from docsum.prompts import render_prompt, render_reduce_prompt


def map_reduce(
    text: str,
    client: LLMClient,
    prompt_template: str,
    reduce_template: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    model: str = "gpt-4",
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

    Returns:
        The final combined summary.
    """
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model=model)

    if len(chunks) == 0:
        return ""
    if len(chunks) == 1:
        prompt = render_prompt(prompt_template, chunks[0])
        return client.complete(prompt)

    # Map: summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        prompt = render_prompt(prompt_template, chunk)
        summary = client.complete(prompt)
        chunk_summaries.append(summary)

    # Reduce: combine all summaries
    reduce_prompt = render_reduce_prompt(reduce_template, chunk_summaries)
    return client.complete(reduce_prompt)


def refine(
    text: str,
    client: LLMClient,
    prompt_template: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    model: str = "gpt-4",
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

    Returns:
        The refined summary.
    """
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model=model)

    if len(chunks) == 0:
        return ""
    if len(chunks) == 1:
        prompt = render_prompt(prompt_template, chunks[0])
        return client.complete(prompt)

    # First chunk: simple summary
    first_prompt = render_prompt(prompt_template, chunks[0])
    running_summary = client.complete(first_prompt)

    # Subsequent chunks: refine the running summary
    refine_instruction = (
        "Here is a current summary of a document, followed by the next section of that document. "
        "Update the summary to incorporate the new information. Keep the summary concise and "
        "well-organized. Preserve important details from both the existing summary and the new text.\n\n"
        "Current summary:\n{summary}\n\n"
        "New text:\n{text}"
    )

    for chunk in chunks[1:]:
        prompt = refine_instruction.replace("{summary}", running_summary).replace("{text}", chunk)
        running_summary = client.complete(prompt)

    return running_summary


def hierarchical(
    text: str,
    client: LLMClient,
    prompt_template: str,
    reduce_template: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    model: str = "gpt-4",
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
        _max_reduce_tokens: Token limit for the reduce step (internal recursion).

    Returns:
        The final hierarchical summary.
    """
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model=model)

    if len(chunks) == 0:
        return ""
    if len(chunks) == 1:
        prompt = render_prompt(prompt_template, chunks[0])
        return client.complete(prompt)

    # Map: summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        prompt = render_prompt(prompt_template, chunk)
        summary = client.complete(prompt)
        chunk_summaries.append(summary)

    # Reduce: combine summaries, recursively if needed
    return _recursive_reduce(
        summaries=chunk_summaries,
        client=client,
        reduce_template=reduce_template,
        max_tokens=_max_reduce_tokens,
        model=model,
    )


def _recursive_reduce(
    summaries: list[str],
    client: LLMClient,
    reduce_template: str,
    max_tokens: int,
    model: str = "gpt-4",
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
        return client.complete(reduce_prompt)

    # Too large — chunk the summaries and reduce each group
    summary_chunks = chunk_text(combined, max_tokens=max_tokens, model=model)

    if len(summary_chunks) <= 1:
        # Edge case: a single summary chunk that's still too large
        # Just send it — the LLM may truncate, but we can't split further
        reduce_prompt = render_reduce_prompt(reduce_template, summaries)
        return client.complete(reduce_prompt)

    # Reduce each group
    reduced_summaries = []
    for chunk in summary_chunks:
        prompt = render_reduce_prompt(reduce_template, [chunk])
        reduced = client.complete(prompt)
        reduced_summaries.append(reduced)

    # Recurse on the reduced summaries
    return _recursive_reduce(
        summaries=reduced_summaries,
        client=client,
        reduce_template=reduce_template,
        max_tokens=max_tokens,
        model=model,
    )
