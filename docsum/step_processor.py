"""Step-by-step document processing: prepare, step, finalize.

Breaks the monolithic summarization into resumable steps:
- prepare: chunk the input and save state
- step: process one chunk, save result to state
- finalize: combine all chunk results into the final output
- get_status: report progress without processing

Each step is a single CLI invocation — no long-running process.
State persists as JSON between calls.
"""

from typing import Optional

from docsum.chunker import chunk_text, count_tokens
from docsum.llm_client import LLMClient
from docsum.prompts import render_prompt, render_reduce_prompt, BUILTIN_PROMPTS
from docsum.step_state import StepState, load_state, save_state
from docsum.algorithms import _recursive_reduce


# --- output cleanup --------------------------------------------------------

def _clean_output(text: str) -> str:
    """Clean up LLM output: strip whitespace, markdown fences, and extra braces.

    - Strip leading/trailing whitespace
    - Strip ```json ... ``` or ``` ... ``` markdown fences
    - Strip extra trailing braces/brackets (LLM sometimes appends a stray one)
    """
    text = text.strip()

    # Strip markdown fences: ```json\n...\n``` or ```\n...\n```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove trailing ``` if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Strip extra trailing braces: if there are more closing than opening,
    # remove the extras from the end
    open_count = text.count("{")
    close_count = text.count("}")
    if close_count > open_count:
        text = text.rstrip("}")
        text = text.rstrip()
        # Re-add the correct number of closing braces
        text += "}" * open_count

    return text


# --- prepare ---------------------------------------------------------------

def prepare(
    text: str,
    state_path: str,
    client: LLMClient,
    mode: str,
    prompt_template: str,
    reduce_template: str,
    model: str,
    base_url: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    max_output_tokens: int = 8192,
    no_max_output_tokens: bool = False,
    stream: bool = False,
    tokenizer_model: str = "gpt-4",
) -> StepState:
    """Chunk the input text and initialize the state file.

    Does NOT call the LLM — just sets up the chunks and saves state.

    Returns:
        The initialized StepState.
    """
    chunks = chunk_text(
        text,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        model=tokenizer_model,
    )

    state = StepState(
        state_path=state_path,
        chunks=chunks,
        mode=mode,
        prompt_template=prompt_template,
        reduce_template=reduce_template,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        max_output_tokens=max_output_tokens,
        no_max_output_tokens=no_max_output_tokens,
        stream=stream,
        tokenizer_model=tokenizer_model,
    )
    save_state(state)
    return state


# --- step ------------------------------------------------------------------

def step(state_path: str, client: LLMClient, retry_backoff: float = 0, max_retries: int = 0) -> dict:
    """Process the next unprocessed chunk and save the result.

    Args:
        state_path: Path to the state JSON file.
        client: LLM client for making completion calls.
        retry_backoff: Seconds to wait between retries on failure (0 = no retry).
        max_retries: Maximum number of retries before giving up (default: 2).

    Returns:
        Dictionary with: chunk_index, result, chunks_done, total_chunks, is_complete
    """
    import time

    state = load_state(state_path)

    if state.total_chunks == 0:
        return {
            "chunk_index": None,
            "result": None,
            "chunks_done": 0,
            "total_chunks": 0,
            "is_complete": True,
            "message": "no chunks to process",
        }

    next_idx = state.next_chunk_index()

    if next_idx is None:
        return {
            "chunk_index": None,
            "result": None,
            "chunks_done": state.chunks_done,
            "total_chunks": state.total_chunks,
            "is_complete": True,
            "message": "all chunks already processed",
        }

    chunk_text_to_process = state.chunks[next_idx]

    if state.mode == "refine" and next_idx > 0 and state.running_summary is not None:
        # Refine mode: pass running summary + new chunk
        # Detect JSON output and reinforce the schema to prevent field loss
        if '"json"' in state.prompt_template.lower() or "return only json" in state.prompt_template.lower() or "return only valid json" in state.prompt_template.lower():
            refine_instruction = (
                "Here is a current JSON analysis of a document, followed by the next section of that document. "
                "Update the JSON to incorporate the new information. "
                "You MUST preserve ALL existing fields — do not drop any field even if the new section doesn't mention it. "
                "Merge new entries into existing arrays (characters, places, themes, key_events, entities, species, technology). "
                "Deduplicate by name where applicable. Keep key_events sorted by order. "
                "Return ONLY valid JSON — no prose, no markdown fences.\n\n"
                "Current JSON:\n{summary}\n\n"
                "New text:\n{text}"
            )
        else:
            refine_instruction = (
                "Here is a current summary of a document, followed by the next section of that document. "
                "Update the summary to incorporate the new information. Keep the summary concise and "
                "well-organized. Preserve important details from both the existing summary and the new text.\n\n"
                "Current summary:\n{summary}\n\n"
                "New text:\n{text}"
            )
        prompt = refine_instruction.replace("{summary}", state.running_summary).replace("{text}", chunk_text_to_process)
    else:
        # Map-reduce, hierarchical, or first chunk of refine: simple summarization
        prompt = render_prompt(state.prompt_template, chunk_text_to_process)

    # Determine max_tokens for the API call
    api_max_tokens = None if state.no_max_output_tokens else state.max_output_tokens

    # Retry loop
    last_error = None
    attempts = max_retries + 1  # always at least 1 attempt
    for attempt in range(attempts):
        try:
            result = client.complete(
                prompt,
                max_tokens=api_max_tokens,
                stream=state.stream,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < attempts - 1:
                if retry_backoff > 0:
                    time.sleep(retry_backoff)
                # retry even with 0 backoff if max_retries > 0
            else:
                raise

    state.record_result(next_idx, result)
    save_state(state)

    return {
        "chunk_index": next_idx,
        "result": result,
        "chunks_done": state.chunks_done,
        "total_chunks": state.total_chunks,
        "is_complete": state.is_complete(),
    }


# --- finalize --------------------------------------------------------------

def finalize(state_path: str, client: LLMClient) -> dict:
    """Combine all chunk results into the final output.

    For map-reduce and hierarchical: applies the reduce prompt (recursively
    if needed for hierarchical).
    For refine: the running summary IS the final result.

    Returns:
        Dictionary with: result, is_complete
    """
    state = load_state(state_path)

    if not state.is_complete():
        return {
            "error": "not all chunks have been processed",
            "chunks_done": state.chunks_done,
            "total_chunks": state.total_chunks,
            "is_complete": False,
        }

    if state.total_chunks == 0:
        return {"result": "", "is_complete": True}

    if state.mode == "refine":
        # Refine: running summary is the final result
        final = state.running_summary or state.get_results()[-1] if state.get_results() else ""
        final = _clean_output(final)
        state.final_result = final
        save_state(state)
        return {"result": final, "is_complete": True}

    if state.total_chunks == 1:
        # Single chunk: result is the chunk summary, no reduce needed
        final = _clean_output(state.get_results()[0])
        state.final_result = final
        save_state(state)
        return {"result": final, "is_complete": True}

    # Check if this is a JSON reduce — use programmatic merge (no LLM call)
    from docsum.json_merge import merge_json_chunks
    json_reduce_template = BUILTIN_PROMPTS.get("json_reduce", "")
    if state.reduce_template == json_reduce_template and json_reduce_template:
        # Programmatic merge: fast, free, deterministic, never times out
        merged = merge_json_chunks(state.get_results())
        final = _clean_output(merged)
        state.final_result = final
        save_state(state)
        return {"result": final, "is_complete": True}

    if state.mode == "hierarchical":
        # Hierarchical: recursively reduce if needed
        results = state.get_results()
        final = _recursive_reduce(
            summaries=results,
            client=client,
            reduce_template=state.reduce_template,
            max_tokens=state.max_tokens,
            model=state.tokenizer_model,
            max_output_tokens=state.max_output_tokens,
        )
    else:
        # Map-reduce: single reduce call
        reduce_prompt = render_reduce_prompt(state.reduce_template, state.get_results())
        final = client.complete(reduce_prompt, max_tokens=state.max_output_tokens)

    final = _clean_output(final)
    state.final_result = final
    save_state(state)
    return {"result": final, "is_complete": True}


# --- get_status ------------------------------------------------------------

def get_status(state_path: str) -> dict:
    """Report current progress without processing any chunks.

    Returns:
        Status dictionary from StepState.status().
    """
    state = load_state(state_path)
    return state.status()
