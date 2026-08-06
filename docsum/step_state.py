"""State management for step-by-step document processing.

Tracks which chunks have been processed and their results, so a long
document can be processed one chunk at a time across multiple CLI
invocations. The state is persisted as JSON so it survives between calls.

For map-reduce and hierarchical: results[i] = summary of chunk i
For refine: running_summary = the accumulated summary after the latest chunk
"""

import json
from typing import Optional

from dataclasses import dataclass, field


@dataclass
class StepState:
    """Persistent state for step-by-step document processing.

    Attributes:
        state_path: Path to the JSON state file.
        chunks: The text chunks to process.
        results: Summaries produced so far (indexed by chunk index).
        mode: Algorithm mode (map-reduce, refine, hierarchical).
        prompt_template: Template with {text} for chunk summarization.
        reduce_template: Template with {summaries} for combining.
        model: Model ID for the LLM.
        base_url: API endpoint URL.
        max_tokens: Max tokens per chunk (for reference).
        overlap_tokens: Token overlap (for reference).
        max_output_tokens: Max tokens for LLM response.
        tokenizer_model: Model name for tokenization.
        running_summary: For refine mode — the accumulated summary.
        final_result: Set after finalize step.
    """

    state_path: str
    chunks: list[str]
    mode: str
    prompt_template: str
    reduce_template: str
    model: str
    base_url: str
    max_tokens: int = 2000
    overlap_tokens: int = 0
    max_output_tokens: int = 8192
    no_max_output_tokens: bool = False  # if True, omit max_tokens from API call
    stream: bool = False  # if True, use streaming mode for LLM calls
    tokenizer_model: str = "gpt-4"
    results: dict[int, str] = field(default_factory=dict)
    running_summary: Optional[str] = None
    final_result: Optional[str] = None

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def chunks_done(self) -> int:
        return len(self.results)

    def is_complete(self) -> bool:
        return self.chunks_done == self.total_chunks

    def next_chunk_index(self) -> Optional[int]:
        """Return the index of the next unprocessed chunk, or None if all done."""
        for i in range(self.total_chunks):
            if i not in self.results:
                return i
        return None

    def record_result(self, index: int, result: str) -> None:
        """Record the result of processing chunk at the given index.

        For refine mode, also updates running_summary.
        """
        self.results[index] = result
        if self.mode == "refine":
            self.running_summary = result

    def get_results(self) -> list[str]:
        """Return results in chunk order."""
        return [self.results[i] for i in range(self.total_chunks) if i in self.results]

    def status(self) -> dict:
        """Return a status summary dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "chunks_done": self.chunks_done,
            "chunks_remaining": self.total_chunks - self.chunks_done,
            "is_complete": self.is_complete(),
            "mode": self.mode,
            "model": self.model,
            "has_final_result": self.final_result is not None,
        }


def save_state(state: StepState) -> None:
    """Save state to a JSON file."""
    data = {
        "chunks": state.chunks,
        "results": {str(k): v for k, v in state.results.items()},
        "mode": state.mode,
        "prompt_template": state.prompt_template,
        "reduce_template": state.reduce_template,
        "model": state.model,
        "base_url": state.base_url,
        "max_tokens": state.max_tokens,
        "overlap_tokens": state.overlap_tokens,
        "max_output_tokens": state.max_output_tokens,
        "no_max_output_tokens": state.no_max_output_tokens,
        "stream": state.stream,
        "tokenizer_model": state.tokenizer_model,
        "running_summary": state.running_summary,
        "final_result": state.final_result,
    }
    with open(state.state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_state(path: str) -> StepState:
    """Load state from a JSON file.

    Raises:
        FileNotFoundError: If the state file doesn't exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {int(k): v for k, v in data.get("results", {}).items()}
    return StepState(
        state_path=path,
        chunks=data["chunks"],
        results=results,
        mode=data["mode"],
        prompt_template=data["prompt_template"],
        reduce_template=data["reduce_template"],
        model=data["model"],
        base_url=data["base_url"],
        max_tokens=data.get("max_tokens", 2000),
        overlap_tokens=data.get("overlap_tokens", 0),
        max_output_tokens=data.get("max_output_tokens", 8192),
        no_max_output_tokens=data.get("no_max_output_tokens", False),
        stream=data.get("stream", False),
        tokenizer_model=data.get("tokenizer_model", "gpt-4"),
        running_summary=data.get("running_summary"),
        final_result=data.get("final_result"),
    )
