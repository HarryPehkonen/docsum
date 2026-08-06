#!/usr/bin/env python3
"""CLI entry point for docsum — document summarizer.

Usage:
    docsum --input FILE --model MODEL [options]

Examples:
    # Basic summary via Hermes proxy
    docsum --input transcript.txt --model z-ai/glm-5.2

    # Theme extraction with map-reduce
    docsum --input book.txt --model z-ai/glm-5.2 --mode map-reduce --prompt themes

    # Custom prompt from file
    docsum --input report.txt --model nvidia/nemotron-3-super-120b-a12b --prompt-file my_prompt.txt

    # Quiet mode (no progress bar, for piping/gateway use)
    docsum --input book.txt --model z-ai/glm-5.2 --quiet
"""

import argparse
import sys

from docsum.algorithms import map_reduce, refine, hierarchical
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS, get_builtin_prompt


def _make_progress_bar(quiet: bool):
    """Create a progress callback. Returns a no-op if quiet is True.

    Uses tqdm for a visual progress bar on stderr. The bar auto-detects
    terminal width and cleans up after completion.
    """
    if quiet:
        return None

    try:
        from tqdm import tqdm
    except ImportError:
        # tqdm not installed — fall back to simple stderr messages
        def simple_progress(phase: str, current: int, total: int) -> None:
            if current == 1:
                print(f"  {phase}: {current}/{total}", file=sys.stderr, flush=True)
            elif current == total:
                print(f"  {phase}: {current}/{total} done", file=sys.stderr, flush=True)
        return simple_progress

    # tqdm progress bar
    bar_holder: dict = {}

    def tqdm_progress(phase: str, current: int, total: int) -> None:
        if current == 1:
            # Start a new bar for this phase
            if bar_holder.get("bar"):
                bar_holder["bar"].close()
            bar_holder["bar"] = tqdm(
                total=total,
                desc=f"  {phase}",
                unit="chunk",
                leave=True,
                file=sys.stderr,
            )
        if bar_holder.get("bar"):
            bar_holder["bar"].update(1)
            if current == total:
                bar_holder["bar"].close()
                bar_holder["bar"] = None

    return tqdm_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docsum",
        description="Summarize large documents using LLMs with algorithms that handle context limits.",
    )

    # Input / output
    parser.add_argument("--input", "-i", required=True, help="input text file")
    parser.add_argument("--output", "-o", help="output file (default: stdout)")

    # LLM connection
    parser.add_argument("--model", "-m", required=True, help="model ID (e.g., z-ai/glm-5.2)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8645/v1", help="OpenAI-compatible API endpoint (default: Hermes proxy)")
    parser.add_argument("--api-key", default="proxy", help="API key (default: 'proxy' for Hermes proxy)")

    # Algorithm
    parser.add_argument(
        "--mode",
        choices=["map-reduce", "refine", "hierarchical"],
        default="map-reduce",
        help="summarization algorithm (default: map-reduce)",
    )
    parser.add_argument("--max-tokens", type=int, default=2000, help="max tokens per chunk (default: 2000)")
    parser.add_argument("--overlap-tokens", type=int, default=0, help="token overlap between chunks (default: 0)")

    # Output control
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="max tokens for LLM response per call (default: 8192)")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress progress bar (for piping/gateway use)")

    # Prompt
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        choices=list(BUILTIN_PROMPTS.keys()),
        default="summary",
        help="built-in prompt template (default: summary)",
    )
    prompt_group.add_argument("--prompt-file", help="file containing a custom prompt template with {text}")

    # Model for tokenization
    parser.add_argument("--tokenizer-model", default="gpt-4", help="model name for token counting (default: gpt-4)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Read input file
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: could not read input file: {e}", file=sys.stderr)
        return 1

    # Load prompt template
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            print(f"error: prompt file not found: {args.prompt_file}", file=sys.stderr)
            return 1
    else:
        prompt_template = get_builtin_prompt(args.prompt)

    reduce_template = get_builtin_prompt("reduce")

    # Create LLM client
    client = LLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )

    # Create progress callback
    progress = _make_progress_bar(args.quiet)

    # Run the selected algorithm
    if args.mode == "map-reduce":
        result = map_reduce(
            text=text,
            client=client,
            prompt_template=prompt_template,
            reduce_template=reduce_template,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            model=args.tokenizer_model,
            max_output_tokens=args.max_output_tokens,
            progress=progress,
        )
    elif args.mode == "refine":
        result = refine(
            text=text,
            client=client,
            prompt_template=prompt_template,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            model=args.tokenizer_model,
            max_output_tokens=args.max_output_tokens,
            progress=progress,
        )
    elif args.mode == "hierarchical":
        result = hierarchical(
            text=text,
            client=client,
            prompt_template=prompt_template,
            reduce_template=reduce_template,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            model=args.tokenizer_model,
            max_output_tokens=args.max_output_tokens,
            progress=progress,
        )
    else:
        print(f"error: unknown mode: {args.mode}", file=sys.stderr)
        return 1

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
