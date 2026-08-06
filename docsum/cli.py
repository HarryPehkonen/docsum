#!/usr/bin/env python3
"""CLI entry point for docsum — document summarizer.

Two modes of operation:

1. Monolithic (default): process the whole document in one invocation.
    docsum --input FILE --model MODEL [options]

2. Step-by-step: process one chunk at a time across multiple invocations.
    docsum prepare --input FILE --model MODEL --state STATE.json [options]
    docsum step --state STATE.json
    ... repeat until complete ...
    docsum finalize --state STATE.json --output result.txt

Step-by-step mode is for agent-orchestrated workflows where each step
is visible, resumable, and retryable. The state persists as JSON between calls.

Examples:
    # Basic summary via Hermes proxy
    docsum --input transcript.txt --model z-ai/glm-5.2

    # Theme extraction with map-reduce
    docsum --input book.txt --model z-ai/glm-5.2 --mode map-reduce --prompt themes

    # Step-by-step with Nemotron (small model, resumable)
    docsum prepare --input episode.txt --model nvidia/nemotron-3-super-120b-a12b \
        --state state.json --max-tokens 500
    docsum step --state state.json
    docsum step --state state.json
    docsum finalize --state state.json --output summary.txt

    # Check progress
    docsum status --state state.json
"""

import argparse
import json
import sys

from docsum.algorithms import map_reduce, refine, hierarchical
from docsum.llm_client import LLMClient
from docsum.prompts import BUILTIN_PROMPTS, get_builtin_prompt
from docsum.step_processor import prepare as step_prepare, step as step_process, finalize as step_finalize, get_status


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


def _load_prompt_template(args):
    """Load prompt template from --prompt-file or --prompt."""
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"error: prompt file not found: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)
    return get_builtin_prompt(args.prompt)


def _print_json(data: dict) -> None:
    """Print a JSON object to stdout."""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docsum",
        description="Summarize large documents using LLMs with algorithms that handle context limits.",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # --- monolithic mode (default, no subcommand) ---
    # Use a "run" subcommand to disambiguate, but also accept bare args
    run_p = subparsers.add_parser("run", help="process the whole document in one invocation (monolithic mode)")

    # Input / output
    run_p.add_argument("--input", "-i", required=True, help="input text file")
    run_p.add_argument("--output", "-o", help="output file (default: stdout)")

    # LLM connection
    run_p.add_argument("--model", "-m", required=True, help="model ID (e.g., z-ai/glm-5.2)")
    run_p.add_argument("--base-url", default="http://127.0.0.1:8645/v1", help="OpenAI-compatible API endpoint (default: Hermes proxy)")
    run_p.add_argument("--api-key", default="proxy", help="API key (default: 'proxy' for Hermes proxy)")

    # Algorithm
    run_p.add_argument(
        "--mode",
        choices=["map-reduce", "refine", "hierarchical"],
        default="map-reduce",
        help="summarization algorithm (default: map-reduce)",
    )
    run_p.add_argument("--max-tokens", type=int, default=2000, help="max tokens per chunk (default: 2000)")
    run_p.add_argument("--overlap-tokens", type=int, default=0, help="token overlap between chunks (default: 0)")

    # Output control
    run_p.add_argument("--max-output-tokens", type=int, default=8192, help="max tokens for LLM response per call (default: 8192)")
    run_p.add_argument("--quiet", "-q", action="store_true", help="suppress progress bar (for piping/gateway use)")

    # Prompt
    run_prompt_group = run_p.add_mutually_exclusive_group()
    run_prompt_group.add_argument(
        "--prompt",
        choices=list(BUILTIN_PROMPTS.keys()),
        default="summary",
        help="built-in prompt template (default: summary)",
    )
    run_prompt_group.add_argument("--prompt-file", help="file containing a custom prompt template with {text}")

    # Model for tokenization
    run_p.add_argument("--tokenizer-model", default="gpt-4", help="model name for token counting (default: gpt-4)")

    # --- prepare subcommand ---
    prep_p = subparsers.add_parser("prepare", help="chunk the input and initialize step-by-step state")

    prep_p.add_argument("--input", "-i", required=True, help="input text file")
    prep_p.add_argument("--state", required=True, help="state file path (will be created)")
    prep_p.add_argument("--model", "-m", required=True, help="model ID")
    prep_p.add_argument("--base-url", default="http://127.0.0.1:8645/v1", help="OpenAI-compatible API endpoint")
    prep_p.add_argument("--api-key", default="proxy", help="API key")

    prep_p.add_argument(
        "--mode",
        choices=["map-reduce", "refine", "hierarchical"],
        default="map-reduce",
        help="summarization algorithm (default: map-reduce)",
    )
    prep_p.add_argument("--max-tokens", type=int, default=2000, help="max tokens per chunk (default: 2000)")
    prep_p.add_argument("--overlap-tokens", type=int, default=0, help="token overlap between chunks (default: 0)")
    prep_p.add_argument("--max-output-tokens", type=int, default=8192, help="max tokens for LLM response per call (default: 8192)")

    prep_prompt_group = prep_p.add_mutually_exclusive_group()
    prep_prompt_group.add_argument(
        "--prompt",
        choices=list(BUILTIN_PROMPTS.keys()),
        default="summary",
        help="built-in prompt template (default: summary)",
    )
    prep_prompt_group.add_argument("--prompt-file", help="file containing a custom prompt template with {text}")

    prep_p.add_argument("--tokenizer-model", default="gpt-4", help="model name for token counting")

    # --- step subcommand ---
    step_p = subparsers.add_parser("step", help="process the next unprocessed chunk")
    step_p.add_argument("--state", required=True, help="state file path")
    step_p.add_argument("--base-url", default="http://127.0.0.1:8645/v1", help="API endpoint")
    step_p.add_argument("--api-key", default="proxy", help="API key")

    # --- finalize subcommand ---
    fin_p = subparsers.add_parser("finalize", help="combine all chunk results into final output")
    fin_p.add_argument("--state", required=True, help="state file path")
    fin_p.add_argument("--output", "-o", help="output file (default: stdout)")
    fin_p.add_argument("--base-url", default="http://127.0.0.1:8645/v1", help="API endpoint")
    fin_p.add_argument("--api-key", default="proxy", help="API key")

    # --- status subcommand ---
    status_p = subparsers.add_parser("status", help="report progress without processing")
    status_p.add_argument("--state", required=True, help="state file path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # If no subcommand, show help
    if not args.subcommand:
        parser.print_help(sys.stderr)
        return 1

    if args.subcommand == "run":
        return _cmd_run(args)
    elif args.subcommand == "prepare":
        return _cmd_prepare(args)
    elif args.subcommand == "step":
        return _cmd_step(args)
    elif args.subcommand == "finalize":
        return _cmd_finalize(args)
    elif args.subcommand == "status":
        return _cmd_status(args)
    else:
        parser.print_help(sys.stderr)
        return 1


def _cmd_run(args) -> int:
    """Monolithic mode: process the whole document in one invocation."""
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: could not read input file: {e}", file=sys.stderr)
        return 1

    prompt_template = _load_prompt_template(args)
    reduce_template = get_builtin_prompt("reduce")

    client = LLMClient(base_url=args.base_url, model=args.model, api_key=args.api_key)
    progress = _make_progress_bar(args.quiet)

    if args.mode == "map-reduce":
        result = map_reduce(
            text=text, client=client, prompt_template=prompt_template,
            reduce_template=reduce_template, max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens, model=args.tokenizer_model,
            max_output_tokens=args.max_output_tokens, progress=progress,
        )
    elif args.mode == "refine":
        result = refine(
            text=text, client=client, prompt_template=prompt_template,
            max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens,
            model=args.tokenizer_model, max_output_tokens=args.max_output_tokens,
            progress=progress,
        )
    elif args.mode == "hierarchical":
        result = hierarchical(
            text=text, client=client, prompt_template=prompt_template,
            reduce_template=reduce_template, max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens, model=args.tokenizer_model,
            max_output_tokens=args.max_output_tokens, progress=progress,
        )
    else:
        print(f"error: unknown mode: {args.mode}", file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)
    return 0


def _cmd_prepare(args) -> int:
    """Prepare: chunk the input and initialize state."""
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1

    prompt_template = _load_prompt_template(args)
    reduce_template = get_builtin_prompt("reduce")

    # Create a client (prepare doesn't call it, but step will need the settings)
    client = LLMClient(base_url=args.base_url, model=args.model, api_key=args.api_key)

    state = step_prepare(
        text=text,
        state_path=args.state,
        client=client,
        mode=args.mode,
        prompt_template=prompt_template,
        reduce_template=reduce_template,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        max_output_tokens=args.max_output_tokens,
        tokenizer_model=args.tokenizer_model,
    )

    _print_json({
        "status": "prepared",
        "total_chunks": state.total_chunks,
        "mode": state.mode,
        "model": state.model,
        "state_file": args.state,
    })
    return 0


def _cmd_step(args) -> int:
    """Step: process the next unprocessed chunk."""
    client = LLMClient(base_url=args.base_url, model="placeholder", api_key=args.api_key)

    # Load state to get the real model
    from docsum.step_state import load_state
    state = load_state(args.state)
    client = LLMClient(base_url=args.base_url, model=state.model, api_key=args.api_key)

    result = step_process(state_path=args.state, client=client)
    _print_json(result)
    return 0


def _cmd_finalize(args) -> int:
    """Finalize: combine all chunk results into final output."""
    from docsum.step_state import load_state
    state = load_state(args.state)
    client = LLMClient(base_url=args.base_url, model=state.model, api_key=args.api_key)

    result = step_finalize(state_path=args.state, client=client)

    if "error" in result:
        _print_json(result)
        return 1

    output_text = result.get("result", "")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        print(output_text)
    return 0


def _cmd_status(args) -> int:
    """Status: report progress without processing."""
    try:
        status = get_status(args.state)
        _print_json(status)
        return 0
    except FileNotFoundError:
        print(f"error: state file not found: {args.state}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
