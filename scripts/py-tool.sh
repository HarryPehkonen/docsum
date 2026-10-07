#!/usr/bin/env bash
#
# How this repo finds the python tools it gates with: ruff, pytest, mypy.
#
#   .venv/bin/<tool>    ->  <tool> on PATH  ->  `uv run --with <tool>`, fetched on demand
#
# That is the kit's python gate's ladder, kept in one place so the policy file
# (gate.toml) can say `scripts/py-tool.sh ruff check .` instead of repeating it.
# docsum pins none of these tools (requirements.txt is the runtime list), so on a
# machine with only uv the third rung is what answers. A missing tool is never a
# silent pass: the last rung fails loudly.
#
# Usage:  scripts/py-tool.sh <tool> [args...]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# ANSI escapes defeat greps and logs; no progress bars, no .pyc litter in the tree.
export NO_COLOR=1 UV_NO_PROGRESS=1 PYTHONDONTWRITEBYTECODE=1
# Flat layout (the package sits at the repo root): the tools must see the sys.path
# this tree declares, never an inherited PYTHONPATH pointing at another copy.
export PYTHONPATH="."

tool="${1:?usage: scripts/py-tool.sh <tool> [args...]}"
shift

if [ -x ".venv/bin/$tool" ]; then
    exec ".venv/bin/$tool" "$@"
elif command -v "$tool" >/dev/null 2>&1; then
    exec "$tool" "$@"
elif command -v uv >/dev/null 2>&1; then
    case "$tool" in
        # pytest comes along on purpose: the test files import it, and mypy would
        # otherwise report import-not-found for every one of them.
        pytest) exec uv run --quiet --with pytest --with-requirements requirements.txt python -m pytest "$@" ;;
        mypy)   exec uv run --quiet --with mypy --with pytest python -m mypy "$@" ;;
        ruff)   exec uv run --quiet --with ruff ruff "$@" ;;
        *)      exec uv run --quiet --with "$tool" "$tool" "$@" ;;
    esac
else
    printf 'GATE FAILED: %s is not installed and uv is not available - nothing was checked\n' "$tool" >&2
    exit 127
fi
