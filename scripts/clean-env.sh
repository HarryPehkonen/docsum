#!/usr/bin/env bash
#
# The clean-environment stage for docsum: does the COMMITTED tree work somewhere
# that is not this dev checkout?
#
# docsum is not a packaged repo - there is no wheel to build, so the artifact is
# the tree itself plus exactly the dependencies requirements.txt declares. Create
# a throwaway venv under $TMPDIR, install that list into it, run the suite from
# THAT interpreter, and ask the CLI for --help: a module-level import error shows
# up here and nowhere else. Nothing is ever installed into the tree.
#
# pytest's exit 5 ("no tests collected") is passed through unchanged, so the
# stage's own `fail_on = "exit:5"` in gate.toml catches it even here.
#
# Called from gate.toml as `[stage.cleanenv] cmd = "scripts/clean-env.sh"`.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export NO_COLOR=1 UV_NO_PROGRESS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="."

status=0
fail() { printf 'clean env: %s\n' "$1" >&2; status=1; }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/docsum-clean-XXXXXX")"
trap 'rm -rf "$tmp"' EXIT INT TERM

if command -v uv >/dev/null 2>&1; then
    uv venv --quiet "$tmp/venv" > "$tmp/venv.log" 2>&1
else
    python3 -m venv "$tmp/venv" > "$tmp/venv.log" 2>&1
fi
py="$tmp/venv/bin/python"
if [ ! -x "$py" ]; then
    tail -10 "$tmp/venv.log"
    fail "the throwaway venv could not be created (python3 -m venv / uv venv)"
    exit 1
fi

install_rc=0
if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$py" -r requirements.txt > "$tmp/install.log" 2>&1 || install_rc=$?
    frozen="$(uv pip freeze --python "$py" 2>/dev/null)"
else
    "$py" -m pip install --quiet -r requirements.txt > "$tmp/install.log" 2>&1 || install_rc=$?
    frozen="$("$py" -m pip freeze 2>/dev/null)"
fi
if [ "$install_rc" -ne 0 ]; then
    tail -15 "$tmp/install.log"
    fail "requirements.txt does not install into a fresh venv - a clean machine cannot reproduce this tree"
    exit 1
fi
# An unpinned list resolves to whatever the index had today, so print the answer:
# that is the evidence a "it passed here, not there" report needs.
printf '%s\n' "$frozen" | sed 's/^/  /'

if "$py" -c 'import pytest' >/dev/null 2>&1; then
    :
else
    printf '  NOTE: pytest is not in requirements.txt - installed for this run only; a clean machine would have no test runner\n'
    "$py" -m pip install --quiet pytest >/dev/null 2>&1
fi

"$py" -m pytest -q -p no:cacheprovider > "$tmp/pytest.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    tail -2 "$tmp/pytest.log" | sed 's/^/  /'
elif [ "$rc" -eq 5 ]; then
    fail "pytest collected no tests in the clean venv - the gate would pass vacuously"
    exit 5
else
    grep -E '^(FAILED|ERROR)|assert|Error' "$tmp/pytest.log" | head -30 | sed 's/^/  /'
    printf '  the suite does not pass in a fresh venv with only requirements.txt installed\n'
    exit "$rc"
fi

# A CLI is what a user actually runs. Prove the clean venv can run it.
if "$py" -m docsum --help >/dev/null 2>"$tmp/smoke.log"; then
    echo "  python -m docsum --help answers in the throwaway venv"
else
    tail -10 "$tmp/smoke.log" | sed 's/^/  /'
    fail "python -m docsum --help fails in the throwaway venv"
fi

exit "$status"
