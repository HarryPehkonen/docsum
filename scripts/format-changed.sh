#!/usr/bin/env bash
#
# The format stage for docsum: check ruff format against the files THIS BRANCH
# touches, not the whole tree.
#
# Why: this repo has pre-existing unformatted files (measured 2026-10-06: 10 of
# 25 .py files would be reformatted). Formatting the whole tree every run reports
# debt nobody wrote in this change, which is how a gate teaches people to ignore
# it. The file list is the kit's python gate's list, unchanged: the branch diff
# against origin/main, the working tree, and untracked files; on a checkout level
# with origin/main (a fresh clone) the last commit is the answer.
#
# Called from gate.toml as `[stage.format] cmd = "scripts/format-changed.sh"`.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

note=""
if git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    base="$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD)"
    touched="$(
        { git diff --name-only --diff-filter=ACMR "$base" HEAD
          git diff --name-only --diff-filter=ACMR HEAD
          # New files are invisible to `git diff` until they are staged.
          git ls-files --others --exclude-standard
        } | sort -u
    )"
    if [ -z "$touched" ]; then
        touched="$(git show --name-only --pretty=format: HEAD | sed '/^$/d')"
        note=" (last commit: this checkout is level with origin/main)"
    fi
else
    # No commits yet: the first commit's files are in the index and nowhere else.
    touched="$(git diff --cached --name-only --diff-filter=ACMR)"
    note=" (first commit)"
fi

files="$(printf '%s\n' "$touched" | grep -E '\.pyi?$' || true)"
if [ -z "$files" ]; then
    printf 'nothing to check: no changed .py file on this branch%s\n' "$note"
    exit 0
fi
printf 'checking %s changed .py file(s)%s\n' "$(printf '%s\n' "$files" | grep -c .)" "$note"
# shellcheck disable=SC2086
exec scripts/py-tool.sh ruff format --check $files
