#!/usr/bin/env bash
#
# The artifact-identity stage for docsum: two copies of one number must agree.
# It is the cheapest possible guard against the most common silent failure - a
# green build that ships something stale.
#
# docsum carries no packaging metadata, so the declared copy is the tracked
# VERSION file and the code's copy is docsum.__version__ (tests/test_version_contract.py
# asserts the same pair, one suite-run later). Bump both from the same edit.
#
# Called from gate.toml as `[stage.identity] cmd = "scripts/identity.sh"`.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="."

status=0
fail() { printf 'identity: %s\n' "$1" >&2; status=1; }

declared="$(sed -n 's/^[[:space:]]*\([0-9][^[:space:]]*\)[[:space:]]*$/\1/p' VERSION | head -1)"
if [ -z "$declared" ]; then
    fail "VERSION does not hold a version on a line of its own"
fi

reported="$(python3 -c 'import docsum as m; print(getattr(m, "__version__", "NO __version__"))' 2>&1 | tail -1)"
printf '  VERSION says %s, the code the tree reports says %s\n' "${declared:-<not found>}" "$reported"
if [ "$reported" = "NO __version__" ]; then
    fail "docsum has no __version__ - the code cannot report which build it is"
elif [ -n "$declared" ] && [ "$declared" != "$reported" ]; then
    fail "VERSION and the code disagree about the version (bump both from the same edit)"
fi

exit "$status"
