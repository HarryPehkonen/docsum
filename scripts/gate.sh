#!/usr/bin/env bash
#
# The gate - run this before you push (and before you deploy).
#
# The POLICY is gate.toml (the stages, their tiers, their failure rules). The
# ENGINE is kit-ci, one binary installed once per machine:
#
#     cmake -S ~/hermes-workspace/KitCI -B build && cmake --build build
#     cmake --install build --prefix ~/.local        # -> ~/.local/bin/kit-ci
#
# This file is what all three callers run, so "the gate passed" means one thing
# no matter who says it:
#
#   1. a human, by hand            scripts/gate.sh [--tier fast]
#   2. git, on commit and on push  .githooks/pre-commit (fast) and pre-push (full)
#   3. a clean checkout elsewhere  a nightly job: fresh `git clone` into a temp
#                                  dir, then this same script
#
# Changing the gate means editing gate.toml, not this file (2026-10-06, card
# t_5ee2ee12: this script used to be a 552-line copy of the kit's python gate).
#
# Bypass deliberately, never accidentally:  git push --no-verify
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# A gate that cannot find its engine must fail loudly rather than read as green.
if ! type -P kit-ci >/dev/null 2>&1; then
    printf 'gate: kit-ci is not installed. Build KitCI, then: cmake --install build --prefix ~/.local\n' >&2
    exit 1
fi

# With no arguments: the tier the caller names (GATE_TIER, the variable this
# script used before the conversion, still works) and otherwise the full tier.
[ "$#" -gt 0 ] && exec kit-ci "$@"
exec kit-ci --tier "${GATE_TIER:-full}"
