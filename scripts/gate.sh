#!/usr/bin/env bash
#
# The gate — run this before you push (and before you deploy).
#
# from AI-DEV-STARTER (plunk-in kit) templates/python/gate.sh — if you edit your copy, say
# why in INCIDENTS.md. The copied repo owns its gate; diff it against a newer kit by hand.
#
# Why this file exists instead of CI: every check here is cheap and belongs to the repo,
# not to somebody else's account. One definition, three callers:
#
#   1. a human, by hand            scripts/gate.sh
#   2. git, on commit and on push  .githooks/pre-commit, .githooks/pre-push
#                                  (arm once per clone: git config core.hooksPath .githooks)
#   3. a clean checkout elsewhere  a nightly job: fresh `git clone` into a temp dir,
#                                  then this same script
#
# Running the SAME file in all three places is the point: "the gate passed" then means
# one thing no matter who says it.
#
# It reports EVERY failure rather than stopping at the first, so one run tells you
# everything that is wrong.
#
# TWO KINDS OF PYTHON REPO, and the gate works out which one it has:
#
#   pyproject.toml    packaged. The clean-environment step is: build the distribution,
#                     install the WHEEL into a throwaway venv, and ask THAT venv what the
#                     version is. A repo that only "works" because of an editable install
#                     or a stray .pth file fails here. Identity = pyproject version <->
#                     the version the installed wheel reports.
#   requirements.txt  not packaged. There is no wheel to build, and building one would
#                     mean inventing packaging metadata the repo does not have. The
#                     clean-environment step is instead: create a throwaway venv, install
#                     the requirements into it, and run the SUITE from that venv. That
#                     proves the declared dependency list is complete and that the suite
#                     does not depend on this machine's stale dev venv. Identity = the
#                     repo's declared VERSION file <-> the version the code reports. In this
#                     branch the tree IS the artifact, so every tool runs with the tree's own
#                     layout on sys.path (see tool_env) — a src/ layout is otherwise not
#                     importable by the tests.
#
# A repo with neither file cannot be reproduced on a clean machine at all, so that is a
# FAIL, not a skip.
#
# Tools are found in this order:   .venv/bin/<tool>  ->  <tool> on PATH  ->  `uv run --with
# <tool>` (fetched on demand). Missing tools SKIP loudly (or FAIL with STRICT_TOOLS=1) —
# a gate that quietly checks nothing is worse than no gate, because it looks like one.
# The tools stage prints which of those answered, every run: "the gate passed" and "the
# gate passed using a linter uv downloaded because this repo never declared one" are
# different statements, and only one of them survives a machine that loses the network.
#
# Tiers. GATE_TIER=full (default) runs everything. GATE_TIER=fast skips the throwaway-venv
# stage and says so: creating and populating a venv is push-time work, not commit-time
# work. The identity stage runs in both tiers — in fast mode it reads the source tree, in
# full mode the freshly created venv. Set fast at commit time only when the measured warm
# full run is longer than a commit can afford (the same measured-time rule the C++ tier
# split uses), and let pre-push run the full gate.
#
# Bypass deliberately, never accidentally:  git push --no-verify
#
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
# No colour: the gate greps its own tools' output, and ANSI escapes defeat both the
# greps here and any log this is piped into. No update nagging: it writes to the same
# streams this script parses.
export NO_COLOR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export UV_NO_PROGRESS=1
export PYTHONDONTWRITEBYTECODE=1

PYPROJECT=${PYPROJECT:-pyproject.toml}
REQUIREMENTS=${REQUIREMENTS:-requirements.txt}
STRICT_TOOLS=${STRICT_TOOLS:-0}   # 1 = a missing tool FAILS the run instead of SKIPping
GATE_TIER=${GATE_TIER:-full}      # full | fast (fast = no throwaway venv; commit-time)

# The no-packaging branch's artifact identity, in two names:
#   IDENTITY_FILE         the tracked file holding the declared copy of the version
#   IDENTITY_IMPORT_NAME  the import name to ask; auto-detected when there is only one
IDENTITY_FILE=${IDENTITY_FILE:-VERSION}
IDENTITY_IMPORT_NAME=${IDENTITY_IMPORT_NAME:-}

# Set below by the branch detection. An unpackaged repo declares its dependencies in
# requirements.txt and has no project for `uv run` to sync, so the uv fallback has to be
# told about them: without this, the test stage runs pytest in an environment holding
# nothing but pytest and every test module dies on import (measured: a tree whose suite
# passes 132 tests reported 10 ModuleNotFoundError lines instead).
UV_WITH_REQUIREMENTS=""

status=0
step() { printf '\n== %s\n' "$1"; }
fail() { printf 'GATE FAILED: %s\n' "$1" >&2; status=1; }
skip() { printf 'SKIP: %s (%s)\n' "$1" "$2"; }

# The tool ladder, decided in ONE place so the printed table cannot drift from what runs.
tool_mode() {  # echoes venv|path|uv|none
    if [ -x ".venv/bin/$1" ]; then printf 'venv'
    elif command -v "$1" >/dev/null 2>&1; then printf 'path'
    elif command -v uv >/dev/null 2>&1; then printf 'uv'
    else printf 'none'
    fi
}

tool_where() {  # a human string for the tools table
    case "$(tool_mode "$1")" in
        venv) printf '.venv/bin/%s' "$1" ;;
        path) printf '%s (on PATH)' "$(command -v "$1")" ;;
        uv)
            if [ "$1" = "pytest" ] && [ -n "$UV_WITH_REQUIREMENTS" ]; then
                printf 'uv run --with pytest --with-requirements %s — fetched on demand; this repo does not pin them' "$UV_WITH_REQUIREMENTS"
            else
                printf 'uv run --with %s — fetched on demand; this repo does not pin it' "$1"
            fi ;;
        *)    printf 'MISSING' ;;
    esac
}

require_tool() {  # require_tool <tool> <stage>
    if [ "$(tool_mode "$1")" != "none" ]; then return 0; fi
    if [ "$STRICT_TOOLS" = "1" ]; then
        fail "$1 not installed (STRICT_TOOLS=1) — $2 was not checked"
        return 1
    fi
    skip "$2" "$1 not installed and uv is not available — nothing was checked"
    return 1
}

run_tool() {  # run_tool <tool> <args...>
    # Everything goes through tool_env (defined below, next to py_env): in an unpackaged repo
    # the tool must see the sys.path the tree declares. Its comment names the measured failure
    # this prevents. Outside that branch tool_env is a plain exec.
    local tool="$1"; shift
    case "$(tool_mode "$tool")" in
        venv) tool_env ".venv/bin/$tool" "$@" ;;
        path) tool_env "$tool" "$@" ;;
        uv)
            case "$tool" in
                # --with keeps the tool out of the project's dependency list; `python -m`
                # puts this directory (the repo root) on sys.path, so a flat-layout package
                # resolves exactly the way it does when a human runs it. --with-requirements
                # is what makes this usable in a repo that has no pyproject for uv to sync.
                pytest)
                    if [ -n "$UV_WITH_REQUIREMENTS" ]; then
                        tool_env uv run --quiet --with pytest --with-requirements "$UV_WITH_REQUIREMENTS" python -m pytest "$@"
                    else
                        tool_env uv run --quiet --with pytest python -m pytest "$@"
                    fi ;;
                # pytest comes along on purpose: test files import it, and mypy would
                # otherwise report import-not-found for every one of them.
                mypy) tool_env uv run --quiet --with mypy --with pytest python -m mypy "$@" ;;
                ruff) tool_env uv run --quiet --with ruff ruff "$@" ;;
                *) tool_env uv tool run --quiet "$tool" "$@" ;;
            esac ;;
        *) return 127 ;;
    esac
}

# The files this branch touches. Used by the format stage: a repo has pre-existing
# non-compliant files, and formatting the whole tree every run buries the signal in
# noise nobody edited.
if git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    base="$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD)"
    touched="$(
        { git diff --name-only --diff-filter=ACMR "$base" HEAD
          git diff --name-only --diff-filter=ACMR HEAD
          # New files are invisible to `git diff` until they are staged, so without this
          # line a brand-new file's formatting is never checked at the moment it is
          # written — only after it has already been committed.
          git ls-files --others --exclude-standard
        } | sort -u
    )"
    # A clean checkout (the nightly job, or any clone sitting exactly on origin/main) is
    # not "ahead of" anything, so the diffs above are empty and the checks below would
    # silently pass on files nobody looked at. Fall back to the last commit that landed.
    if [ -z "$touched" ]; then
        touched="$(git show --name-only --pretty=format: HEAD | sed '/^$/d')"
        late_note=" (last commit, since this checkout is level with origin/main)"
    fi
else
    # No commits yet: the first commit's files are in the index and nowhere else.
    touched="$(git diff --cached --name-only --diff-filter=ACMR)"
    late_note=" (first commit)"
fi

ci_version() { sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$PYPROJECT" | head -1; }
ci_dist_name() { sed -n 's/^name *= *"\([^"]*\)".*/\1/p' "$PYPROJECT" | head -1; }
# The import name is not always the distribution name (my-lib vs my_lib): look for the
# package where it actually lives before guessing.
ci_import_name() {
    local guess candidate
    for candidate in "$(ci_dist_name)" "$(ci_dist_name | tr '-' '_')"; do
        for guess in "src/$candidate" "$candidate"; do
            if [ -f "$guess/__init__.py" ]; then printf '%s' "$candidate"; return 0; fi
        done
    done
    return 1
}

# The unpackaged branch has no distribution name to go on: take the one top-level
# directory that holds an __init__.py and is not obviously not a package. A repo with
# several packages sets IDENTITY_IMPORT_NAME — the identity check names exactly one
# package, the one whose version is the artifact's version.
ci_tree_import_name() {
    if [ -n "$IDENTITY_IMPORT_NAME" ]; then printf '%s' "$IDENTITY_IMPORT_NAME"; return 0; fi
    local dir name
    for dir in src/*/ */; do
        dir=${dir%/}; name=${dir#src/}
        [ "$name" = "src" ] && continue
        [ -f "$dir/__init__.py" ] || continue
        case "$name" in
            tests|test|docs|scripts|tools|build|dist|venv|.venv|node_modules) continue ;;
        esac
        printf '%s' "$name"; return 0
    done
    return 1
}

# Where the repo root has to be on sys.path for `python -c`/`python -m` to import the
# package: "." for a flat layout (the repo root is already sys.path[0] for -c and -m, so
# nothing needs exporting) and "src" for a src layout. Empty means "import the installed
# distribution, not the tree" — which is what the packaging branch wants.
PY_IMPORT_PATH="."
for dir in src/*/; do
    [ -f "$dir/__init__.py" ] && PY_IMPORT_PATH=src
done

# Run python with a sys.path that is exactly what this repo declares — never with whatever
# the caller happened to have exported. An inherited PYTHONPATH pointing at another copy of
# the tree is how "it passed on my machine" happens.
py_env() {
    if [ -n "$PY_IMPORT_PATH" ] && [ "$PY_IMPORT_PATH" != "." ]; then
        env -u PYTHONPATH PYTHONPATH="$PY_IMPORT_PATH" "$@"
    else
        env -u PYTHONPATH "$@"
    fi
}

# Every tool the gate runs goes through this. An UNPACKAGED repo has no packaging metadata for
# a runner to install, so the tree IS the artifact and the tools must see the sys.path the tree
# declares — py_env above. MEASURED, 2026-09-20: a src-layout unpackaged repo (package under
# src/, VERSION + requirements.txt at the root, no pyproject) died in the tests stage with
# `ModuleNotFoundError: No module named '<pkg>'`, because pytest's rootdir insertion reaches
# only the tests directory and `python -m pytest` reaches only the repo root, while the
# clean-environment stage — which does use py_env — passed the same suite. A flat layout hides
# it (the package sits on the repo root, which `python -m` already has on sys.path), which is
# why docsum never showed it.
#
# The packaged branch is deliberately left exactly as it was: its tests run in the environment
# the repo configures (uv syncs the project itself, so `uv run` has it installed), and
# PY_IMPORT_PATH is emptied before the smoke import so the identity check asks the installed
# wheel rather than the tree.
tool_env() {  # tool_env <command...>
    if [ "$BRANCH" = "unpackaged" ]; then
        py_env "$@"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------- which kind of repo
if [ -f "$PYPROJECT" ]; then
    BRANCH=packaged
elif [ -f "$REQUIREMENTS" ]; then
    BRANCH=unpackaged
    UV_WITH_REQUIREMENTS="$REQUIREMENTS"
else
    printf 'GATE FAILED: neither %s nor %s is present — a repo that declares neither cannot be rebuilt on a clean machine, and there would be nothing to install, test, or identify\n' \
        "$PYPROJECT" "$REQUIREMENTS" >&2
    exit 1
fi

printf '== python gate: %s branch, tier=%s, repo %s\n' \
    "$BRANCH" "$GATE_TIER" "$(git rev-parse --short HEAD 2>/dev/null || echo 'no commits yet')"

# ---------------------------------------------------------------- 0. tools
step "tools (which of them answers, and where from)"
printf '  python3    %s\n' "$(python3 -V 2>&1 || echo 'MISSING')"
printf '  uv         %s\n' "$(uv --version 2>&1 | head -1 || echo 'not installed')"
printf '  ruff       %s\n' "$(tool_where ruff)"
printf '  pytest     %s\n' "$(tool_where pytest)"
if grep -qE '^\[tool\.mypy\]|^\[mypy\]' "$PYPROJECT" 2>/dev/null || [ -f mypy.ini ] || [ -f .mypy.ini ]; then
    printf '  mypy       %s\n' "$(tool_where mypy)"
fi

# ---------------------------------------------------------------- 1. lint
step "lint"
if require_tool ruff "lint"; then
    if run_tool ruff check . > /tmp/.gate-ruff-lint.$$ 2>&1; then
        printf '  %s\n' "$(grep -E '^All checks passed|^Found 0 errors' /tmp/.gate-ruff-lint.$$ || echo 'clean')"
    else
        head -40 /tmp/.gate-ruff-lint.$$
        printf '  full output: ruff check .\n'
        fail "ruff check . (findings above)"
    fi
    rm -f /tmp/.gate-ruff-lint.$$
fi

# ---------------------------------------------------------------- 2. format (touched files only)
files="$(printf '%s\n' "$touched" | grep -E '\.pyi?$' || true)"
step "format ($(printf '%s\n' "$files" | grep -c . ) changed .py file(s) of this branch)"
if [ -z "$files" ]; then
    echo "  nothing to check"
elif require_tool ruff "format"; then
    # shellcheck disable=SC2086
    if run_tool ruff format --check $files; then
        echo "  formatted"
    else
        fail "ruff format --check (fix: ruff format $files)"
    fi
fi

# ---------------------------------------------------------------- 3. tests
step "tests"
if require_tool pytest "tests"; then
    run_tool pytest -q -p no:cacheprovider > /tmp/.gate-pytest.$$ 2>&1
    pytest_rc=$?
    if [ "$pytest_rc" -eq 0 ]; then
        tail -2 /tmp/.gate-pytest.$$ | sed 's/^/  /'
    elif [ "$pytest_rc" -eq 5 ]; then
        # Exit 5 is "no tests collected": a green suite of zero tests is the most
        # expensive failure mode there is, so it is a gate failure here.
        fail "pytest collected no tests — the gate would pass vacuously"
    else
        grep -E "^(FAILED|ERROR)|assert|Error" /tmp/.gate-pytest.$$ | head -30 | sed 's/^/  /'
        printf '  full output: pytest -q\n'
        fail "pytest (failures above)"
    fi
    rm -f /tmp/.gate-pytest.$$
fi

# ---------------------------------------------------------------- 4. types (only when configured)
step "types"
if grep -qE '^\[tool\.mypy\]|^\[mypy\]' "$PYPROJECT" 2>/dev/null || [ -f mypy.ini ] || [ -f .mypy.ini ]; then
    if require_tool mypy "types"; then
        if run_tool mypy . > /tmp/.gate-mypy.$$ 2>&1; then
            printf '  %s\n' "$(grep -E 'Success: no issues found' /tmp/.gate-mypy.$$ || echo 'clean')"
        else
            head -30 /tmp/.gate-mypy.$$
            fail "mypy (findings above)"
        fi
        rm -f /tmp/.gate-mypy.$$
    fi
else
    echo "  no [tool.mypy] section — add one when you want types as a gate"
fi

# ---------------------------------------------------------------- 5. clean environment
# Both branches answer the same question — "does the COMMITTED tree work somewhere that is
# not this dev environment?" — with whatever artifact the repo actually produces. The
# throwaway venv lives under mktemp -d with a trap: nothing is ever installed into the tree.
IDENTITY_PY=""        # the interpreter the identity stage asks; the fresh venv when there is one
CLEAN_TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate-clean-XXXXXX")"
trap 'rm -rf "$CLEAN_TMP"' EXIT INT TERM

if [ "$GATE_TIER" = "fast" ]; then
    step "clean environment (skipped)"
    skip "clean environment" "GATE_TIER=fast — creating and populating a venv is push-time work; pre-push runs the full gate"
    IDENTITY_PY="$(command -v python3 || true)"
    if [ -z "$IDENTITY_PY" ]; then
        fail "python3 not found — the identity stage has no interpreter to ask in fast mode"
    fi
elif command -v uv >/dev/null 2>&1; then
    uv venv --quiet "$CLEAN_TMP/venv" > "$CLEAN_TMP/venv.log" 2>&1 \
        || { tail -10 "$CLEAN_TMP/venv.log" | sed 's/^/  /'; fail "uv venv (the throwaway venv could not be created)"; }
    [ -x "$CLEAN_TMP/venv/bin/python" ] && IDENTITY_PY="$CLEAN_TMP/venv/bin/python"
else
    python3 -m venv "$CLEAN_TMP/venv" > "$CLEAN_TMP/venv.log" 2>&1 \
        || { tail -10 "$CLEAN_TMP/venv.log" | sed 's/^/  /'; fail "python3 -m venv (the throwaway venv could not be created)"; }
    [ -x "$CLEAN_TMP/venv/bin/python" ] && IDENTITY_PY="$CLEAN_TMP/venv/bin/python"
fi

if [ "$BRANCH" = "packaged" ]; then
    # ------------------------------------------------------------ 5a. build, install, smoke
    step "build (wheel + sdist) and smoke it from a throwaway venv"
    if command -v uv >/dev/null 2>&1; then
        uv build --out-dir "$CLEAN_TMP/dist" > "$CLEAN_TMP/build.log" 2>&1 \
            || { tail -15 "$CLEAN_TMP/build.log" | sed 's/^/  /'; fail "uv build (sdist+wheel must build from the committed tree)"; }
    else
        if python3 -m build --outdir "$CLEAN_TMP/dist" > "$CLEAN_TMP/build.log" 2>&1; then
            :
        else
            tail -15 "$CLEAN_TMP/build.log" | sed 's/^/  /'
            fail "python -m build (install 'build' or 'uv', or the distribution is not buildable)"
        fi
    fi
    if [ -n "$(ls -1 "$CLEAN_TMP/dist"/*.whl 2>/dev/null)" ]; then
        wheel="$(ls -1 "$CLEAN_TMP/dist"/*.whl | head -1)"
        printf '  built %s + %s\n' "$(basename "$wheel")" "$(basename "$(ls -1 "$CLEAN_TMP/dist"/*.tar.gz 2>/dev/null | head -1)" 2>/dev/null || echo 'sdist')"
        if command -v uv >/dev/null 2>&1; then
            uv pip install --quiet --python "$CLEAN_TMP/venv/bin/python" "$wheel" > "$CLEAN_TMP/install.log" 2>&1 \
                || { tail -15 "$CLEAN_TMP/install.log" | sed 's/^/  /'; fail "the built wheel does not install into a fresh venv"; }
        else
            "$CLEAN_TMP/venv/bin/python" -m pip install --quiet "$wheel" > "$CLEAN_TMP/install.log" 2>&1 \
                || { tail -15 "$CLEAN_TMP/install.log" | sed 's/^/  /'; fail "the built wheel does not install into a fresh venv"; }
        fi
        # The wheel is installed in that venv, so ask IT. Importing the source tree here
        # would hide exactly the packaging bug this stage exists to catch.
        PY_IMPORT_PATH=""
        import_name="$(ci_import_name || true)"
        if [ -z "$import_name" ]; then
            fail "cannot find the importable package (no src/<name>/__init__.py and no <name>/__init__.py); set the name or delete the smoke step"
        fi
        # A console script is what a user actually runs; prove it exists and answers.
        script_name="$(awk '/^\[project.scripts\]/{f=1;next} /^\[/{f=0} f && /^[A-Za-z0-9_.-]+ *=/{sub(/ *=.*/,"");print;exit}' "$PYPROJECT")"
        if [ -n "$script_name" ] && [ -x "$CLEAN_TMP/venv/bin/$script_name" ]; then
            printf '  %s --version -> %s\n' "$script_name" "$("$CLEAN_TMP/venv/bin/$script_name" --version 2>&1 | tail -1)"
        fi
    else
        fail "no wheel in the build output"
    fi
else
    # ------------------------------------------------------------ 5b. requirements + suite from a fresh venv
    # There is no wheel, so the artifact is the tree itself: what a clean machine gets is
    # this source plus exactly these dependencies. Install the declared list and run the
    # suite from THAT interpreter — never from the dev tree's environment, which is where
    # a dependency nobody declared is hiding.
    step "clean environment (throwaway venv + $REQUIREMENTS, then the suite from that venv)"
    if [ -n "$IDENTITY_PY" ]; then
        if command -v uv >/dev/null 2>&1; then
            uv pip install --quiet --python "$IDENTITY_PY" -r "$REQUIREMENTS" > "$CLEAN_TMP/install.log" 2>&1
        else
            "$IDENTITY_PY" -m pip install --quiet -r "$REQUIREMENTS" > "$CLEAN_TMP/install.log" 2>&1
        fi
        if [ $? -ne 0 ]; then
            tail -15 "$CLEAN_TMP/install.log" | sed 's/^/  /'
            fail "$REQUIREMENTS does not install into a fresh venv (a clean machine cannot reproduce this tree)"
        else
            # What the clean environment actually was. An unpinned list resolves to
            # whatever the index had today, so print the answer: that is the evidence a
            # "it passed here, not there" report needs.
            if command -v uv >/dev/null 2>&1; then
                frozen="$(uv pip freeze --python "$IDENTITY_PY" 2>/dev/null)"
            else
                frozen="$("$IDENTITY_PY" -m pip freeze 2>/dev/null)"
            fi
            while IFS= read -r req; do
                case "$req" in ''|\#*) continue ;; esac
                name="$(printf '%s' "$req" | sed 's/[<>=!~].*$//; s/[[:space:]]//g')"
                line="$(printf '%s\n' "$frozen" | grep -i "^${name}==" | head -1)"
                printf '  %s\n' "${line:-${name}: NOT INSTALLED (the declared entry did not resolve)}"
            done < "$REQUIREMENTS"

            if "$IDENTITY_PY" -c 'import pytest' > /dev/null 2>&1; then
                :
            else
                # The suite cannot run without the test runner. Install it, but say so: a
                # runner missing from the declared list is the "works on my machine" bug
                # this whole stage exists to find.
                printf '  NOTE: pytest is not in %s — installed for this run only; a clean machine would have no test runner\n' "$REQUIREMENTS"
                if command -v uv >/dev/null 2>&1; then
                    uv pip install --quiet --python "$IDENTITY_PY" pytest > /dev/null 2>&1
                else
                    "$IDENTITY_PY" -m pip install --quiet pytest > /dev/null 2>&1
                fi
            fi

            py_env "$IDENTITY_PY" -m pytest -q -p no:cacheprovider > "$CLEAN_TMP/pytest.log" 2>&1
            clean_rc=$?
            if [ "$clean_rc" -eq 0 ]; then
                tail -2 "$CLEAN_TMP/pytest.log" | sed 's/^/  /'
            elif [ "$clean_rc" -eq 5 ]; then
                fail "pytest collected no tests in the clean venv — the gate would pass vacuously"
            else
                grep -E "^(FAILED|ERROR)|assert|Error" "$CLEAN_TMP/pytest.log" | head -30 | sed 's/^/  /'
                printf '  full output: %s -m pytest -q   (in the throwaway venv)\n' "$IDENTITY_PY"
                fail "the suite does not pass in a fresh venv with only $REQUIREMENTS installed"
            fi
        fi
        # A CLI is what a user actually runs. If the package has a __main__, prove the
        # fresh venv can run it: a module-level import error shows up here and nowhere else.
        smoke_pkg="$(ci_tree_import_name || true)"
        if [ -n "$smoke_pkg" ] && [ -f "$PY_IMPORT_PATH/$smoke_pkg/__main__.py" ]; then
            if py_env "$IDENTITY_PY" -m "$smoke_pkg" --help > /dev/null 2>"$CLEAN_TMP/smoke.log"; then
                printf '  python -m %s --help answers in the throwaway venv\n' "$smoke_pkg"
            else
                tail -10 "$CLEAN_TMP/smoke.log" | sed 's/^/  /'
                fail "python -m $smoke_pkg --help fails in the throwaway venv"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------- 6. artifact identity
# Two copies of one number must agree. It is the cheapest possible guard against the most
# common silent failure: a green build that ships something stale.
step "artifact identity"
if [ "$BRANCH" = "packaged" ]; then
    declared_label="$PYPROJECT version"
    declared_value="$(ci_version)"
    import_name="${import_name:-$(ci_import_name || true)}"
    if [ -z "$import_name" ]; then
        fail "cannot find the importable package to ask for __version__"
    else
        [ -z "$PY_IMPORT_PATH" ] || PY_IMPORT_PATH="$(if [ -d "src/$import_name" ]; then printf 'src'; else printf '.'; fi)"
    fi
    identity_py_label="the code in the tree"
    [ -z "$PY_IMPORT_PATH" ] && identity_py_label="the installed wheel"
else
    PY_IMPORT_PATH="$(if [ -d src ]; then printf 'src'; else printf '.'; fi)"
    declared_label="$IDENTITY_FILE"
    if [ ! -f "$IDENTITY_FILE" ]; then
        fail "no $IDENTITY_FILE — a requirements-only repo has no packaging metadata to hold the version, so the declared copy must be a tracked file; create it holding the same number the code reports, or point IDENTITY_FILE at the file that already holds it"
        declared_value=""
    else
        declared_value="$(sed -n 's/^[[:space:]]*\([0-9][^[:space:]]*\)[[:space:]]*$/\1/p' "$IDENTITY_FILE" | head -1)"
        [ -n "$declared_value" ] || fail "$IDENTITY_FILE does not hold a version on a line of its own"
    fi
    import_name="$(ci_tree_import_name || true)"
    if [ -z "$import_name" ]; then
        fail "cannot find the importable package to ask for __version__ (set IDENTITY_IMPORT_NAME)"
    fi
    identity_py_label="the code the tree reports"
fi

if [ -n "$import_name" ] && [ -n "$IDENTITY_PY" ]; then
    reported="$(py_env "$IDENTITY_PY" -c "import $import_name as m; print(getattr(m, '__version__', 'NO __version__'))" 2>&1 | tail -1)"
    printf '  %s says %s, %s says %s\n' "$declared_label" "${declared_value:-<not found>}" "$identity_py_label" "$reported"
    if [ "$reported" = "NO __version__" ]; then
        fail "$import_name has no __version__ — the code cannot report which build it is"
    elif [ -z "$declared_value" ] || [ "$declared_value" != "$reported" ]; then
        fail "$declared_label and $identity_py_label disagree about the version (bump both from the same edit)"
    fi
fi

if [ "$status" -eq 0 ]; then
    printf '\nGATE PASSED%s\n' "${late_note:-}"
else
    printf '\nGATE FAILED - do not push this\n' >&2
fi

exit "$status"
