# INCIDENTS — every real failure, and the check that now catches it

Newest first. One entry per incident that changed how this repo works.

The rule: **when something breaks, the fix is not done until a check exists that would
have caught it, and the incident is written here next to that check.** A check with no
written rationale looks arbitrary to the next hurried contributor (or agent), and
arbitrary checks get deleted. The rationale is the load-bearing part.

    ## YYYY-MM-DD — <one-line failure>
    What broke:        <the user-visible symptom>
    Check added:       <file> + <gate stage that now catches it>
    Why it must stay:  <why deleting this check re-enables the bug>

---

## 2026-09-19 — `--stream` is accepted and silently ignored by the monolithic path

What broke:        `docsum --input FILE --model M --stream` parses the flag and does
                   nothing with it. `docsum/cli.py:141` advertises it ("use streaming mode
                   (keeps connection active, helps avoid 524 timeouts)"), `_cmd_run` reads
                   it into `run_stream` at `cli.py:264`, that local is never used, and
                   `map_reduce` / `refine` / `hierarchical` take no `stream` parameter.
                   The step path does honour it (`_cmd_prepare` passes `args.stream` into
                   `step_prepare`, `cli.py:338`), so the flag works or not depending on
                   which subcommand you are in — a user cannot tell from the help text.
Check added:       Nothing yet — this one is REPORTED, NOT FIXED. The wiring card was
                   tooling-only ("do not change docsum's behaviour"), and wiring the flag
                   through the monolithic algorithms is a behaviour change with its own
                   review. The check that exists today is `ruff.toml`'s explicit `F841`
                   entry: the assignment is one of the seven unused-local findings that
                   would otherwise be lost in a rule that is switched on by default.
Why it must stay:  An accepted-and-ignored flag is the worst kind of interface lie: the
                   caller believes the connection is being kept alive against a gateway
                   timeout while the request is exactly as interruptible as before. When
                   the decision on this flag is finally taken (implement it, or remove it
                   from the parser and the README), delete the `F841` entry in
                   `ruff.toml` and this section in the same commit.

---

## 2026-09-19 — the test stage ran pytest in an environment holding none of the project's dependencies

What broke:        On a machine with no `.venv` (a fresh clone, a CI-like host), the gate's
                   tool ladder falls back to `uv run --with pytest`, which builds an
                   environment containing pytest and nothing else. This repo declares its
                   dependencies in `requirements.txt` and has no `pyproject.toml` for
                   `uv run` to sync, so the tests stage reported ten failures of the shape
                   `ModuleNotFoundError: No module named 'tiktoken'` / `'openai'` for a
                   tree whose suite passes 132 tests. A red gate that says nothing about
                   the code is worse than no gate: it teaches the reader to skim it.
Check added:       `scripts/gate.sh` — the uv fallback for pytest adds
                   `--with-requirements requirements.txt` when the repo declares one
                   (`UV_WITH_REQUIREMENTS`), and the new `tools` stage prints which rung of
                   the ladder answered (`uv run --with pytest --with-requirements
                   requirements.txt — fetched on demand; this repo does not pin them`).
Why it must stay:  The whole value of the gate is that it is the thing you can trust on a
                   machine that is not yours. A fallback that cannot import the project
                   makes every run on such a machine noisy, and noise is what gets a gate
                   muted — after which the real failure (a dependency that is imported but
                   never declared) arrives unnoticed.

---

## 2026-09-19 — nothing checked the lint, the format, or the two copies of the version

What broke:        The tree had no lint config and no gate. `ruff check .` under the rules
                   the installed ruff happened to enable reported 58 findings on HEAD
                   (25 F401, 14 I001, 7 F841, 5 UP045, 2 F811, 2 EXE001, plus one each of
                   UP035, TRY002, SIM114) and 14 of 22 Python files were unformatted. Worse,
                   because the gate fetches its tools (`uv run --with ruff`), the rule set
                   was a property of the machine's ruff release rather than of the repo:
                   a release that adds a rule would have turned the gate red with no code
                   change, and no one would have known why.
Check added:       `ruff.toml` states the rule set explicitly (ruff's historical core:
                   `E4`, `E7`, `E9`, `F`) and names the three rules this tree currently
                   violates (`F401`, `F811`, `F841`) one at a time, so re-enabling them is
                   a visible edit. `scripts/gate.sh` runs `ruff check .` over the tree
                   (`lint` stage) and `ruff format --check` over the files the branch
                   touched (`format` stage), and the `tools` stage prints which ruff
                   answered.
Why it must stay:  The explicit rule list is what stops the verdict depending on the
                   calendar; the per-rule comments are what stop someone re-enabling a rule
                   the tree cannot yet satisfy (or silently keeping one off after the debt
                   is paid). Deleting the `frozen` rule list returns the gate to
                   "whatever ruff thinks this month", which is not a gate.

---

## 2026-09-19 — a new file's formatting was never checked

What broke:        A source file written and committed during a working session failed the
                   format gate the next morning — the drift had been in the commit the
                   whole time. `git diff` cannot see a file that is untracked, so the
                   format stage looked at an empty file list for exactly the files that
                   were newest, and passed.
Check added:       `scripts/gate.sh` format stage: the touched-file list is built from
                   `git diff` PLUS `git ls-files --others --exclude-standard`, so a
                   brand-new file is checked the moment it exists, not after it has already
                   been committed. (Same fix as the Deno and C++ gates in the kit this gate
                   came from — the trap is language-independent.)
Why it must stay:  Removing the `git ls-files --others` line makes the gate blind to
                   exactly the files a contributor just wrote — the ones most likely to be
                   wrong. It passed on the old list, so nothing else will notice.

---

## Known lint debt (accepted findings, not incidents)

Recorded here because these were left in place deliberately, and a future reader should
be able to tell a decision from an oversight.

- **58 ruff findings under the rules this repo has not adopted.** Reproduce with
  `uv run --with ruff ruff check --isolated .` (`--isolated` ignores `ruff.toml`, so this
  is the rule set ruff enables with no config at all — the exact state this repo was in
  before the gate). Breakdown: 25 unused imports
  (`F401`), 14 unformatted/sorted import blocks (`I001`), 7 unused locals (`F841`),
  5 `Optional[X]` that could be `X | None` (`UP045`), 2 redefinitions in
  `tests/test_opus5_bugs.py` (`F811`), 2 shebang-without-exec-bit (`EXE001`), plus
  `UP035`, `TRY002`, `SIM114`. Fixing them edits application and test code, which this
  card's scope ("tooling only, do not change behaviour") excludes. One of them is a
  possible real bug and is an incident above (`F841` / `--stream`).
- **14 files would be reformatted** (`uv run --with ruff ruff format --check .` prints
  exactly that). The tree has never been formatted as a whole, and the gate checks only
  the files a branch touches, so this debt stays until someone decides to spend one commit
  on it. That is the kit's deliberate rule, not an oversight.
- **`requirements.txt` is unpinned** (`pytest`, `pytest-mock`, `openai`, `tiktoken`,
  `tqdm` — no version specifiers). The gate's clean-environment stage resolves whatever the
  index has today and prints what it got; at the time of writing that was
  `pytest 9.1.1`, `pytest-mock 3.15.1`, `openai 3.16.2`, `tiktoken 0.14.0`, `tqdm 4.70.1`.
  Deliberately not tightened here: there was no working `.venv` in the repo to read the
  versions this tree is known good against, and pinning to today's newest would be an
  upgrade disguised as a pin. Next step is a dev lock or explicit pins, decided by whoever
  knows which versions the `work_all/` runs used.
- **types are not a gate.** `mypy` is not configured, and running it finds 4 errors
  (2 `call-overload` in `llm_client.py` for the `kwargs` dict passed to `create`,
  1 `import-untyped` for `tqdm`, 1 in `tests/`). Configuring `mypy` in `ruff.toml`/`mypy.ini`
  would turn those into a failing stage; the wiring card left them as a follow-up instead
  of forcing a red gate on day one.
- **398 untracked scratch files** live in `work_all/` and `work_s3b/` (run logs and state
  files). Nothing there is committed, and no `.py` file lives there, so the gate ignores
  them — but they appear in every `git status` and in the gate's touched-file list.
  Ignoring or deleting them is a repo-hygiene decision, not a gate decision.
