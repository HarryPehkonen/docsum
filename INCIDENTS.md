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

## 2026-09-20 — `git status` held 398 untracked files, so a real stray file was invisible

What broke:        Every `git status` in this repo listed four scratch entries carrying
                   398 files: `work_all/` (366 files, 18 MB of per-episode state JSON and
                   logs from the TNG batch runs), `work_s3b/` (30 files), and the two
                   drivers `run_all_batch.sh` / `run_s3b_batch.sh` at the repo root. A
                   status output nobody can read is a status output nobody checks: the
                   one stray file that mattered — a source file written but never
                   `git add`ed — would have been invisible in the same list. The gate's
                   touched-file list also carried all 398 on every run.
Check added:       The scratch was MOVED (not deleted) out of the repo to
                   `~/hermes-workspace/docsum-scratch/`, with a README recording what it
                   is, why it was kept, and the one `mv` that puts it back. `.gitignore`
                   gained `work_*/` so a re-run of the batch drivers — which still write
                   into `$DOCSUM/work_all` — stays a named scratch directory instead of
                   `git status` noise. `git ls-files --others --exclude-standard` now
                   returns 0 in a clean tree.
Why it must stay:  The repo has no `tree` stage (the C++ gate's "every file is either
                   committed or ignored" rule), so untracked clutter has nothing to
                   catch it — `git status` IS the check here, and it only works if it is
                   short enough to read. If the `work_*/` line is deleted, the next batch
                   run silently refills the root and the check is gone again.

---

## 2026-09-20 — the `types` stage was a no-op while four mypy findings were known

What broke:        The gate's `types` stage only runs when the repo has a mypy config,
                   and this repo had none — so it printed "no [tool.mypy] section — add
                   one when you want types as a gate" on every run. Meanwhile `mypy` found
                   four real findings (`json_merge.py` splatting `list[Any | None]` into a
                   `*objects: dict` parameter; `llm_client.py` building the SDK call from
                   an untyped `kwargs` dict, twice). Worse, the verdict depended on which
                   environment answered: with a `.venv` mypy saw the real `openai` stub and
                   reported the `call-overload` errors, without one it reported
                   `import-not-found` for `openai` and `tiktoken` instead — and `tqdm` is
                   `import-untyped` in both.
Check added:       `mypy.ini` (the gate looks for `[tool.mypy]`, `mypy.ini` or
                   `.mypy.ini`) with `python_version = 3.11` pinned, plus
                   `ignore_missing_imports` for exactly the three libraries that ship no
                   types. The two code findings are fixed rather than silenced: the
                   `json_merge.py` splat filters as the neighbouring branches already did,
                   and `kwargs` is annotated `dict[str, Any]`. Verified green in BOTH
                   environments the gate can find itself in —
                   `uv run --with mypy --with pytest python -m mypy .` and the same with
                   `--with-requirements requirements.txt`: "Success: no issues found in 23
                   source files".
Why it must stay:  A stage that prints a sentence instead of checking something is worse
                   than no stage, because the run still ends in `GATE PASSED` and a reader
                   counts it as covered. The per-module overrides are what keep the stage
                   green on a clean machine without installing the dependencies — deleting
                   them does not make the code more correct, it makes the gate red for an
                   environment fact, and a gate that goes red for environmental reasons is
                   a gate that gets muted. The default rule set is deliberate: `--strict`
                   is a separate decision with its own debt, and this file says so.

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
Check added:       FIXED (2026-09-20, card t_90ed3a35) by implementing the flag rather
                   than removing it: `map_reduce`, `refine`, `hierarchical` and
                   `_recursive_reduce` now take `stream: bool = False` and pass it to
                   every `client.complete(...)`, and `_cmd_run` passes `args.stream`
                   in — so the flag means the same thing in `run` and `prepare`. The
                   check that keeps it honest is
                   `tests/test_streaming.py::TestStreamFlagReachesTheApiCall`: six
                   tests assert that every `client.complete` call the algorithms make
                   carries `stream=True` with the flag, `stream=False` without it, and
                   that `docsum run --stream` reaches the client at all (a CLI-level
                   run with a stubbed `LLMClient`). Both subcommands now take their
                   `--help` text from one `_STREAM_HELP` constant, so they cannot
                   drift apart again. Measured live against the Hermes proxy:
                   `--stream` and no-flag runs both return a summary.
Why it must stay:  An accepted-and-ignored flag is the worst kind of interface lie: the
                   caller believes the connection is being kept alive against a gateway
                   timeout while the request is exactly as interruptible as before. The
                   test is what stops the wiring from being dropped again by a refactor
                   — without it, "the algorithms take no stream parameter" is invisible
                   in a green suite, which is how it survived the first time.

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

## Known debt (accepted on purpose, not incidents)

Recorded here because these were left in place deliberately, and a future reader should
be able to tell a decision from an oversight. Status as of 2026-09-20 (card t_90ed3a35).

### Cleared on 2026-09-20

- **The shipped code (`docsum/`) has no findings under the adopted rule set.** Before this
  card the gate tolerated `F401`/`F841`/`F811` tree-wide (25 unused imports, 7 unused
  locals, 2 redefinitions) — one of them was the `--stream` bug above. `docsum/` is now
  clean: `uv run --with ruff ruff check --isolated --select E4,E7,E9,F docsum` →
  "All checks passed!".
- **`types` is a real stage now.** `mypy.ini` exists, the four findings are fixed rather
  than silenced, and the stage is green in both environments the gate can find itself in
  (incident above).
- **Untracked clutter: 398 → 0 files.** The batch scratch moved to
  `~/hermes-workspace/docsum-scratch/`, `work_*/` is ignored, and `git ls-files --others
  --exclude-standard` returns nothing in a clean tree (incident above).
- **The format debt shrank from 14 files to 10.** `docsum/cli.py`, `docsum/algorithms.py`,
  `docsum/step_processor.py` and `tests/test_streaming.py` are now formatted. This is not
  a free win: the gate's format stage requires every file a branch TOUCHES to be
  formatted, so editing a drifting file costs its whole-file reformat in that commit.

### Still open (each one is a decision, not an oversight)

- **24 ruff findings remain, all in seven TEST files**, tolerated per file in
  `ruff.toml`'s `[lint.per-file-ignores]` rather than globally — a new test file gets the
  full rule set. Breakdown: 18 unused imports (`F401`), 4 unused locals (`F841`),
  2 redefinitions (`F811`). Per file, with the format drift that clearing it would drag
  in: `test_opus5_bugs.py` 13 findings / 29 lines · `test_fixes.py` 5 / 166 ·
  `test_algorithms.py` 2 / 30 · `test_step_processor.py` 1 / 78 ·
  `test_integration.py` 1 / 107 · `test_cli.py` 1 / 252 · `test_chunker.py` 1 / 8.
  Every one of those files is also unformatted, so deleting a one-line unused import makes
  the gate demand a whole-file reformat: 670 lines of churn unrelated to the finding. This
  card was told "do not mass-reformat", so the seven are deferred as ONE decision (format
  the tree, or leave it) instead of being paid in arbitrary fragments. The two commands
  are in the comment above that block in `ruff.toml`.
- **10 files would still be reformatted** (`uv run --with ruff ruff format --check .`).
  The gate checks only the files a branch touches, so this debt costs nothing until
  someone edits one of them — and then it costs that file's reformat, in that commit.
  `tests/test_json_merge.py` is in the list and has no lint finding at all.
- **Rules this repo has never adopted** stay off on purpose: `I001` (12 unsorted import
  blocks — 14 before this card's edits), `UP045`/`UP035` (5 + 1 `Optional[X]` that could be
  `X | None`), `EXE001` (2 shebangs without the exec bit), `TRY002`, `SIM114`. Measured
  with the no-config command the old baseline used, `uv run --with ruff ruff check
  --isolated .`: **58 findings before this card, 46 after**. Switching one of these rules
  on the way the others were switched off: fix the findings, add the rule to `select`, same
  commit.
- **`requirements.txt` is unpinned** (`pytest`, `pytest-mock`, `openai`, `tiktoken`,
  `tqdm` — no version specifiers). The gate's clean-environment stage resolves whatever the
  index has today and prints what it got; on 2026-09-20 that was `pytest 9.1.1`,
  `pytest-mock 3.15.1`, `openai 3.16.2`, `tiktoken 0.14.0`, `tqdm 4.70.1`. Deliberately not
  tightened: there is still no working `.venv` in the repo holding the versions this tree is
  known good against, and pinning to today's newest would be an upgrade disguised as a pin.
  The follow-up is a dev lock or explicit pins, decided by whoever knows which versions the
  batch runs used — one commit.

