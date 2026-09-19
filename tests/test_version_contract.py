"""Contract test for the artifact-identity invariant.

Two copies of one number must agree: the VERSION file at the repo root (the declared
copy — what a release or install script reads) and ``docsum.__version__`` (the runtime
copy — what the code reports). The gate's `artifact identity` stage checks the same pair
at gate time; this test makes the suite check it too, so a desynchronised bump fails in
the place a developer is already running rather than at push time.
"""

from pathlib import Path

import docsum

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_file_and_package_version_agree():
    """A one-sided bump fails here, and in the gate's artifact-identity stage."""
    declared = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert declared, "VERSION is empty — it holds the declared copy of the version"
    assert declared == docsum.__version__, (
        f"VERSION says {declared!r} but docsum.__version__ says "
        f"{docsum.__version__!r} — bump both in one edit"
    )
