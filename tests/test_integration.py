"""Integration tests that call a real LLM via the Hermes proxy.

These are marked with @pytest.mark.integration and excluded from normal runs.
Run them explicitly:

    pytest -m integration --base-url http://127.0.0.1:8645/v1

They use a small/cheap model (Nemotron) to verify the full chain works.
"""

import os
import tempfile

import pytest

from docsum.cli import main
from docsum.llm_client import LLMClient

# Skip all integration tests unless a base URL is available
BASE_URL = os.environ.get("DOCSUM_TEST_BASE_URL", "http://127.0.0.1:8645/v1")
TEST_MODEL = os.environ.get("DOCSUM_TEST_MODEL", "nvidia/nemotron-3-super-120b-a12b")

pytestmark = pytest.mark.integration


@pytest.fixture
def temp_text_file():
    """Create a temporary text file with enough content for multi-chunk processing."""
    content = (
        "The bridge of the Enterprise was quiet. Captain Picard stared at the viewscreen. "
        "Data reported that the anomaly was growing. Worf prepared for combat. "
        "The crew worked together to restore power and jumped to safety."
    ) * 5  # Enough for 2-3 chunks with small max_tokens

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


class TestIntegrationMapReduce:
    """End-to-end map-reduce with a real LLM."""

    def test_produces_summary(self, temp_text_file):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
            out_path = out.name
        try:
            main([
                "--input", temp_text_file,
                "--output", out_path,
                "--model", TEST_MODEL,
                "--base-url", BASE_URL,
                "--mode", "map-reduce",
                "--max-tokens", "300",
            ])
            with open(out_path) as f:
                result = f.read()
            assert len(result) > 0
            assert "error" not in result.lower()
        finally:
            os.unlink(out_path)


class TestIntegrationRefine:
    """End-to-end refine with a real LLM."""

    def test_produces_summary(self, temp_text_file):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
            out_path = out.name
        try:
            main([
                "--input", temp_text_file,
                "--output", out_path,
                "--model", TEST_MODEL,
                "--base-url", BASE_URL,
                "--mode", "refine",
                "--max-tokens", "300",
            ])
            with open(out_path) as f:
                result = f.read()
            assert len(result) > 0
        finally:
            os.unlink(out_path)


class TestIntegrationHierarchical:
    """End-to-end hierarchical with a real LLM."""

    def test_produces_summary(self, temp_text_file):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
            out_path = out.name
        try:
            main([
                "--input", temp_text_file,
                "--output", out_path,
                "--model", TEST_MODEL,
                "--base-url", BASE_URL,
                "--mode", "hierarchical",
                "--max-tokens", "300",
            ])
            with open(out_path) as f:
                result = f.read()
            assert len(result) > 0
        finally:
            os.unlink(out_path)


class TestIntegrationThemes:
    """End-to-end theme extraction with a real LLM."""

    def test_extracts_themes(self, temp_text_file):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
            out_path = out.name
        try:
            main([
                "--input", temp_text_file,
                "--output", out_path,
                "--model", TEST_MODEL,
                "--base-url", BASE_URL,
                "--mode", "map-reduce",
                "--prompt", "themes",
                "--max-tokens", "300",
            ])
            with open(out_path) as f:
                result = f.read()
            assert len(result) > 0
        finally:
            os.unlink(out_path)
