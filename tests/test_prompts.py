"""Tests for prompt template handling."""

import pytest

from docsum.prompts import render_prompt, get_builtin_prompt, BUILTIN_PROMPTS


class TestRenderPrompt:
    """Prompt template rendering with {text} substitution."""

    def test_simple_substitution(self):
        result = render_prompt("Summarize: {text}", "Hello world")
        assert result == "Summarize: Hello world"

    def test_no_placeholder(self):
        """Prompt without {text} should still work (text is appended)."""
        result = render_prompt("Just return OK", "Hello world")
        assert "Hello world" in result

    def test_empty_text(self):
        result = render_prompt("Summarize: {text}", "")
        assert result == "Summarize: "

    def test_multiple_placeholders(self):
        """If the prompt has multiple {text} placeholders, all are replaced."""
        result = render_prompt("Before: {text}\nAfter: {text}", "content")
        assert result == "Before: content\nAfter: content"

    def test_text_with_braces(self):
        """Text containing {braces} should not cause issues."""
        result = render_prompt("Summarize: {text}", "A {weird} sentence")
        assert "A {weird} sentence" in result


class TestBuiltinPrompts:
    """Built-in prompt templates."""

    def test_summary_prompt_exists(self):
        assert "summary" in BUILTIN_PROMPTS
        assert "{text}" in BUILTIN_PROMPTS["summary"]

    def test_themes_prompt_exists(self):
        assert "themes" in BUILTIN_PROMPTS
        assert "{text}" in BUILTIN_PROMPTS["themes"]

    def test_characters_prompt_exists(self):
        assert "characters" in BUILTIN_PROMPTS
        assert "{text}" in BUILTIN_PROMPTS["characters"]

    def test_key_events_prompt_exists(self):
        assert "key_events" in BUILTIN_PROMPTS
        assert "{text}" in BUILTIN_PROMPTS["key_events"]

    def test_get_builtin_prompt(self):
        prompt = get_builtin_prompt("summary")
        assert "{text}" in prompt

    def test_get_builtin_prompt_unknown(self):
        with pytest.raises(KeyError):
            get_builtin_prompt("nonexistent")

    def test_reduce_prompt_exists(self):
        """The reduce prompt combines intermediate results."""
        assert "reduce" in BUILTIN_PROMPTS
        assert "{summaries}" in BUILTIN_PROMPTS["reduce"]
