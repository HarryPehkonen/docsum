"""Prompt templates for document summarization.

Built-in prompts for common summarization tasks, plus a render function
that substitutes text into a template containing {text} or {summaries} placeholders.
"""

BUILTIN_PROMPTS: dict[str, str] = {
    "summary": (
        "Summarize the following text, preserving the key ideas, important details, "
        "and overall structure. Be concise but comprehensive.\n\n"
        "---\n\n"
        "{text}"
    ),
    "themes": (
        "Identify the main themes, recurring motifs, and central ideas in the following text. "
        "List each theme with a brief explanation of how it appears in the text.\n\n"
        "---\n\n"
        "{text}"
    ),
    "characters": (
        "Extract all characters, entities, or key figures mentioned in the following text. "
        "For each, note their role, key actions, and relationships to other characters.\n\n"
        "---\n\n"
        "{text}"
    ),
    "key_events": (
        "List the key events, plot points, and important moments from the following text, "
        "in the order they occur. Be brief but specific.\n\n"
        "---\n\n"
        "{text}"
    ),
    "reduce": (
        "The following are analyses of different sections of a larger document. "
        "Combine them into a single coherent result that captures the overall themes, "
        "key points, and important details. Remove redundancies and resolve any "
        "contradictions.\n\n"
        "---\n\n"
        "{summaries}"
    ),
}


def get_builtin_prompt(name: str) -> str:
    """Return a built-in prompt template by name.

    Args:
        name: One of 'summary', 'themes', 'characters', 'key_events', 'reduce'.

    Returns:
        The prompt template string containing {text} or {summaries}.

    Raises:
        KeyError: If the name is not a recognized built-in prompt.
    """
    return BUILTIN_PROMPTS[name]


def render_prompt(template: str, text: str) -> str:
    """Render a prompt template by substituting text.

    If the template contains {text}, it is replaced with the given text.
    If the template does not contain {text}, the text is appended.

    Args:
        template: The prompt template string.
        text: The text to insert.

    Returns:
        The rendered prompt with text substituted.
    """
    if "{text}" in template:
        return template.replace("{text}", text)
    else:
        return f"{template}\n\n{text}"


def render_reduce_prompt(template: str, summaries: list[str]) -> str:
    """Render a reduce prompt by combining intermediate summaries.

    Args:
        template: The reduce prompt template containing {summaries}.
        summaries: List of intermediate summary strings.

    Returns:
        The rendered reduce prompt.
    """
    combined = "\n\n---\n\n".join(summaries)
    if "{summaries}" in template:
        return template.replace("{summaries}", combined)
    else:
        return f"{template}\n\n{combined}"
