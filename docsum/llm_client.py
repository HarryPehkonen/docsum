"""LLM client for OpenAI-compatible API endpoints.

Wraps the OpenAI Python client to make simple completion calls.
Works with any OpenAI-compatible endpoint, including the Hermes proxy.
"""

from openai import OpenAI


class LLMClient:
    """A thin wrapper around the OpenAI client for chat completions.

    Args:
        base_url: API endpoint URL (e.g., http://127.0.0.1:8645/v1 for Hermes proxy).
        model: Model ID to use for completions.
        api_key: API key. For the Hermes proxy, any non-empty string works.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "proxy",
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> str:
        """Send a prompt to the LLM and return the response text.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt for role/instructions.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).
            max_tokens: Maximum tokens to generate in the response.

        Returns:
            The response text from the model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""
