"""LLM client for OpenAI-compatible API endpoints.

Wraps the OpenAI Python client to make simple completion calls.
Works with any OpenAI-compatible endpoint, including the Hermes proxy.

Supports:
- Non-streaming mode (default): single response
- Streaming mode: tokens flow incrementally (keeps connection active,
  avoids Cloudflare 524 timeouts on slow models)
- max_tokens=None: omit the max_tokens parameter entirely (lets the
  model use its default maximum — may help or hurt with timeouts)
"""

from typing import Optional

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
        max_tokens: Optional[int] = 8192,
        stream: bool = False,
    ) -> str:
        """Send a prompt to the LLM and return the response text.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt for role/instructions.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).
            max_tokens: Maximum tokens to generate in the response.
                Set to None to omit the parameter entirely (use model default).
            stream: If True, use streaming mode — tokens arrive incrementally,
                keeping the connection active and avoiding idle-timeout 524s.

        Returns:
            The response text from the model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build kwargs — omit max_tokens if None
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if stream:
            kwargs["stream"] = True
            response = self._client.chat.completions.create(**kwargs)
            # Collect streaming chunks
            result_parts: list[str] = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    result_parts.append(chunk.choices[0].delta.content)
            return "".join(result_parts)
        else:
            response = self._client.chat.completions.create(**kwargs)
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return ""
