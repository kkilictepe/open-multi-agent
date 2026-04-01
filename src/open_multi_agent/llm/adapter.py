"""LLM adapter factory."""

from __future__ import annotations

from ..types import LLMAdapter, SupportedProvider


def create_adapter(provider: SupportedProvider, api_key: str | None = None) -> LLMAdapter:
    """Instantiate the appropriate LLMAdapter for the given provider.

    Adapters are imported lazily so that projects using only one provider
    are not forced to install the SDK for the other.
    """
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(api_key)  # type: ignore[return-value]
    elif provider == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(api_key)  # type: ignore[return-value]
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
