"""Provider-agnostic chat model factory. The only place a vendor is selected."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# MODEL_PROVIDER value -> (LangChain provider id, required API key variable)
PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openai": ("openai", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
    "google": ("google_genai", "GOOGLE_API_KEY"),
    "ollama": ("ollama", None),
}


def get_chat_model() -> BaseChatModel:
    """Build the chat model selected by MODEL_PROVIDER and MODEL_NAME."""
    provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    name = os.environ.get("MODEL_NAME", "").strip()
    if provider not in PROVIDERS:
        raise ValueError(f"MODEL_PROVIDER must be one of {sorted(PROVIDERS)}, got {provider!r}")
    if not name:
        raise ValueError("MODEL_NAME is not set")
    langchain_provider, key_variable = PROVIDERS[provider]
    if key_variable and not os.environ.get(key_variable):
        raise ValueError(f"{key_variable} is required when MODEL_PROVIDER={provider}")
    return init_chat_model(name, model_provider=langchain_provider)
