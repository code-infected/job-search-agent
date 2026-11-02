import ast
from pathlib import Path

import pytest

from models import get_chat_model

ROOT = Path(__file__).parent.parent
PROVIDER_MODULES = (
    "openai",
    "anthropic",
    "google",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_ollama",
    "ollama",
)


@pytest.mark.parametrize(
    "provider, name, key, expected",
    [
        ("openai", "any-model", "OPENAI_API_KEY", "ChatOpenAI"),
        ("anthropic", "any-model", "ANTHROPIC_API_KEY", "ChatAnthropic"),
        ("google", "any-model", "GOOGLE_API_KEY", "ChatGoogleGenerativeAI"),
        ("ollama", "any-model", None, "ChatOllama"),
    ],
)
def test_provider_is_selected_by_environment_alone(monkeypatch, provider, name, key, expected):
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    monkeypatch.setenv("MODEL_NAME", name)
    if key:
        monkeypatch.setenv(key, "test-key")
    assert type(get_chat_model()).__name__ == expected


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_NAME", "any-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_chat_model()


def test_unknown_provider_and_missing_name_fail_loudly(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "unknown")
    monkeypatch.setenv("MODEL_NAME", "any-model")
    with pytest.raises(ValueError, match="MODEL_PROVIDER"):
        get_chat_model()
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_NAME", " ")
    with pytest.raises(ValueError, match="MODEL_NAME"):
        get_chat_model()


def test_no_provider_sdk_is_imported_outside_the_factory():
    offenders = []
    for package in ("graph", "memory", "evals", "api"):
        for path in (ROOT / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                offenders += [
                    f"{path}: {name}" for name in names if name.split(".")[0] in PROVIDER_MODULES
                ]
    assert offenders == []
