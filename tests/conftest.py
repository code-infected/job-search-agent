import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from graph.build import Dependencies, open_checkpointer
from graph.facts import load_facts
from graph.llm import Budget
from graph.nodes.discover import Posting
from graph.nodes.send import FileDraftSink
from memory.store import MemoryStore

ROOT = Path(__file__).parent.parent


class HashEmbedding(EmbeddingFunction[Documents]):
    """Bag-of-words embedding so tests need no model download."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = [0.0] * 64
            for word in text.lower().split():
                vector[hashlib.sha256(word.encode()).digest()[0] % 64] += 1.0
            vectors.append(vector)
        return vectors

    @staticmethod
    def name() -> str:
        return "test-hash"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()


class _Structured:
    def __init__(self, model: "FakeModel", schema: type) -> None:
        self.model = model
        self.schema = schema

    def invoke(self, messages: list) -> dict:
        self.model.calls.append(messages)
        response = self.model.responses[self.schema.__name__].pop(0)
        if isinstance(response, Exception):
            raise response
        raw = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50})
        if isinstance(response, dict):  # a malformed reply: rejected by the real schema
            try:
                parsed = self.schema.model_validate(response)
            except ValueError as exc:
                return {"raw": raw, "parsed": None, "parsing_error": exc}
            return {"raw": raw, "parsed": parsed, "parsing_error": None}
        return {"raw": raw, "parsed": response, "parsing_error": None}


class FakeModel:
    """Scripted chat model. `responses` maps a schema name to its queued replies."""

    def __init__(self, responses: dict[str, list]) -> None:
        self.responses = responses
        self.calls: list[list] = []

    def with_structured_output(self, schema: type, include_raw: bool = False) -> _Structured:
        assert include_raw
        return _Structured(self, schema)


@pytest.fixture
def facts():
    return load_facts(ROOT / "config" / "facts.example.json")


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "chroma", embedding_function=HashEmbedding())


@pytest.fixture
def checkpointer(tmp_path):
    saver = open_checkpointer(tmp_path / "checkpoints.sqlite")
    yield saver
    saver.conn.close()


@pytest.fixture
def budget():
    return Budget(max_tokens=1_000_000, max_cost_usd=100.0)


@pytest.fixture
def make_deps(store, facts, budget, tmp_path):
    def build(model, postings=(), threshold=0.7) -> Dependencies:
        return Dependencies(
            model=model,
            store=store,
            facts=facts,
            target_profile="Backend roles",
            fit_score_threshold=threshold,
            sink=FileDraftSink(tmp_path / "drafts"),
            budget=budget,
            postings=iter(postings),
        )

    return build


POSTING = Posting(
    "https://example.com/jobs/backend",
    "Backend Engineer at Acme. Python, FastAPI, Kubernetes. Five years of Kubernetes required.",
)
