from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config import Settings
from graph.nodes.fit_score import FitAssessment
from graph.nodes.tailor import TailoredDraft
from runner import run_batch
from tests.conftest import POSTING, FakeModel

TOKEN = "test-token-long-enough"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SCRIPT = {
    "FitAssessment": [FitAssessment(fit_score=0.9, fit_reasoning="Match")],
    "TailoredDraft": [
        TailoredDraft(
            message="I built REST APIs in Python and FastAPI backed by PostgreSQL.",
            cited_fact_ids=["exp-api"],
        )
    ],
}


@pytest.fixture
def settings(tmp_path):
    base = Settings.from_env()
    return replace(
        base,
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        drafts_dir=tmp_path / "drafts",
        max_postings_per_run=5,
        review_api_token=TOKEN,
    )


@pytest.fixture
def client(settings, store, make_deps):
    app = create_app(settings, lambda _: make_deps(FakeModel({})), store)
    with TestClient(app) as test_client:
        yield test_client


def seed_review(client, make_deps) -> str:
    responses = {name: list(items) for name, items in SCRIPT.items()}
    deps = make_deps(FakeModel(responses), [POSTING])
    [paused] = run_batch(deps, client.app.state.checkpointer, 5)
    return paused["run_id"]


def test_app_refuses_to_start_without_a_strong_token(settings):
    with pytest.raises(RuntimeError, match="REVIEW_API_TOKEN"):
        create_app(replace(settings, review_api_token="short"))


@pytest.mark.parametrize(
    "method, path",
    [
        ("get", "/reviews"),
        ("get", "/reviews/digest"),
        ("post", "/reviews/any"),
        ("post", "/runs"),
        ("post", "/outcomes"),
        ("get", "/dead-letters"),
        ("post", "/dead-letters/release"),
    ],
)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_every_data_route_requires_the_token(client, method, path, headers):
    assert getattr(client, method)(path, headers=headers).status_code == 401


def test_health_and_dashboard_are_open_and_carry_no_data(client):
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_schema_is_not_exposed(client, path):
    assert client.get(path).status_code == 404


def test_release_of_a_non_dead_letter_is_not_found(client):
    body = {"url": "https://example.com/x"}
    assert client.post("/dead-letters/release", json=body, headers=AUTH).status_code == 404


def test_dashboard_never_renders_with_inner_html():
    source = Path("api/dashboard.html").read_text(encoding="utf-8")
    assert "innerHTML =" not in source
    assert "textContent" in source


def test_review_lifecycle_over_http(client, make_deps, tmp_path):
    run_id = seed_review(client, make_deps)
    [review] = client.get("/reviews", headers=AUTH).json()
    assert review["run_id"] == run_id
    assert review["payload"]["fit_reasoning"] == "Match"

    response = client.post(f"/reviews/{run_id}", json={"decision": "approved"}, headers=AUTH)
    assert response.json() == {"run_id": run_id, "approval_status": "approved"}
    assert (tmp_path / "drafts" / f"{run_id}.md").exists()
    assert client.get("/reviews", headers=AUTH).json() == []


def test_a_decision_cannot_be_applied_twice(client, make_deps):
    run_id = seed_review(client, make_deps)
    body = {"decision": "approved"}
    assert client.post(f"/reviews/{run_id}", json=body, headers=AUTH).status_code == 200
    assert client.post(f"/reviews/{run_id}", json=body, headers=AUTH).status_code == 409


def test_unknown_run_is_a_conflict(client):
    response = client.post("/reviews/nope", json={"decision": "approved"}, headers=AUTH)
    assert response.status_code == 409


def test_invalid_decisions_leave_the_review_pending(client, make_deps):
    run_id = seed_review(client, make_deps)
    for body in ({"decision": "rejected"}, {"decision": "auto"}, {"approved": True}):
        assert client.post(f"/reviews/{run_id}", json=body, headers=AUTH).status_code == 422
    assert len(client.get("/reviews", headers=AUTH).json()) == 1


def test_digest_counts_pending_reviews(client, make_deps):
    seed_review(client, make_deps)
    digest = client.get("/reviews/digest", headers=AUTH).json()
    assert digest == {"pending": 1, "stale": []}


def test_outcome_for_unknown_url_is_not_found(client):
    body = {"url": "https://example.com/x", "outcome": "interview"}
    assert client.post("/outcomes", json=body, headers=AUTH).status_code == 404


def test_dead_letter_can_be_released_for_retry(client, make_deps):
    with patch("graph.llm.time.sleep"):
        model = FakeModel({"FitAssessment": [RuntimeError("down")] * 3})
        run_batch(make_deps(model, [POSTING]), client.app.state.checkpointer, 5)
    [record] = client.get("/dead-letters", headers=AUTH).json()
    assert "down" in record["error"]
    response = client.post("/dead-letters/release", json={"url": record["url"]}, headers=AUTH)
    assert response.status_code == 200
    assert client.get("/dead-letters", headers=AUTH).json() == []


def test_run_endpoint_starts_a_batch(client):
    assert client.post("/runs", headers=AUTH).status_code == 202
