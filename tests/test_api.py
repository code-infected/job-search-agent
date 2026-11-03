from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config import Settings
from graph.nodes.fit_score import FitAssessment
from graph.nodes.tailor import TailoredDraft
from runner import run_batch
from tests.conftest import POSTING, FakeModel

LOCAL = "http://127.0.0.1:8000"
SCRIPT = {
    "FitAssessment": [FitAssessment(fit_score=0.9, fit_reasoning="Match")],
    "TailoredDraft": [
        TailoredDraft(
            message="I built REST APIs in Python and FastAPI backed by PostgreSQL.",
            cited_fact_ids=["exp-api"],
        )
    ],
}


def scripted(responses=None) -> FakeModel:
    return FakeModel({name: list(items) for name, items in (responses or SCRIPT).items()})


@pytest.fixture
def settings(tmp_path):
    base = Settings.from_env()
    return replace(
        base,
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        drafts_dir=tmp_path / "drafts",
        max_postings_per_run=5,
    )


@pytest.fixture
def client(settings, store, make_deps):
    app = create_app(settings, lambda _: make_deps(scripted(), [POSTING]), store)
    with TestClient(app, base_url=LOCAL) as test_client:
        yield test_client


def seed_review(client, make_deps) -> str:
    [paused] = run_batch(make_deps(scripted(), [POSTING]), client.app.state.checkpointer, 5)
    return paused["run_id"]


def test_health_and_page_are_served_with_security_headers(client):
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "192.168.1.5:8000"])
def test_requests_for_another_host_are_refused(client, host):
    """A DNS-rebinding page reaches this server under its own hostname."""
    assert client.get("/jobs", headers={"host": host}).status_code == 403


@pytest.mark.parametrize("origin", ["https://evil.example", "null"])
def test_cross_site_run_is_refused(client, origin):
    response = client.post("/runs", headers={"origin": origin})
    assert response.status_code == 403
    assert client.get("/status").json()["message"] == "No run yet."


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_schema_is_not_exposed(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/reviews", "/reviews/any", "/outcomes", "/dead-letters"])
def test_the_page_cannot_approve_or_change_decisions(client, path):
    assert client.post(path, json={"decision": "approved"}).status_code in (404, 405)


def test_page_never_renders_with_inner_html():
    source = Path("api/dashboard.html").read_text(encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent" in source


def test_run_reports_progress_and_its_jobs(client):
    response = client.post("/runs", headers={"origin": LOCAL})
    assert response.status_code == 202
    progress = client.get("/status").json()
    assert progress["running"] is False
    assert progress["checked"] == 1
    assert progress["drafted"] == 1
    assert progress["message"].startswith("Finished: 1 new postings checked, 1 drafts")

    [job] = client.get("/jobs").json()
    assert job["fit_score"] == 0.9
    assert job["fit_reasoning"] == "Match"
    assert job["status"] == "Waiting for review"


def test_a_failed_run_is_reported_and_does_not_block_the_next(settings, store, make_deps):
    calls = []

    def factory(_):
        calls.append(1)
        if len(calls) == 2:  # the first call is startup, the second is the run
            raise RuntimeError("facts file missing")
        return make_deps(scripted(), [])

    with TestClient(create_app(settings, factory, store), base_url=LOCAL) as test_client:
        assert test_client.post("/runs").status_code == 202
        progress = test_client.get("/status").json()
        assert progress["running"] is False
        assert progress["message"] == "Run failed: facts file missing"
        assert test_client.post("/runs").status_code == 202


def test_jobs_csv_downloads_every_posting(client, make_deps):
    seed_review(client, make_deps)
    response = client.get("/jobs.csv")
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "Waiting for review" in response.text
