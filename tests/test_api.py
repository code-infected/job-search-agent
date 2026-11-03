import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from boards import BoardMatch
from config import Settings
from graph.llm import Budget
from graph.nodes.fit_score import FitAssessment
from graph.nodes.tailor import TailoredDraft
from memory.schema import DecisionRecord
from resume import SuggestedFact, SuggestedFacts
from runner import run_batch
from tests.conftest import POSTING, ROOT, FakeModel

LOCAL = "http://127.0.0.1:8000"
CROSS_SITE = {"origin": "https://evil.example"}
SCRIPT = {
    "FitAssessment": [FitAssessment(fit_score=0.9, fit_reasoning="Match")],
    "TailoredDraft": [
        TailoredDraft(
            message="I built REST APIs in Python and FastAPI backed by PostgreSQL.",
            cited_fact_ids=["exp-api"],
        )
    ],
}
SEARCH = {
    "target_profile": "Backend roles",
    "exclude_keywords": ["senior"],
    "queries": [{"source": "greenhouse", "board": "example-board"}],
}


def scripted(responses=None) -> FakeModel:
    return FakeModel({name: list(items) for name, items in (responses or SCRIPT).items()})


@pytest.fixture(autouse=True)
def restore_environment():
    """Setup routes write settings into the process environment."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    facts = tmp_path / "facts.json"
    shutil.copy(ROOT / "config" / "facts.example.json", facts)
    search = tmp_path / "search.json"
    search.write_text(json.dumps(SEARCH), encoding="utf-8")
    return replace(
        Settings.from_env(),
        vector_store_path=tmp_path / "chroma",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        drafts_dir=tmp_path / "drafts",
        facts_path=facts,
        search_config_path=search,
        max_postings_per_run=5,
        auto_run_hours=0,
    )


@pytest.fixture
def env_file(tmp_path) -> Path:
    path = tmp_path / ".env"
    path.write_text("# kept\nMODEL_PROVIDER=anthropic\n", encoding="utf-8")
    return path


def make_client(settings, store, factory, env_file):
    app = create_app(settings, factory, store, env_path=env_file, scheduler=False)
    return TestClient(app, base_url=LOCAL)


@pytest.fixture
def client(settings, store, make_deps, env_file):
    with make_client(settings, store, lambda _: make_deps(scripted(), [POSTING]), env_file) as test:
        yield test


def finish_run(client) -> dict:
    client.app.state.batch_thread.join(timeout=30)
    return client.get("/status").json()


def seed_review(client, make_deps) -> str:
    [paused] = run_batch(make_deps(scripted(), [POSTING]), client.app.state.checkpointer, 5)
    return paused["run_id"]


# Serving and the local-only guard


def test_pages_are_served_with_security_headers(client):
    assert client.get("/health").json() == {"status": "ok"}
    for path in ("/", "/setup"):
        page = client.get(path)
        assert page.status_code == 200
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "192.168.1.5:8000"])
def test_requests_for_another_host_are_refused(client, host):
    """A DNS-rebinding page reaches this server under its own hostname."""
    for path in ("/jobs", "/setup/state"):
        assert client.get(path, headers={"host": host}).status_code == 403


@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/runs"),
        ("put", "/setup/facts"),
        ("put", "/setup/search"),
        ("post", "/setup/model"),
        ("put", "/setup/preferences"),
        ("post", "/setup/rescore"),
    ],
)
def test_cross_site_changes_are_refused(client, method, path):
    assert getattr(client, method)(path, headers=CROSS_SITE).status_code == 403


def test_cross_site_run_does_not_start(client):
    client.post("/runs", headers={"origin": "null"})
    assert client.get("/status").json()["message"] == "No run yet."


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_schema_is_not_exposed(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/reviews", "/reviews/any", "/outcomes", "/dead-letters"])
def test_the_web_app_cannot_approve_or_change_decisions(client, path):
    assert client.post(path, json={"decision": "approved"}).status_code in (404, 405)


@pytest.mark.parametrize("page", ["dashboard.html", "setup.html"])
def test_pages_never_render_with_inner_html(page):
    source = (ROOT / "api" / page).read_text(encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent" in source


# Runs


def test_run_reports_progress_and_its_jobs(client):
    assert client.post("/runs", headers={"origin": LOCAL}).status_code == 202
    progress = finish_run(client)
    assert progress["running"] is False
    assert progress["checked"] == 1
    assert progress["drafted"] == 1
    assert progress["tokens"] == 300
    assert progress["message"].startswith("Finished: 1 new postings checked, 1 drafts")

    [job] = client.get("/jobs").json()
    assert job["fit_score"] == 0.9
    assert job["fit_reasoning"] == "Match"
    assert job["status"] == "Waiting for review"


def test_cost_is_shown_once_prices_are_set(settings, store, make_deps, env_file):
    priced = replace(settings, input_usd_per_mtok=2.0, output_usd_per_mtok=10.0)

    def factory(_):
        deps = make_deps(scripted(), [POSTING])
        deps.budget = Budget(10**6, 100.0, 2.0, 10.0)
        return deps

    with make_client(priced, store, factory, env_file) as client:
        client.post("/runs")
        assert finish_run(client)["cost_usd"] == pytest.approx(0.0014)


def test_a_failed_run_is_reported_and_does_not_block_the_next(settings, store, env_file):
    def broken(_):
        raise RuntimeError("facts file missing")

    with make_client(settings, store, broken, env_file) as client:
        assert client.post("/runs").status_code == 202
        progress = finish_run(client)
        assert progress["running"] is False
        assert progress["message"] == "Run failed: facts file missing"
        assert client.post("/runs").status_code == 202
        finish_run(client)


def test_a_run_needs_finished_setup(client, settings):
    settings.facts_path.unlink()
    response = client.post("/runs")
    assert response.status_code == 409
    assert "facts" in response.json()["detail"]
    assert client.get("/status").json()["missing"] == ["facts"]


def test_jobs_csv_downloads_every_posting(client, make_deps):
    seed_review(client, make_deps)
    response = client.get("/jobs.csv")
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "Waiting for review" in response.text


def test_scheduled_runs_start_when_due_and_not_again_until_the_interval(
    settings, store, make_deps, env_file
):
    hourly = replace(settings, auto_run_hours=6)
    factory = lambda _: make_deps(scripted(), [POSTING])  # noqa: E731
    with make_client(hourly, store, factory, env_file) as client:
        tick = client.app.state.scheduled_tick
        assert tick(client.app) is True
        finish_run(client)
        assert tick(client.app) is False


def test_scheduled_runs_are_off_by_default(client):
    assert client.app.state.scheduled_tick(client.app) is False


# Setup


def test_setup_state_reports_what_is_missing(client, monkeypatch):
    assert client.get("/setup/state").json()["ready"] is True
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    state = client.get("/setup/state").json()
    assert state["missing"] == ["model"]
    assert state["model"]["key_set"] is False


def test_model_choice_is_saved_and_the_key_is_never_returned(client, env_file):
    body = {"provider": "openai", "name": "some-model", "api_key": "sk-secret-value"}
    response = client.post("/setup/model", json=body)
    assert response.status_code == 200
    assert "sk-secret-value" not in response.text
    assert response.json()["key_set"] is True
    saved = env_file.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-secret-value" in saved
    assert "MODEL_PROVIDER=openai" in saved
    assert "# kept" in saved
    assert "sk-secret-value" not in client.get("/setup/state").text


def test_an_unwritable_env_file_is_explained(settings, store, make_deps, tmp_path):
    read_only = tmp_path / "not-a-file"
    read_only.mkdir()
    factory = lambda _: make_deps(scripted(), [POSTING])  # noqa: E731
    with make_client(settings, store, factory, read_only) as client:
        response = client.post("/setup/model", json={"provider": "openai", "name": "m"})
    assert response.status_code == 503
    assert "set them in your environment" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"provider": "unknown", "name": "m"},
        {"provider": "openai", "name": "m", "api_key": "a b"},
        {"provider": "openai", "name": "m\nOTHER=1"},
        {"provider": "openai", "name": "m", "api_key": "x\nMODEL_PROVIDER=ollama"},
    ],
)
def test_unsafe_model_settings_are_refused(client, env_file, body):
    before = env_file.read_text(encoding="utf-8")
    assert client.post("/setup/model", json=body).status_code == 422
    assert env_file.read_text(encoding="utf-8") == before


def test_facts_are_saved_and_invalid_facts_are_refused(client, settings):
    facts = {"candidate_name": "Sam", "facts": [{"id": "a", "category": "skill", "text": "Go"}]}
    assert client.put("/setup/facts", json=facts).json() == {"saved": 1}
    assert json.loads(settings.facts_path.read_text(encoding="utf-8")) == facts

    duplicate = {**facts, "facts": facts["facts"] * 2}
    assert client.put("/setup/facts", json=duplicate).status_code == 422
    assert json.loads(settings.facts_path.read_text(encoding="utf-8")) == facts


def test_search_is_saved_with_new_sources(client, settings):
    search = {
        **SEARCH,
        "queries": [
            {"source": "ashby", "board": "acme"},
            {"source": "remotive", "search": "python"},
            {"source": "adzuna", "country": "in", "search": "developer"},
        ],
    }
    assert client.put("/setup/search", json=search).json() == {"saved": 3}
    saved = json.loads(settings.search_config_path.read_text(encoding="utf-8"))
    assert saved["queries"][2] == {
        "source": "adzuna",
        "search": "developer",
        "country": "in",
        "keywords": [],
        "exclude_keywords": [],
        "locations": [],
    }


@pytest.mark.parametrize(
    "query",
    [
        {"source": "adzuna", "search": "developer"},
        {"source": "remotive"},
        {"source": "ashby", "board": "../admin"},
        {"source": "url", "url": "file:///etc/passwd"},
    ],
)
def test_invalid_sources_are_refused(client, query):
    assert client.put("/setup/search", json={**SEARCH, "queries": [query]}).status_code == 422


def test_preferences_apply_immediately_and_keys_stay_hidden(client, env_file):
    preferences = {
        "fit_score_threshold": 0.5,
        "max_postings_per_run": 10,
        "max_cost_per_run_usd": 1.5,
        "auto_run_hours": 12,
        "input_usd_per_mtok": 2,
        "output_usd_per_mtok": 10,
        "adzuna_app_id": "app-id",
        "adzuna_app_key": "app-secret",
    }
    response = client.put("/setup/preferences", json=preferences)
    assert response.status_code == 200
    assert "app-secret" not in response.text
    assert response.json()["adzuna_set"] is True
    assert client.app.state.settings.fit_score_threshold == 0.5
    assert client.get("/status").json()["auto_run_hours"] == 12
    assert "ADZUNA_APP_KEY=app-secret" in env_file.read_text(encoding="utf-8")


def test_automatic_runs_cannot_be_too_frequent(client):
    preferences = {
        "fit_score_threshold": 0.7,
        "max_postings_per_run": 10,
        "max_cost_per_run_usd": 1,
        "auto_run_hours": 1,
        "input_usd_per_mtok": 0,
        "output_usd_per_mtok": 0,
    }
    assert client.put("/setup/preferences", json=preferences).status_code == 422


def test_resume_upload_suggests_facts_without_saving_them(client, settings, monkeypatch):
    before = settings.facts_path.read_text(encoding="utf-8")
    suggested = SuggestedFacts(
        candidate_name="Sam",
        facts=[
            SuggestedFact(category="experience", text="Built APIs in Go at Example Corp"),
            SuggestedFact(category="skill", text="Kubernetes, call me on +1 555 010 0199"),
        ],
    )
    model = FakeModel({"SuggestedFacts": [suggested]})
    monkeypatch.setattr("api.setup.get_chat_model", lambda: model)
    resume = b"Sam\nBuilt APIs in Go at Example Corp, 2022 - 2025\n"
    result = client.post("/setup/resume", content=resume).json()
    assert [fact["id"] for fact in result["facts"]] == ["experience-1", "skill-1"]
    assert result["facts"][0]["unsupported"] == []
    assert result["facts"][1] == {
        "id": "skill-1",
        "category": "skill",
        "text": "Kubernetes, call me on",
        "unsupported": ["Kubernetes"],
    }
    assert settings.facts_path.read_text(encoding="utf-8") == before


def test_resume_upload_rejects_empty_and_oversized_files(client):
    assert client.post("/setup/resume", content=b"   ").status_code == 400
    oversized = client.post("/setup/resume", content=b"x" * 5_000_001)
    assert oversized.status_code in (400, 413)


def test_board_search_returns_matches(client, monkeypatch):
    match = BoardMatch("lever", "acme", 4, ("Engineer",))
    monkeypatch.setattr("api.setup.find_boards", lambda query: [match])
    assert client.post("/setup/boards", json={"query": "Acme"}).json() == [
        {"source": "lever", "board": "acme", "open_jobs": 4, "sample_titles": ["Engineer"]}
    ]


def test_rescore_forgets_only_postings_the_model_ruled_out(client, store):
    for url, record_status in (
        ("https://example.com/a", "discarded"),
        ("https://example.com/b", "rejected"),
    ):
        store.write(
            DecisionRecord(
                url=url,
                posting_excerpt="Role",
                status=record_status,
                run_id="r",
                decided_at="2026-01-01T00:00:00+00:00",
            )
        )
    assert client.post("/setup/rescore").json() == {"released": 1}
    assert not store.has_url("https://example.com/a")
    assert store.has_url("https://example.com/b")
