import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.types import Command

from graph.build import build_graph, open_checkpointer
from graph.llm import Budget
from graph.nodes.fit_score import FitAssessment
from graph.nodes.human_review import ReviewDecision
from graph.nodes.tailor import TailoredDraft
from runner import is_pending, pending_reviews, run_batch, stale_reviews, submit_decision
from tests.conftest import POSTING, ROOT, FakeModel

HIGH = FitAssessment(fit_score=0.9, fit_reasoning="Strong Python match")
LOW = FitAssessment(fit_score=0.2, fit_reasoning="Kubernetes gap")
GOOD = TailoredDraft(
    message="I built REST APIs in Python and FastAPI backed by PostgreSQL.",
    cited_fact_ids=["exp-api"],
)
BAD = TailoredDraft(message="I have 5 years of Kubernetes.", cited_fact_ids=["exp-api"])


@pytest.fixture(autouse=True)
def no_sleep():
    with patch("graph.llm.time.sleep"):
        yield


def test_low_score_is_discarded_with_reasoning_and_no_draft(make_deps, checkpointer, store):
    model = FakeModel({"FitAssessment": [LOW]})
    [result] = run_batch(make_deps(model, [POSTING]), checkpointer, 5)
    assert result["approval_status"] == "discarded"
    assert result["draft_message"] is None
    assert store.by_status("discarded")[0].fit_reasoning == "Kubernetes gap"


def test_high_score_pauses_at_review_and_creates_nothing(make_deps, checkpointer, store, tmp_path):
    deps = make_deps(FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [GOOD]}), [POSTING])
    [result] = run_batch(deps, checkpointer, 5)
    graph = build_graph(deps, checkpointer)
    assert result["approval_status"] == "pending"
    assert is_pending(graph, result["run_id"])
    assert not (tmp_path / "drafts").exists()
    assert store.by_status("applied") == []
    assert [r.payload["draft_message"] for r in pending_reviews(graph, checkpointer)] == [
        GOOD.message
    ]


def test_approval_creates_a_draft_and_writes_memory(make_deps, checkpointer, store, tmp_path):
    deps = make_deps(FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [GOOD]}), [POSTING])
    [paused] = run_batch(deps, checkpointer, 5)
    graph = build_graph(deps, checkpointer)
    final = submit_decision(graph, paused["run_id"], ReviewDecision(decision="approved"))
    assert final["approval_status"] == "approved"
    assert GOOD.message in (tmp_path / "drafts" / f"{paused['run_id']}.md").read_text()
    assert store.by_status("applied")[0].url == POSTING.url
    assert pending_reviews(graph, checkpointer) == []


def test_rejection_skips_send_and_becomes_precedent(make_deps, checkpointer, store, tmp_path):
    deps = make_deps(FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [GOOD]}), [POSTING])
    [paused] = run_batch(deps, checkpointer, 5)
    graph = build_graph(deps, checkpointer)
    decision = ReviewDecision(decision="rejected", reason="On-call heavy")
    final = submit_decision(graph, paused["run_id"], decision)
    assert final["approval_status"] == "rejected"
    assert not (tmp_path / "drafts").exists()
    assert store.similar_decisions(POSTING.text)[0].rejection_reason == "On-call heavy"


def test_precedent_from_one_run_reaches_the_next_runs_prompt(make_deps, checkpointer):
    deps = make_deps(FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [GOOD]}), [POSTING])
    [paused] = run_batch(deps, checkpointer, 5)
    decision = ReviewDecision(decision="rejected", reason="Unique rejection marker")
    submit_decision(build_graph(deps, checkpointer), paused["run_id"], decision)

    similar = type(POSTING)("https://example.com/jobs/other", POSTING.text + " Remote.")
    model = FakeModel({"FitAssessment": [LOW]})
    run_batch(make_deps(model, [similar]), checkpointer, 5)
    assert "Unique rejection marker" in model.calls[0][1].content


def test_a_malformed_resume_cannot_pass_the_gate(make_deps, checkpointer, tmp_path):
    deps = make_deps(FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [GOOD]}), [POSTING])
    [paused] = run_batch(deps, checkpointer, 5)
    graph = build_graph(deps, checkpointer)
    config = {"configurable": {"thread_id": paused["run_id"]}}
    with pytest.raises(ValueError):
        graph.invoke(Command(resume={"decision": "yes"}), config)
    assert not (tmp_path / "drafts").exists()
    assert is_pending(graph, paused["run_id"])


def test_ungrounded_draft_dead_letters_and_never_reaches_review(make_deps, checkpointer, store):
    model = FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [BAD, BAD]})
    deps = make_deps(model, [POSTING])
    [result] = run_batch(deps, checkpointer, 5)
    assert result["approval_status"] == "failed"
    assert pending_reviews(build_graph(deps, checkpointer), checkpointer) == []
    assert "kubernet" in store.by_status("failed")[0].error


def test_model_outage_dead_letters_instead_of_crashing(make_deps, checkpointer, store):
    second = type(POSTING)("https://example.com/jobs/two", "Second posting")
    model = FakeModel({"FitAssessment": [RuntimeError("500")] * 3 + [LOW]})
    results = run_batch(make_deps(model, [POSTING, second]), checkpointer, 5)
    assert [r["approval_status"] for r in results] == ["failed", "discarded"]


def test_dead_letter_is_not_rediscovered(make_deps, checkpointer):
    model = FakeModel({"FitAssessment": [RuntimeError("500")] * 3})
    run_batch(make_deps(model, [POSTING]), checkpointer, 5)
    again = FakeModel({})
    assert run_batch(make_deps(again, [POSTING]), checkpointer, 5) == []
    assert again.calls == []


def test_max_postings_cap_is_respected(make_deps, checkpointer):
    postings = [type(POSTING)(f"https://example.com/{i}", "text") for i in range(3)]
    model = FakeModel({"FitAssessment": [LOW] * 3})
    assert len(run_batch(make_deps(model, postings), checkpointer, 2)) == 2


def test_budget_stops_the_batch(make_deps, checkpointer):
    postings = [type(POSTING)(f"https://example.com/{i}", "text") for i in range(3)]
    deps = make_deps(FakeModel({"FitAssessment": [LOW] * 3}), postings)
    deps.budget = Budget(max_tokens=100, max_cost_usd=100.0)
    assert len(run_batch(deps, checkpointer, 5)) == 1


def test_in_progress_review_is_not_rediscovered(make_deps, checkpointer):
    deps = make_deps(FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [GOOD]}), [POSTING])
    run_batch(deps, checkpointer, 5)
    again = FakeModel({})
    assert run_batch(make_deps(again, [POSTING]), checkpointer, 5) == []


def test_digest_lists_only_stale_reviews(make_deps, checkpointer):
    deps = make_deps(FakeModel({"FitAssessment": [HIGH], "TailoredDraft": [GOOD]}), [POSTING])
    run_batch(deps, checkpointer, 5)
    reviews = pending_reviews(build_graph(deps, checkpointer), checkpointer)
    assert stale_reviews(reviews, 24) == []
    assert stale_reviews(reviews, 0) == reviews


def test_in_memory_checkpointer_is_not_an_option():
    with pytest.raises(ValueError, match="CHECKPOINT_DB"):
        open_checkpointer(Path(":memory:"))


RESTART_PREAMBLE = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from graph.build import build_graph, open_checkpointer
    from graph.nodes.discover import Posting
    from graph.nodes.fit_score import FitAssessment
    from graph.nodes.human_review import ReviewDecision
    from graph.nodes.send import FileDraftSink
    from graph.nodes.tailor import TailoredDraft
    from graph.facts import load_facts
    from graph.llm import Budget
    from graph.build import Dependencies
    from memory.store import MemoryStore
    from runner import pending_reviews, run_batch, submit_decision
    from tests.conftest import FakeModel, HashEmbedding

    base = Path(sys.argv[1])
    model = FakeModel({
        "FitAssessment": [FitAssessment(fit_score=0.9, fit_reasoning="Match")],
        "TailoredDraft": [TailoredDraft(
            message="I built REST APIs in Python and FastAPI backed by PostgreSQL.",
            cited_fact_ids=["exp-api"],
        )],
    })
    deps = Dependencies(
        model=model,
        store=MemoryStore(base / "chroma", embedding_function=HashEmbedding()),
        facts=load_facts(Path("config/facts.example.json")),
        target_profile="Backend",
        fit_score_threshold=0.7,
        sink=FileDraftSink(base / "drafts"),
        budget=Budget(10**6, 100.0),
        postings=iter([Posting("https://example.com/restart", "Backend role")]),
    )
    saver = open_checkpointer(base / "checkpoints.sqlite")
    """
)


def run_process(body: str, base: Path) -> str:
    script = RESTART_PREAMBLE + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", script, str(base)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_pending_review_survives_a_real_process_restart(tmp_path):
    """Pause in one OS process, resume in a separate one."""
    started = json.loads(
        run_process(
            """
            import json
            [values] = run_batch(deps, saver, 5)
            print(json.dumps({key: values[key] for key in ("run_id", "approval_status")}))
            """,
            tmp_path,
        )
    )
    assert started["approval_status"] == "pending"
    assert not (tmp_path / "drafts").exists()

    resumed = run_process(
        """
        graph = build_graph(deps, saver)
        [review] = pending_reviews(graph, saver)
        assert not model.calls, "a resume must not redo earlier nodes"
        print(submit_decision(graph, review.run_id, ReviewDecision(decision="approved"))["approval_status"])
        """,
        tmp_path,
    )
    assert resumed == "approved"
    assert (tmp_path / "drafts" / f"{started['run_id']}.md").exists()
