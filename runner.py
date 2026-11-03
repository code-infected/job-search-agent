"""Runs discovery batches and resolves pending reviews against the checkpointer."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from config import Settings
from graph.build import Dependencies, build_graph
from graph.facts import load_facts
from graph.llm import Budget
from graph.nodes.discover import FetchLog, PoliteClient, fetch_postings, load_search_config
from graph.nodes.human_review import ReviewDecision
from graph.nodes.send import build_draft_sink
from graph.sources.directory import BoardDirectory
from graph.state import initial_state
from memory.store import MemoryStore
from models import get_chat_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingReview:
    run_id: str
    created_at: str
    payload: dict[str, Any]


def build_dependencies(settings: Settings, store: Optional[MemoryStore] = None) -> Dependencies:
    """Load everything a run needs, failing at startup rather than mid-graph."""
    search = load_search_config(settings.search_config_path)
    client = PoliteClient(delay_seconds=settings.request_delay_seconds)
    fetch_log = FetchLog(settings.checkpoint_db.parent / "source_fetches.json")
    directory = BoardDirectory(directory_path(settings))
    return Dependencies(
        model=get_chat_model(),
        store=store or MemoryStore(settings.vector_store_path),
        facts=load_facts(settings.facts_path),
        target_profile=search.target_profile,
        fit_score_threshold=settings.fit_score_threshold,
        sink=build_draft_sink(settings.draft_backend, settings.drafts_dir),
        budget=Budget(
            settings.max_tokens_per_run,
            settings.max_cost_per_run_usd,
            settings.input_usd_per_mtok,
            settings.output_usd_per_mtok,
        ),
        postings=fetch_postings(search.resolved_queries(), client, fetch_log, directory),
    )


def directory_path(settings: Settings) -> Path:
    return settings.checkpoint_db.parent / "board_directory.json"


def build_read_only_graph(store: MemoryStore, checkpointer: SqliteSaver):
    """A graph for reading paused runs. Nodes are never invoked, so it needs no model or facts."""
    deps = Dependencies(
        model=None,
        store=store,
        facts=None,
        target_profile="",
        fit_score_threshold=1.0,
        sink=None,
        budget=None,
        postings=iter(()),
    )
    return build_graph(deps, checkpointer)


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def run_batch(
    deps: Dependencies,
    checkpointer: SqliteSaver,
    max_postings: int,
    on_result: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """Process postings until discovery runs dry, or a cap or the budget stops the batch."""
    graph = build_graph(deps, checkpointer)
    results = []
    for _ in range(max_postings):
        if deps.budget.exceeded:
            logger.warning("stopping batch: spend ceiling reached")
            break
        run_id = uuid4().hex
        values = graph.invoke(initial_state(run_id), _config(run_id))
        if not values["posting_url"]:
            break
        results.append(values)
        if on_result:
            on_result(values)
    return results


def pending_reviews(graph, checkpointer: SqliteSaver) -> list[PendingReview]:
    thread_ids = {
        checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpointer.list(None)
    }
    pending = []
    for thread_id in thread_ids:
        snapshot = graph.get_state(_config(thread_id))
        if snapshot.interrupts:
            pending.append(
                PendingReview(thread_id, snapshot.created_at, snapshot.interrupts[0].value)
            )
    return sorted(pending, key=lambda review: review.created_at)


def is_pending(graph, run_id: str) -> bool:
    return bool(graph.get_state(_config(run_id)).interrupts)


def submit_decision(graph, run_id: str, decision: ReviewDecision) -> dict:
    """Resume a paused run. This is the only way past the human_review interrupt."""
    return graph.invoke(Command(resume=decision.model_dump()), _config(run_id))


def stale_reviews(
    reviews: list[PendingReview], stale_hours: float, now: Optional[datetime] = None
) -> list[PendingReview]:
    """The daily digest: reviews that have waited longer than `stale_hours`."""
    cutoff = (now or datetime.now(UTC)) - timedelta(hours=stale_hours)
    return [r for r in reviews if datetime.fromisoformat(r.created_at) <= cutoff]
