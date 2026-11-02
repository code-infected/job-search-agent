"""StateGraph assembly: two decision gates and one memory loop."""

import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from graph.facts import FactsStore
from graph.llm import Budget
from graph.nodes.discover import Posting, make_discover
from graph.nodes.fit_score import make_fit_score
from graph.nodes.human_review import human_review
from graph.nodes.memory_write import make_memory_write
from graph.nodes.send import DraftSink, make_send
from graph.nodes.tailor import make_tailor
from graph.state import JobSearchState
from memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class Dependencies:
    model: BaseChatModel
    store: MemoryStore
    facts: FactsStore
    target_profile: str
    fit_score_threshold: float
    sink: DraftSink
    budget: Budget
    postings: Iterator[Posting]


def open_checkpointer(path: Path) -> SqliteSaver:
    """Open the durable checkpointer that lets a pending review survive a restart."""
    if ":memory:" in str(path) or not str(path).strip("."):
        raise ValueError("CHECKPOINT_DB must be a file: a pending review has to survive a restart")
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    saver.setup()
    return saver


def mark_discarded(state: JobSearchState) -> dict:
    logger.info("discarded %s at score %s", state["posting_url"], state["fit_score"])
    return {"approval_status": "discarded"}


def dead_letter(state: JobSearchState) -> dict:
    logger.error("dead letter %s: %s", state["posting_url"], state["error"])
    return {"approval_status": "failed"}


def build_graph(deps: Dependencies, checkpointer: SqliteSaver):
    def after_discover(state: JobSearchState) -> str:
        return "fit_score" if state["posting_url"] else END

    def after_fit_score(state: JobSearchState) -> str:
        if state["error"]:
            return "dead_letter"
        if state["fit_score"] >= deps.fit_score_threshold:
            return "tailor"
        return "mark_discarded"

    def after_tailor(state: JobSearchState) -> str:
        return "dead_letter" if state["error"] else "human_review"

    def after_review(state: JobSearchState) -> str:
        return "send" if state["approval_status"] == "approved" else "memory_write"

    def after_send(state: JobSearchState) -> str:
        return "dead_letter" if state["error"] else "memory_write"

    graph = StateGraph(JobSearchState)
    graph.add_node("discover", make_discover(deps.postings, deps.store))
    graph.add_node(
        "fit_score",
        make_fit_score(deps.model, deps.store, deps.facts, deps.target_profile, deps.budget),
    )
    graph.add_node("mark_discarded", mark_discarded)
    graph.add_node("tailor", make_tailor(deps.model, deps.facts, deps.budget))
    graph.add_node("human_review", human_review)
    graph.add_node("send", make_send(deps.sink))
    graph.add_node("dead_letter", dead_letter)
    graph.add_node("memory_write", make_memory_write(deps.store))

    graph.add_edge(START, "discover")
    graph.add_conditional_edges("discover", after_discover, ["fit_score", END])
    graph.add_conditional_edges(
        "fit_score", after_fit_score, ["tailor", "mark_discarded", "dead_letter"]
    )
    graph.add_edge("mark_discarded", "memory_write")
    graph.add_conditional_edges("tailor", after_tailor, ["human_review", "dead_letter"])
    graph.add_conditional_edges("human_review", after_review, ["send", "memory_write"])
    graph.add_conditional_edges("send", after_send, ["memory_write", "dead_letter"])
    graph.add_edge("dead_letter", "memory_write")
    graph.add_edge("memory_write", END)
    return graph.compile(checkpointer=checkpointer)
