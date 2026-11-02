"""Writes every terminal state to long-term memory for future scoring."""

from datetime import UTC, datetime

from graph.state import JobSearchState
from memory.schema import DecisionRecord, Status
from memory.store import EXCERPT_CHARS, MemoryStore

TERMINAL_STATUS: dict[str, Status] = {
    "approved": "applied",
    "rejected": "rejected",
    "discarded": "discarded",
    "failed": "failed",
}


def make_memory_write(store: MemoryStore):
    def memory_write(state: JobSearchState) -> dict:
        store.write(
            DecisionRecord(
                url=state["posting_url"],
                posting_excerpt=state["posting_text"][:EXCERPT_CHARS],
                status=TERMINAL_STATUS[state["approval_status"]],
                run_id=state["run_id"],
                fit_score=state["fit_score"],
                fit_reasoning=state["fit_reasoning"],
                rejection_reason=state["rejection_reason"],
                error=state["error"],
                decided_at=datetime.now(UTC).isoformat(),
            )
        )
        return {}

    return memory_write
