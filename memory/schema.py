"""Long-term decision records kept across runs."""

from typing import Literal, Optional

from pydantic import BaseModel

Status = Literal["in_progress", "discarded", "rejected", "applied", "failed"]

# Only finished human or threshold decisions count as precedent for future scoring.
PRECEDENT_STATUSES: tuple[Status, ...] = ("discarded", "rejected", "applied")


class DecisionRecord(BaseModel):
    url: str
    posting_excerpt: str
    status: Status
    run_id: str
    fit_score: Optional[float] = None
    fit_reasoning: Optional[str] = None
    rejection_reason: Optional[str] = None
    error: Optional[str] = None
    outcome: Optional[str] = None
    decided_at: str
