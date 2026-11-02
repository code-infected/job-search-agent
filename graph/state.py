"""The full contract passed between graph nodes. One run handles one posting."""

from typing import Literal, Optional, TypedDict


class JobSearchState(TypedDict):
    posting_url: str
    posting_text: str
    fit_score: Optional[float]
    fit_reasoning: Optional[str]
    draft_message: Optional[str]
    approval_status: Literal["pending", "approved", "rejected", "discarded", "failed"]
    rejection_reason: Optional[str]
    error: Optional[str]
    run_id: str


def initial_state(run_id: str) -> JobSearchState:
    return JobSearchState(
        posting_url="",
        posting_text="",
        fit_score=None,
        fit_reasoning=None,
        draft_message=None,
        approval_status="pending",
        rejection_reason=None,
        error=None,
        run_id=run_id,
    )
