"""The human approval gate. Execution stops here until an external decision arrives."""

from typing import Literal, Optional

from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph.state import JobSearchState


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected"]
    reason: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def rejection_has_reason(self) -> "ReviewDecision":
        if self.decision == "rejected" and not (self.reason or "").strip():
            raise ValueError("a rejection needs a reason, it becomes precedent for later runs")
        return self


def human_review(state: JobSearchState) -> dict:
    payload = interrupt(
        {
            "run_id": state["run_id"],
            "posting_url": state["posting_url"],
            "posting_text": state["posting_text"],
            "fit_score": state["fit_score"],
            "fit_reasoning": state["fit_reasoning"],
            "draft_message": state["draft_message"],
        }
    )
    decision = ReviewDecision.model_validate(payload)
    return {"approval_status": decision.decision, "rejection_reason": decision.reason}
