"""Builds the eval set from a tracker-shaped CSV of past human decisions.

Required columns: url, posting_text, decision, reason. A decision is one of qualified,
vqualified, not-qualified, or not_qualified.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

QUALIFIED = {"qualified", "vqualified"}
NOT_QUALIFIED = {"not-qualified", "not_qualified"}
REQUIRED_COLUMNS = {"url", "posting_text", "decision", "reason"}


@dataclass(frozen=True)
class EvalCase:
    url: str
    posting_text: str
    qualified: bool
    reason: str


def load_cases(path: Path, limit: Optional[int] = None, seed: int = 7) -> list[EvalCase]:
    """Load cases, keeping the qualified/not-qualified mix when sampling down to `limit`."""
    frame = pd.read_csv(path).fillna("")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["decision"] = frame["decision"].str.strip().str.lower()
    unknown = set(frame["decision"]) - QUALIFIED - NOT_QUALIFIED
    if unknown:
        raise ValueError(f"{path} has unknown decisions: {sorted(unknown)}")
    frame = frame[frame["posting_text"].str.strip() != ""]
    if limit is not None and limit < len(frame):
        fraction = limit / len(frame)
        frame = frame.groupby("decision", group_keys=False).sample(frac=fraction, random_state=seed)
    return [
        EvalCase(row.url, row.posting_text, row.decision in QUALIFIED, row.reason)
        for row in frame.itertuples()
    ]
