"""Every posting the agent has seen, as rows for the dashboard, the terminal, or a CSV file."""

import csv
import io
from collections.abc import Iterable

from memory.schema import DecisionRecord
from memory.store import MemoryStore, canonical_url

STATUS_LABELS = {
    "in_progress": "Waiting for review",
    "applied": "Approved, draft saved",
    "rejected": "Rejected by you",
    "discarded": "Below fit threshold",
    "failed": "Failed",
}
COLUMNS = (
    "fit_score",
    "title",
    "location",
    "status",
    "url",
    "fit_reasoning",
    "rejection_reason",
    "outcome",
    "decided_at",
)
# A spreadsheet treats a cell starting with one of these as a formula, and titles come from the web.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _title_and_location(excerpt: str) -> tuple[str, str]:
    lines = excerpt.splitlines()
    title = lines[0].strip() if lines else ""
    location = ""
    if len(lines) > 1 and lines[1].startswith("Location: "):
        location = lines[1].removeprefix("Location: ").strip()
    return title, location


def to_row(record: DecisionRecord) -> dict:
    title, location = _title_and_location(record.posting_excerpt)
    return {
        "fit_score": record.fit_score,
        "title": title,
        "location": location,
        "status": STATUS_LABELS[record.status],
        "url": record.url,
        "fit_reasoning": record.fit_reasoning or record.error or "",
        "rejection_reason": record.rejection_reason or "",
        "outcome": record.outcome or "",
        "decided_at": record.decided_at,
    }


def job_rows(store: MemoryStore, pending: Iterable[dict] = ()) -> list[dict]:
    """Best fit first. Postings never scored, such as failures, go last.

    A posting waiting for review keeps its score in the paused run, so pass the
    pending review payloads to fill it in.
    """
    waiting = {canonical_url(payload["posting_url"]): payload for payload in pending}
    rows = []
    for record in store.all_records():
        row = to_row(record)
        payload = waiting.get(record.url)
        if record.status == "in_progress" and payload:
            row["fit_score"] = payload["fit_score"]
            row["fit_reasoning"] = payload["fit_reasoning"]
        rows.append(row)
    return sorted(rows, key=lambda row: (row["fit_score"] is None, -(row["fit_score"] or 0)))


def _safe_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def to_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _safe_cell(row[key]) for key in COLUMNS})
    return buffer.getvalue()
