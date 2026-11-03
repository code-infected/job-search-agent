"""Every posting the agent has seen, as rows for the dashboard, the terminal, or a CSV file."""

import csv
import io
from collections.abc import Iterable
from urllib.parse import urlsplit

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
    "company",
    "location",
    "status",
    "source",
    "url",
    "fit_reasoning",
    "rejection_reason",
    "outcome",
    "decided_at",
)
# A spreadsheet treats a cell starting with one of these as a formula, and titles come from the web.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _header(excerpt: str) -> dict:
    """Title, location, and company from the header discovery writes into a posting."""
    lines = excerpt.splitlines()
    header = {"title": lines[0].strip() if lines else "", "location": "", "company": ""}
    for line in lines[1:3]:
        for field in ("location", "company"):
            label = f"{field.capitalize()}: "
            if line.startswith(label):
                header[field] = line.removeprefix(label).strip()
    return header


def _source(url: str) -> str:
    """The site a posting came from, which also credits aggregators that ask for it."""
    return (urlsplit(url).hostname or "").removeprefix("www.")


def to_row(record: DecisionRecord) -> dict:
    header = _header(record.posting_excerpt)
    return {
        "fit_score": record.fit_score,
        "title": header["title"],
        "company": header["company"],
        "location": header["location"],
        "status": STATUS_LABELS[record.status],
        "source": _source(record.url),
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
