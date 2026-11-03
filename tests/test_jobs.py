import csv
import io

from jobs import job_rows, to_csv
from memory.schema import DecisionRecord


def record(url: str, status: str, score: float | None, excerpt: str) -> DecisionRecord:
    return DecisionRecord(
        url=url,
        posting_excerpt=excerpt,
        status=status,
        run_id=url,
        fit_score=score,
        fit_reasoning="reason" if score is not None else None,
        error="model down" if status == "failed" else None,
        decided_at="2026-01-01T00:00:00+00:00",
    )


def test_rows_are_best_fit_first_with_unscored_last(store):
    store.write(record("https://example.com/a", "discarded", 0.2, "Role A\nLocation: Remote"))
    store.write(record("https://example.com/b", "failed", None, "Role B\nLocation: Remote"))
    store.write(record("https://example.com/c", "applied", 0.9, "Role C\nLocation: City\n\nBody"))
    rows = job_rows(store)
    assert [row["title"] for row in rows] == ["Role C", "Role A", "Role B"]
    assert rows[0]["location"] == "City"
    assert rows[0]["status"] == "Approved, draft saved"
    assert rows[2]["fit_reasoning"] == "model down"


def test_csv_neutralises_spreadsheet_formulas_from_the_web(store):
    hostile = '=HYPERLINK("https://evil.example","click")\nLocation: @SUM(A1)'
    store.write(record("https://example.com/x", "discarded", 0.1, hostile))
    [row] = csv.DictReader(io.StringIO(to_csv(job_rows(store))))
    assert row["title"].startswith("'=")
    assert row["location"].startswith("'@")
