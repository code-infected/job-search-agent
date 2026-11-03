"""Resume reading, .env writing, and the job list columns."""

import os

import pytest

from envfile import update_env
from graph.llm import Budget
from jobs import job_rows
from memory.schema import DecisionRecord
from resume import (
    ResumeError,
    SuggestedFact,
    SuggestedFacts,
    resume_text,
    strip_contact_details,
    suggest_facts,
    unsupported_terms,
)
from tests.conftest import FakeModel


def tiny_pdf(text: str) -> bytes:
    """A one-page PDF with a line of text, built by hand so tests need no PDF writer."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return out


def test_text_is_read_from_a_pdf():
    assert "Built APIs in Go" in resume_text(tiny_pdf("Built APIs in Go"))


def test_text_is_read_from_a_plain_text_file():
    assert resume_text("﻿Sam\nEngineer".encode()) == "Sam\nEngineer"


@pytest.mark.parametrize("data", [b"", b"   ", b"\xff\xfe\x00bad", b"%PDF-1.4 broken"])
def test_unreadable_uploads_fail_with_a_reason(data):
    with pytest.raises(ResumeError):
        resume_text(data)


def test_contact_details_are_removed_but_date_ranges_are_kept():
    assert strip_contact_details("Mail me at sam@example.com") == "Mail me at"
    assert strip_contact_details("Call +91 98765 43210") == "Call"
    assert strip_contact_details("Engineer, 2022 - 2025") == "Engineer, 2022 - 2025"


def test_names_and_figures_missing_from_the_resume_are_flagged():
    resume = "Built a FastAPI service at Example Corp handling 2,000 requests."
    assert unsupported_terms("Built a FastAPI service at Example Corp", resume) == []
    assert unsupported_terms("Built a Django service handling 5,000 requests", resume) == [
        "5,000",
        "Django",
    ]


def test_suggestions_get_ids_per_category():
    suggested = SuggestedFacts(
        candidate_name="Sam",
        facts=[
            SuggestedFact(category="skill", text="Python"),
            SuggestedFact(category="skill", text="Go"),
            SuggestedFact(category="education", text="BSc"),
        ],
    )
    model = FakeModel({"SuggestedFacts": [suggested]})
    result = suggest_facts(model, "Python Go BSc", Budget(10**6, 100.0))
    assert [fact["id"] for fact in result["facts"]] == ["skill-1", "skill-2", "education-1"]
    assert "untrusted_data_" in model.calls[0][1].content


def test_env_updates_keep_other_lines_and_reach_the_process(tmp_path, monkeypatch):
    monkeypatch.delenv("JSA_TEST_SETTING", raising=False)
    path = tmp_path / ".env"
    path.write_text("# comment\nA=1\nJSA_TEST_SETTING=old  # note\n", encoding="utf-8")
    update_env(path, {"JSA_TEST_SETTING": "new", "B": "2"})
    assert path.read_text(encoding="utf-8") == "# comment\nA=1\nJSA_TEST_SETTING=new\nB=2\n"
    assert os.environ["JSA_TEST_SETTING"] == "new"
    monkeypatch.delenv("B")


@pytest.mark.parametrize(
    "key, value",
    [
        ("lower", "x"),
        ("A", "two words"),
        ("A", "x\nB=y"),
        ("A", "x#y"),
        ("A", "$OTHER"),
        ("A", "a'b"),
    ],
)
def test_env_updates_refuse_values_that_would_change_parsing(tmp_path, key, value):
    path = tmp_path / ".env"
    path.write_text("A=1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_env(path, {key: value})
    assert path.read_text(encoding="utf-8") == "A=1\n"


def test_job_rows_show_company_and_source(store):
    store.write(
        DecisionRecord(
            url="https://www.jobs.example.com/1",
            posting_excerpt="Developer\nLocation: City A\nCompany: Acme\n\nBody",
            status="discarded",
            run_id="r",
            fit_score=0.4,
            fit_reasoning="Stack mismatch",
            decided_at="2026-01-01T00:00:00+00:00",
        )
    )
    [row] = job_rows(store)
    assert (row["title"], row["company"], row["location"]) == ("Developer", "Acme", "City A")
    assert row["source"] == "jobs.example.com"
