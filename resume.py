"""Turn a resume into suggested facts for the user to review.

Suggestions are never saved directly. Every draft is checked against the facts store,
so a mistake made here would later pass that check as if it were verified. The user
confirms each fact first, and anything a suggestion names that the resume does not
(a number, a technology, an employer) is flagged for that review.
"""

import io
import re
from collections import Counter
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from graph.llm import UNTRUSTED_DATA_RULE, Budget, invoke_structured, wrap_untrusted

MAX_RESUME_BYTES = 5_000_000
MAX_RESUME_CHARS = 30_000
MAX_PDF_PAGES = 10
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
MIN_PHONE_DIGITS = 10  # so a date range such as "2022 - 2025" is kept
# Capitalised words and anything with a digit: names, employers, technologies, and figures.
CHECKED_TERM_RE = re.compile(r"\b(?:[A-Z][\w+#.-]*|[\w+#.-]*\d[\w+#.,-]*)")

SYSTEM_PROMPT = f"""You turn a resume into short facts that a job-search assistant may quote.

Rules:
- Use only what the resume states. Never add, infer, round, or embellish anything.
- Copy names, employers, schools, job titles, dates, numbers, and technologies exactly.
- One fact per role, project, achievement, degree, or skill group. Each fact must stand on
  its own, so name the employer or project inside every fact about it.
- Keep every hedge the resume uses, such as "estimated" or "targeting".
- Leave out phone numbers, email addresses, home addresses, and references.
- category is one of: experience, project, education, skill, certification, award, other.

{UNTRUSTED_DATA_RULE}"""

Category = Literal["experience", "project", "education", "skill", "certification", "award", "other"]


class ResumeError(ValueError):
    """The upload could not be read as a resume."""


class SuggestedFact(BaseModel):
    category: Category
    text: str = Field(min_length=1, max_length=600)


class SuggestedFacts(BaseModel):
    candidate_name: str = Field(min_length=1, max_length=200)
    facts: list[SuggestedFact] = Field(min_length=1, max_length=80)


def resume_text(data: bytes) -> str:
    """Text from a PDF or a plain-text resume."""
    if len(data) > MAX_RESUME_BYTES:
        raise ResumeError("the file is larger than 5 MB")
    if data.startswith(b"%PDF"):
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = reader.pages[:MAX_PDF_PAGES]
            text = "\n".join(page.extract_text() or "" for page in pages)
        except (PdfReadError, ValueError, KeyError, TypeError) as exc:
            raise ResumeError(f"could not read the PDF: {exc}") from exc
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ResumeError("upload a PDF or a plain-text file") from exc
    text = text.strip()
    if not text:
        raise ResumeError("no text found. A scanned PDF has no text to read, so paste the text")
    return text[:MAX_RESUME_CHARS]


def strip_contact_details(text: str) -> str:
    text = EMAIL_RE.sub("", text)
    text = PHONE_RE.sub(
        lambda match: "" if sum(ch.isdigit() for ch in match[0]) >= MIN_PHONE_DIGITS else match[0],
        text,
    )
    return text.strip(" ,;")


def unsupported_terms(fact: str, resume: str) -> list[str]:
    """Names and figures in a fact that the resume does not contain."""
    haystack = resume.lower()
    terms = (term.strip(".,-") for term in CHECKED_TERM_RE.findall(fact))
    return sorted({term for term in terms if term and term.lower() not in haystack})


def suggest_facts(model: BaseChatModel, resume: str, budget: Budget) -> dict:
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"Resume:\n{wrap_untrusted(resume)}"),
    ]
    result = invoke_structured(model, SuggestedFacts, messages, budget)
    counts: Counter[str] = Counter()
    facts = []
    for fact in result.facts:
        text = strip_contact_details(fact.text)
        if not text:
            continue
        counts[fact.category] += 1
        facts.append(
            {
                "id": f"{fact.category}-{counts[fact.category]}",
                "category": fact.category,
                "text": text,
                "unsupported": unsupported_terms(text, resume),
            }
        )
    return {"candidate_name": result.candidate_name, "facts": facts}
