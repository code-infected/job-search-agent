"""Job search APIs that need a free key from their provider, set in .env.

Keys travel in URLs or headers, so a failure is reported by status alone and never
repeats the request.
"""

import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import httpx

from graph.sources.common import PoliteClient, Posting, html_to_text, matches, posting_text

if TYPE_CHECKING:
    from graph.nodes.discover import Query

SUMMARY_NOTE = "Only a short summary of this posting was available."
ADZUNA_PAGE_SIZE = 50
REED_PAGE_SIZE = 100
USAJOBS_PAGE_SIZE = 100


def _key(*names: str) -> list[str]:
    values = [os.environ.get(name, "") for name in names]
    if not all(values):
        raise ValueError(f"this source needs {' and '.join(names)} in .env")
    return values


def _call(source: str, request: Callable[[], httpx.Response]) -> Any:
    try:
        return request().json()
    except (httpx.HTTPError, ValueError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", type(exc).__name__)
        raise ValueError(f"{source} request failed: {status}") from None


def _summary(text: str) -> str:
    return f"{html_to_text(text)}\n\n{SUMMARY_NOTE}"


def adzuna(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Keyword search across employers in one country."""
    app_id, app_key = _key("ADZUNA_APP_ID", "ADZUNA_APP_KEY")
    for page in (1, 2):
        url = f"https://api.adzuna.com/v1/api/jobs/{query.country}/search/{page}"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": query.search,
            "results_per_page": str(ADZUNA_PAGE_SIZE),
            "max_days_old": "30",
            "content-type": "application/json",
        }
        if query.where:
            params["where"] = query.where
        results = _call("Adzuna", lambda u=url, p=params: client.get(u, **p)).get("results", [])
        for job in results:
            title = html_to_text(job.get("title", ""))
            location = (job.get("location") or {}).get("display_name", "")
            if matches(title, location, query):
                company = (job.get("company") or {}).get("display_name", "")
                body = _summary(job.get("description", ""))
                yield Posting(job["redirect_url"], posting_text(title, location, body, company))
        if len(results) < ADZUNA_PAGE_SIZE:
            break


def jooble(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Search across many job sites in most countries."""
    (key,) = _key("JOOBLE_API_KEY")
    body = {"keywords": query.search, "location": query.where or ""}
    data = _call(
        "Jooble", lambda: client.request("POST", f"https://jooble.org/api/{key}", json=body)
    )
    for job in data.get("jobs", []):
        title = html_to_text(job.get("title", ""))
        location = job.get("location", "")
        if matches(title, location, query):
            text = posting_text(
                title, location, _summary(job.get("snippet", "")), job.get("company")
            )
            yield Posting(job["link"], text)


def usajobs(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """United States federal government jobs."""
    key, email = _key("USAJOBS_API_KEY", "USAJOBS_EMAIL")
    params = {"Keyword": query.search, "ResultsPerPage": str(USAJOBS_PAGE_SIZE)}
    if query.where:
        params["LocationName"] = query.where
    headers = {"Authorization-Key": key, "User-Agent": email}
    data = _call(
        "USAJobs",
        lambda: client.request(
            "GET", "https://data.usajobs.gov/api/search", params=params, headers=headers
        ),
    )
    for item in (data.get("SearchResult") or {}).get("SearchResultItems", []):
        job = item.get("MatchedObjectDescriptor") or {}
        title = job.get("PositionTitle", "")
        location = job.get("PositionLocationDisplay", "")
        if matches(title, location, query):
            details = ((job.get("UserArea") or {}).get("Details") or {}).get("JobSummary", "")
            body = "\n\n".join(
                part for part in (details, job.get("QualificationSummary", "")) if part
            )
            company = job.get("OrganizationName", "")
            yield Posting(job["PositionURI"], posting_text(title, location, body, company))


def reed(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Jobs in the United Kingdom."""
    (key,) = _key("REED_API_KEY")
    params = {"keywords": query.search, "resultsToTake": str(REED_PAGE_SIZE)}
    if query.where:
        params["locationName"] = query.where
    data = _call(
        "Reed",
        lambda: client.request(
            "GET", "https://www.reed.co.uk/api/1.0/search", params=params, auth=(key, "")
        ),
    )
    for job in data.get("results", []):
        title = job.get("jobTitle", "")
        location = job.get("locationName", "")
        if matches(title, location, query):
            body = _summary(job.get("jobDescription", ""))
            company = job.get("employerName", "")
            yield Posting(job["jobUrl"], posting_text(title, location, body, company))


def findwork(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Software jobs gathered from many boards."""
    (key,) = _key("FINDWORK_API_KEY")
    params = {"search": query.search}
    if query.where:
        params["location"] = query.where
    data = _call(
        "Findwork",
        lambda: client.request(
            "GET",
            "https://findwork.dev/api/jobs/",
            params=params,
            headers={"Authorization": f"Token {key}"},
        ),
    )
    for job in data.get("results", []):
        title = job.get("role", "")
        location = job.get("location") or ""
        if job.get("remote"):
            location = f"{location}; Remote" if location else "Remote"
        if matches(title, location, query):
            body = html_to_text(job.get("text", ""))
            yield Posting(job["url"], posting_text(title, location, body, job.get("company_name")))
