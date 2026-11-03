"""Public job feeds that list postings from many employers, read without an account.

Each of these asks that it is credited and linked, so every posting names its source
and keeps the feed's own URL, which the job list links to.
"""

import html
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

from defusedxml import ElementTree

from graph.sources.common import (
    PoliteClient,
    Posting,
    html_to_text,
    join,
    matches,
    posting_text,
)

if TYPE_CHECKING:
    from graph.nodes.discover import Query

ARBEITNOW_MAX_PAGES = 3
HIMALAYAS_PAGE_SIZE = 20
HIMALAYAS_MAX_PAGES = 5
LANDING_JOBS_PAGE_SIZE = 50
LANDING_JOBS_MAX_PAGES = 4
FOUR_DAY_WEEK_MAX_PAGES = 8
GET_ON_BOARD_PAGE_SIZE = 50
HN_MAX_TITLE_CHARS = 300


def _credited(source: str, body: str) -> str:
    return f"Found through {source}.\n\n{body}"


def remotive(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    data = client.get("https://remotive.com/api/remote-jobs", search=query.search).json()
    for job in data.get("jobs", []):
        title = job.get("title", "")
        location = f"Remote ({job.get('candidate_required_location') or 'Worldwide'})"
        if matches(title, location, query):
            body = _credited("Remotive", html_to_text(job.get("description", "")))
            yield Posting(job["url"], posting_text(title, location, body, job.get("company_name")))


def arbeitnow(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Mostly jobs in Germany and Europe. The feed has no search, so titles are filtered here."""
    for page in range(1, ARBEITNOW_MAX_PAGES + 1):
        data = client.get("https://www.arbeitnow.com/api/job-board-api", page=str(page)).json()
        for job in data.get("data", []):
            title = job.get("title", "")
            location = job.get("location", "")
            if job.get("remote"):
                location = f"{location}; Remote" if location else "Remote"
            if matches(title, location, query):
                body = _credited("Arbeitnow", html_to_text(job.get("description", "")))
                company = job.get("company_name", "")
                yield Posting(job["url"], posting_text(title, location, body, company))
        if not (data.get("links") or {}).get("next"):
            break


def remoteok(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """The first entry is Remote OK's terms notice, not a job."""
    for job in client.get("https://remoteok.com/api").json()[1:]:
        if not isinstance(job, dict):
            continue
        title = job.get("position", "")
        location = f"Remote ({job.get('location') or 'Worldwide'})"
        if matches(title, location, query):
            body = _credited("Remote OK", html_to_text(job.get("description", "")))
            yield Posting(job["url"], posting_text(title, location, body, job.get("company")))


def jobicy(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    params = {"count": "50"}
    if query.search:
        params["tag"] = query.search
    data = client.get("https://jobicy.com/api/v2/remote-jobs", **params).json()
    for job in data.get("jobs", []):
        title = html.unescape(job.get("jobTitle", ""))
        location = f"Remote ({job.get('jobGeo') or 'Anywhere'})"
        if matches(title, location, query):
            body = _credited("Jobicy", html_to_text(job.get("jobDescription", "")))
            company = job.get("companyName", "")
            yield Posting(job["url"], posting_text(title, location, body, company))


def himalayas(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    for page in range(1, HIMALAYAS_MAX_PAGES + 1):
        params = {"limit": str(HIMALAYAS_PAGE_SIZE), "page": str(page)}
        if query.search:
            params["q"] = query.search
        jobs = client.get("https://himalayas.app/jobs/api/search", **params).json().get("jobs", [])
        for job in jobs:
            title = job.get("title", "")
            places = job.get("locationRestrictions") or ["Worldwide"]
            location = f"Remote ({', '.join(str(place) for place in places)})"
            if matches(title, location, query):
                body = _credited("Himalayas", html_to_text(job.get("description", "")))
                url = job.get("applicationLink") or job["guid"]
                yield Posting(url, posting_text(title, location, body, job.get("companyName")))
        if len(jobs) < HIMALAYAS_PAGE_SIZE:
            break


def working_nomads(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    for job in client.get("https://www.workingnomads.com/api/exposed_jobs/").json():
        title = job.get("title", "")
        location = f"Remote ({job.get('location') or 'Anywhere'})"
        if matches(title, location, query):
            body = _credited("Working Nomads", html_to_text(job.get("description", "")))
            company = job.get("company_name", "")
            yield Posting(job["url"], posting_text(title, location, body, company))


def we_work_remotely(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """An RSS feed. Titles read "Company: Role"."""
    root = ElementTree.fromstring(client.get("https://weworkremotely.com/remote-jobs.rss").content)
    for item in root.iterfind("./channel/item"):
        heading = item.findtext("title") or ""
        company, _, title = heading.partition(": ")
        if not title:
            company, title = "", heading
        location = f"Remote ({item.findtext('region') or 'Anywhere'})"
        if matches(title, location, query):
            body = _credited("We Work Remotely", html_to_text(item.findtext("description") or ""))
            yield Posting(item.findtext("link") or "", posting_text(title, location, body, company))


def hacker_news(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Each top-level reply to the latest "Who is hiring?" thread is one employer's post.

    Its first line usually reads "Company | Role | Location | ...", so that line is used
    as both the title and the location for filtering.
    """
    stories = client.get(
        "https://hn.algolia.com/api/v1/search_by_date",
        tags="story,author_whoishiring",
        hitsPerPage="10",
    ).json()
    story = next(
        (
            hit
            for hit in stories.get("hits", [])
            if "who is hiring" in str(hit.get("title", "")).lower()
        ),
        None,
    )
    if story is None or not re.fullmatch(r"\d{1,12}", str(story.get("objectID", ""))):
        return
    thread = client.get(f"https://hn.algolia.com/api/v1/items/{story['objectID']}").json()
    for reply in thread.get("children", []):
        text = reply.get("text") or ""
        if not text or not isinstance(reply.get("id"), int):
            continue
        first_line = html_to_text(text.split("<p>", 1)[0])[:HN_MAX_TITLE_CHARS]
        if matches(first_line, first_line, query):
            company = first_line.split("|", 1)[0].strip()
            body = _credited("Hacker News: Who is hiring?", html_to_text(text))
            url = f"https://news.ycombinator.com/item?id={reply['id']}"
            yield Posting(url, posting_text(first_line, first_line, body, company))


def get_on_board(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Mostly Latin America. Spanish and Portuguese postings are common."""
    data = client.get(
        "https://www.getonbrd.com/api/v0/search/jobs",
        query=query.search or "",
        per_page=str(GET_ON_BOARD_PAGE_SIZE),
    ).json()
    for job in data.get("data", []):
        fields = job.get("attributes") or {}
        title = fields.get("title", "")
        location = ", ".join(str(country) for country in fields.get("countries") or [])
        if fields.get("remote"):
            location = f"{location}; Remote" if location else "Remote"
        if matches(title, location, query):
            body = _credited(
                "Get on Board",
                "\n\n".join(
                    html_to_text(fields.get(key) or "")
                    for key in ("description", "functions", "desirable")
                ),
            )
            url = (job.get("links") or {}).get("public_url", "")
            yield Posting(url, posting_text(title, location, body))


def landing_jobs(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Technology jobs, mostly in Portugal and the rest of Europe."""
    for page in range(LANDING_JOBS_MAX_PAGES):
        offset = str(page * LANDING_JOBS_PAGE_SIZE)
        jobs = client.get(
            "https://landing.jobs/api/v1/jobs", limit=str(LANDING_JOBS_PAGE_SIZE), offset=offset
        ).json()
        for job in jobs:
            title = job.get("title", "")
            places = [
                join(place.get("city"), place.get("country_code"))
                for place in job.get("locations") or []
                if isinstance(place, dict)
            ]
            if job.get("remote"):
                places.append("Remote")
            location = "; ".join(place for place in places if place)
            if matches(title, location, query):
                body = _credited(
                    "Landing.jobs",
                    "\n\n".join(
                        html_to_text(job.get(key) or "")
                        for key in ("role_description", "main_requirements", "nice_to_have")
                    ),
                )
                yield Posting(job["url"], posting_text(title, location, body))
        if len(jobs) < LANDING_JOBS_PAGE_SIZE:
            break


def jobtech(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Sweden's public employment service. Many postings are in Swedish."""
    data = client.get(
        "https://jobsearch.api.jobtechdev.se/search", q=query.search or "", limit="100"
    ).json()
    for job in data.get("hits", []):
        title = job.get("headline", "")
        place = job.get("workplace_address") or {}
        location = join(place.get("municipality"), place.get("region"), place.get("country"))
        if matches(title, location, query):
            body = _credited("Arbetsförmedlingen", (job.get("description") or {}).get("text", ""))
            company = (job.get("employer") or {}).get("name", "")
            yield Posting(job["webpage_url"], posting_text(title, location, body, company))


def four_day_week(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Jobs with a four-day week or reduced hours."""
    for page in range(1, FOUR_DAY_WEEK_MAX_PAGES + 1):
        data = client.get("https://4dayweek.io/api", page=str(page)).json()
        for job in data.get("jobs", []):
            title = job.get("title", "")
            location = job.get("location_original") or join(
                job.get("location_city"), job.get("location_country")
            )
            if job.get("is_remote") and "remote" not in location.lower():
                location = f"{location}; Remote" if location else "Remote"
            if matches(title, location, query):
                body = _credited("4 Day Week", html_to_text(job.get("description", "")))
                company = job.get("company_name", "")
                yield Posting(job["url"], posting_text(title, location, body, company))
        if not data.get("has_more"):
            break
