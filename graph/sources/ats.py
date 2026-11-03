"""Company job boards on applicant tracking systems, each addressed by its board name."""

import html
import logging
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from graph.sources.common import (
    PoliteClient,
    Posting,
    html_to_text,
    join,
    matches,
    place_matches,
    posting_text,
    title_matches,
)

if TYPE_CHECKING:
    from graph.nodes.discover import Query

logger = logging.getLogger(__name__)

SMARTRECRUITERS_PAGE_SIZE = 100
SMARTRECRUITERS_MAX_PAGES = 10
RIPPLING_MAX_PAGES = 10
# Job ids are put into detail URLs, so anything unexpected is skipped.
SAFE_ID = re.compile(r"[A-Za-z0-9-]{1,64}")

SLUG = r"([A-Za-z0-9_-]{1,100})"
SUBDOMAIN = r"([a-z0-9][a-z0-9-]{0,62})"
# Path segments and subdomains that belong to the board software, not to a company.
RESERVED = frozenset(
    {"embed", "j", "api", "jobs", "v0", "v1", "job_board", "boards", "www", "app", "help", "blog"}
)
URL_PATTERNS = (
    ("greenhouse", re.compile(r"greenhouse\.io/embed/job_board(?:/js)?\?for=" + SLUG)),
    ("greenhouse", re.compile(r"(?:job-boards|boards)(?:\.eu)?\.greenhouse\.io/" + SLUG)),
    ("lever", re.compile(r"jobs\.lever\.co/" + SLUG)),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/" + SLUG)),
    ("workable", re.compile(r"apply\.workable\.com/" + SLUG)),
    ("smartrecruiters", re.compile(r"(?:jobs|careers)\.smartrecruiters\.com/" + SLUG)),
    ("recruitee", re.compile(r"//" + SUBDOMAIN + r"\.recruitee\.com")),
    ("breezy", re.compile(r"//" + SUBDOMAIN + r"\.breezy\.hr")),
    ("rippling", re.compile(r"ats\.rippling\.com/" + SLUG)),
    ("gem", re.compile(r"jobs\.gem\.com/" + SLUG)),
)
WORKDAY_LINK = re.compile(
    r"https://[a-z0-9-]{1,63}\.wd\d{1,3}\.myworkdayjobs\.com/(?:[a-z]{2}-[A-Z]{2}/)?[A-Za-z0-9_-]{1,100}"
)


def workday_links(text: str) -> list[str]:
    """Workday careers sites in a URL or page markup, as their base links."""
    return list(dict.fromkeys(WORKDAY_LINK.findall(text)))


def boards_in_text(text: str) -> list[tuple[str, str]]:
    """Board references in a URL or a page's markup, in order, without duplicates."""
    found: dict[tuple[str, str], None] = {}
    for source, pattern in URL_PATTERNS:
        for slug in pattern.findall(text):
            if slug.lower() not in RESERVED:
                found[(source, slug)] = None
    return list(found)


def greenhouse(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{query.board}/jobs"
    for job in client.get(url, content="true").json().get("jobs", []):
        title = job.get("title", "")
        location = (job.get("location") or {}).get("name", "")
        if matches(title, location, query):
            body = html_to_text(html.unescape(job.get("content", "")))
            yield Posting(job["absolute_url"], posting_text(title, location, body, query.board))


def greenhouse_light(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """Reads titles first and opens only matching postings, for scanning many boards."""
    base = f"https://boards-api.greenhouse.io/v1/boards/{query.board}/jobs"
    for job in client.get(base).json().get("jobs", []):
        title = job.get("title", "")
        location = (job.get("location") or {}).get("name", "")
        if not matches(title, location, query) or not SAFE_ID.fullmatch(str(job.get("id", ""))):
            continue
        detail = client.get(f"{base}/{job['id']}").json()
        body = html_to_text(html.unescape(detail.get("content", "")))
        yield Posting(job["absolute_url"], posting_text(title, location, body, query.board))


def lever(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    url = f"https://api.lever.co/v0/postings/{query.board}"
    for job in client.get(url, mode="json").json():
        title = job.get("text", "")
        location = (job.get("categories") or {}).get("location", "")
        if matches(title, location, query):
            body = job.get("descriptionPlain", "")
            yield Posting(job["hostedUrl"], posting_text(title, location, body, query.board))


def ashby(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    url = f"https://api.ashbyhq.com/posting-api/job-board/{query.board}"
    for job in client.get(url).json().get("jobs", []):
        if job.get("isListed") is False:
            continue
        title = job.get("title", "")
        places = [job.get("location") or ""]
        places += [
            place.get("location", "")
            for place in job.get("secondaryLocations") or []
            if isinstance(place, dict)
        ]
        if job.get("isRemote"):
            places.append("Remote")
        location = "; ".join(place for place in places if place)
        if matches(title, location, query):
            body = job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml", ""))
            yield Posting(job["jobUrl"], posting_text(title, location, body, query.board))


def workable(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    url = f"https://apply.workable.com/api/v1/widget/accounts/{query.board}"
    data = client.get(url, details="true").json()
    company = data.get("name") or query.board
    for job in data.get("jobs", []):
        title = job.get("title", "")
        places = [
            join(place.get("city"), place.get("region"), place.get("country"))
            for place in job.get("locations") or []
            if isinstance(place, dict)
        ] or [join(job.get("city"), job.get("state"), job.get("country"))]
        if job.get("telecommuting"):
            places.append("Remote")
        location = "; ".join(place for place in places if place)
        if matches(title, location, query):
            body = html_to_text(job.get("description", ""))
            yield Posting(job["url"], posting_text(title, location, body, company))


def smartrecruiters(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """The list holds no description, so only postings that pass the filters are opened."""
    base = f"https://api.smartrecruiters.com/v1/companies/{query.board}/postings"
    for page in range(SMARTRECRUITERS_MAX_PAGES):
        offset = page * SMARTRECRUITERS_PAGE_SIZE
        data = client.get(base, limit=str(SMARTRECRUITERS_PAGE_SIZE), offset=str(offset)).json()
        for job in data.get("content", []):
            place = job.get("location") or {}
            location = place.get("fullLocation") or join(place.get("city"), place.get("country"))
            location = re.sub(r"(?:\s*,)+\s*", ", ", location).strip(" ,")  # drop empty parts
            if place.get("remote"):
                location = f"{location}; Remote" if location else "Remote"
            title = job.get("name", "")
            if not matches(title, location, query):
                continue
            if not SAFE_ID.fullmatch(str(job.get("id", ""))):
                logger.warning("skipping SmartRecruiters posting with an unexpected id")
                continue
            detail = client.get(f"{base}/{job['id']}").json()
            sections = (detail.get("jobAd") or {}).get("sections") or {}
            body = "\n\n".join(
                html_to_text(section.get("text", ""))
                for key in ("jobDescription", "qualifications", "additionalInformation")
                if isinstance(section := sections.get(key), dict)
            )
            company = (job.get("company") or {}).get("name") or query.board
            url = (
                detail.get("postingUrl")
                or f"https://jobs.smartrecruiters.com/{query.board}/{job['id']}"
            )
            yield Posting(url, posting_text(title, location, body, company))
        if offset + SMARTRECRUITERS_PAGE_SIZE >= data.get("totalFound", 0):
            break


def recruitee(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    url = f"https://{query.board.lower()}.recruitee.com/api/offers/"
    for job in client.get(url).json().get("offers", []):
        if job.get("status", "published") != "published":
            continue
        title = job.get("title", "")
        location = job.get("location") or join(job.get("city"), job.get("country"))
        if job.get("remote"):
            location = f"{location}; Remote" if location else "Remote"
        if matches(title, location, query):
            body = "\n\n".join(
                html_to_text(job.get(key) or "") for key in ("description", "requirements")
            )
            company = job.get("company_name") or query.board
            yield Posting(job["careers_url"], posting_text(title, location, body, company))


def breezy(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """The list holds no description, so only matching postings are opened."""
    for job in client.get(f"https://{query.board.lower()}.breezy.hr/json").json():
        place = job.get("location") or {}
        location = place.get("name") or join(
            place.get("city"),
            (place.get("state") or {}).get("name"),
            (place.get("country") or {}).get("name"),
        )
        if place.get("is_remote"):
            location = f"{location}; Remote" if location else "Remote"
        title = job.get("name", "")
        # The posting page is opened, so it must be on Breezy itself.
        if not (urlsplit(str(job.get("url", ""))).hostname or "").endswith(".breezy.hr"):
            continue
        if matches(title, location, query):
            body = html_to_text(client.get(job["url"]).text)
            company = (job.get("company") or {}).get("name") or query.board
            yield Posting(job["url"], posting_text(title, location, body, company))


def rippling(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """The list holds no description, so only matching postings are opened."""
    base = f"https://ats.rippling.com/api/v2/board/{query.board}/jobs"
    for page in range(RIPPLING_MAX_PAGES):
        data = client.get(base, page=str(page)).json()
        for job in data.get("items", []):
            places = [p for p in job.get("locations") or [] if isinstance(p, dict)]
            location = "; ".join(p.get("name", "") for p in places if p.get("name"))
            if any(p.get("workplaceType") == "REMOTE" for p in places):
                location = f"{location}; Remote" if location else "Remote"
            title = job.get("name", "")
            if not matches(title, location, query) or not SAFE_ID.fullmatch(str(job.get("id"))):
                continue
            detail = client.get(f"{base}/{job['id']}").json()
            parts = (detail.get("description") or {}).values()
            body = "\n\n".join(html_to_text(part) for part in parts if isinstance(part, str))
            company = detail.get("companyName") or query.board
            yield Posting(job["url"], posting_text(title, location, body, company))
        if page + 1 >= data.get("totalPages", 0):
            break


WORKDAY_HOST = re.compile(r"([a-z0-9-]{1,63})\.(wd\d{1,3})\.myworkdayjobs\.com")
WORKDAY_PAGE_SIZE = 20
WORKDAY_MAX_PAGES = 10
SAFE_WORKDAY_PATH = re.compile(r"/job/[A-Za-z0-9._~%/-]{1,500}")


def workday_site(url: str) -> tuple[str, str, str]:
    """Host, tenant, and site from a careers link such as https://acme.wd5.myworkdayjobs.com/Careers."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    match = WORKDAY_HOST.fullmatch(host)
    segments = [s for s in parts.path.split("/") if s and not re.fullmatch(r"[a-z]{2}-[A-Z]{2}", s)]
    if not match or not segments or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", segments[0]):
        raise ValueError(
            "not a Workday careers link, which looks like https://acme.wd5.myworkdayjobs.com/Careers"
        )
    return host, match.group(1), segments[0]


def workday(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    """A company's Workday careers site, read through the same API its own page uses."""
    host, tenant, site = workday_site(query.url)
    base = f"https://{host}/wday/cxs/{tenant}/{site}"
    for page in range(WORKDAY_MAX_PAGES):
        search = {
            "appliedFacets": {},
            "limit": WORKDAY_PAGE_SIZE,
            "offset": page * WORKDAY_PAGE_SIZE,
            "searchText": query.search or "",
        }
        data = client.request("POST", f"{base}/jobs", json=search).json()
        jobs = data.get("jobPostings", [])
        for job in jobs:
            title = job.get("title", "")
            listed = job.get("locationsText", "")
            # "3 Locations" names none of them, so the location is checked on the full posting.
            named = not re.fullmatch(r"\d+ Locations?", listed)
            if not title_matches(title, query) or (named and not place_matches(listed, query)):
                continue
            path = str(job.get("externalPath", ""))
            if not SAFE_WORKDAY_PATH.fullmatch(path):
                continue
            detail = client.get(f"{base}{path}").json().get("jobPostingInfo") or {}
            places = [detail.get("location") or listed]
            places += [str(place) for place in detail.get("additionalLocations") or []]
            if detail.get("remoteType"):
                places.append(str(detail["remoteType"]))
            location = "; ".join(place for place in places if place)
            if not named and not place_matches(location, query):
                continue
            body = html_to_text(detail.get("jobDescription", ""))
            url = detail.get("externalUrl") or f"https://{host}/{site}{path}"
            yield Posting(url, posting_text(title, location, body, tenant))
        if (page + 1) * WORKDAY_PAGE_SIZE >= data.get("total", 0) or not jobs:
            break


def gem(client: PoliteClient, query: "Query") -> Iterator[Posting]:
    url = f"https://api.gem.com/job_board/v0/{query.board}/job_posts/"
    for job in client.get(url).json():
        title = job.get("title", "")
        location = (job.get("location") or {}).get("name", "")
        if job.get("location_type") == "remote":
            location = f"{location}; Remote" if location else "Remote"
        if matches(title, location, query):
            body = job.get("content_plain") or html_to_text(job.get("content", ""))
            yield Posting(job["absolute_url"], posting_text(title, location, body, query.board))
