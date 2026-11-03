"""Discovery: pulls postings from the curated query list. Makes no model call."""

import html
import ipaddress
import json
import logging
import os
import re
import socket
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph.state import JobSearchState
from memory.store import MemoryStore

logger = logging.getLogger(__name__)

MAX_POSTING_CHARS = 20_000
MAX_MARKUP_CHARS = 1_000_000
MAX_RESPONSE_BYTES = 50_000_000
MAX_URL_CHARS = 2000
REQUEST_DEADLINE_SECONDS = 90.0
MAX_ATTEMPTS = 4
MAX_RETRY_WAIT_SECONDS = 60.0
USER_AGENT = "job-search-agent/1.0"


# A company's own job board on an applicant tracking system, addressed by its board name.
BOARD_SOURCES = ("greenhouse", "lever", "ashby", "workable", "smartrecruiters")
# Keyword search across many employers.
SEARCH_SOURCES = ("remotive", "adzuna")
FILTER_FIELDS = ("keywords", "exclude_keywords", "locations")
# A posting open to anyone, anywhere, passes any location filter.
ANYWHERE = ("worldwide", "anywhere")
REMOTIVE_MIN_INTERVAL_SECONDS = 6 * 3600  # Remotive asks for at most four fetches a day
ADZUNA_PAGE_SIZE = 50
SMARTRECRUITERS_PAGE_SIZE = 100
SMARTRECRUITERS_MAX_PAGES = 10


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "greenhouse", "lever", "ashby", "workable", "smartrecruiters", "remotive", "adzuna", "url"
    ]
    board: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$", max_length=100)
    url: Optional[str] = Field(default=None, pattern=r"^https?://", max_length=MAX_URL_CHARS)
    search: Optional[str] = Field(default=None, min_length=1, max_length=200)
    country: Optional[str] = Field(default=None, pattern=r"^[a-z]{2}$")
    keywords: list[str] = []
    exclude_keywords: list[str] = []
    locations: list[str] = []

    @model_validator(mode="after")
    def has_target(self) -> "Query":
        if self.source == "url" and not self.url:
            raise ValueError("url queries need a url")
        if self.source in BOARD_SOURCES and not self.board:
            raise ValueError(f"{self.source} queries need a board")
        if self.source in SEARCH_SOURCES and not self.search:
            raise ValueError(f"{self.source} queries need search terms")
        if self.source == "adzuna" and not self.country:
            raise ValueError("adzuna queries need a two-letter country code")
        return self


class SearchConfig(BaseModel):
    """What to look for. Top-level filters apply to every query that does not set its own."""

    model_config = ConfigDict(extra="forbid")

    target_profile: str = Field(min_length=1, max_length=4000)
    keywords: list[str] = []
    exclude_keywords: list[str] = []
    locations: list[str] = []
    queries: list[Query] = []

    def resolved_queries(self) -> list[Query]:
        return [
            query.model_copy(
                update={
                    field: getattr(query, field) or getattr(self, field) for field in FILTER_FIELDS
                }
            )
            for query in self.queries
        ]


def load_search_config(path: Path) -> SearchConfig:
    return SearchConfig.model_validate_json(path.read_text(encoding="utf-8"))


class FetchLog:
    """Remembers when a rate-limited source was last fetched, across processes."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock

    def allow(self, source: str, min_interval_seconds: float) -> bool:
        try:
            fetched = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            fetched = {}
        now = self._clock()
        if source in fetched and now - float(fetched[source]) < min_interval_seconds:
            return False
        fetched[source] = now
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(fetched), encoding="utf-8")
        return True


@dataclass(frozen=True)
class Posting:
    url: str
    text: str


def is_safe_posting_url(url: object) -> bool:
    """A posting URL comes from a remote board and is later written into the draft file."""
    if not isinstance(url, str) or len(url) > MAX_URL_CHARS:
        return False
    if re.search(r"[\s\x00-\x1f\x7f]", url):
        return False
    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _posting_text(title: object, location: object, body: str, company: object = "") -> str:
    header = f"{title}\nLocation: {location}\nCompany: {company}"
    return f"{header}\n\n{body}"[:MAX_POSTING_CHARS]


def reject_internal_address(request: httpx.Request) -> None:
    """Runs on every hop, so a public page cannot redirect the fetch to an internal address."""
    host = request.url.host
    addresses = {info[4][0] for info in socket.getaddrinfo(host, request.url.port or 443)}
    if not addresses or not all(ipaddress.ip_address(addr).is_global for addr in addresses):
        raise ValueError(f"refusing to fetch non-public address: {host}")


def html_to_text(markup: str) -> str:
    soup = BeautifulSoup(markup[:MAX_MARKUP_CHARS], "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))[:MAX_POSTING_CHARS]


class PoliteClient:
    """HTTP GET with a minimum delay between requests and backoff on 429 and 5xx."""

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            event_hooks={"request": [reject_internal_address]},
        )
        self._delay = delay_seconds
        self._sleep = sleep
        self._attempts = attempts
        self._last_request = 0.0

    def get(self, url: str, **params: str) -> httpx.Response:
        for attempt in range(self._attempts):
            wait = self._delay - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            response = self._fetch(url, params)
            if response.status_code != 429 and response.status_code < 500:
                break
            if attempt + 1 < self._attempts:
                self._sleep(min(self._retry_wait(response, attempt), MAX_RETRY_WAIT_SECONDS))
        response.raise_for_status()
        return response

    def _fetch(self, url: str, params: dict[str, str]) -> httpx.Response:
        """Read a capped body under an overall deadline, which per-read timeouts do not give."""
        deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
        body = bytearray()
        with self._client.stream("GET", url, params=params) as response:
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError(f"response too large: {url}")
                if time.monotonic() > deadline:
                    raise ValueError(f"response too slow: {url}")
            kept = ("content-type", "retry-after")
            headers = {key: value for key, value in response.headers.items() if key in kept}
            return httpx.Response(
                response.status_code,
                headers=headers,
                content=bytes(body),
                request=response.request,
            )

    @staticmethod
    def _retry_wait(response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("Retry-After", "")
        return float(header) if header.isdigit() else float(2 ** (attempt + 1))


def _matches(title: str, location: str, query: Query) -> bool:
    """Filter before any model call, so a posting the user cannot take costs nothing."""
    lowered = title.lower()
    if any(word.lower() in lowered for word in query.exclude_keywords):
        return False
    if query.keywords and not any(word.lower() in lowered for word in query.keywords):
        return False
    place = location.lower()
    if not query.locations or any(word in place for word in ANYWHERE):
        return True
    return any(word.lower() in place for word in query.locations)


def _join(*parts: object) -> str:
    return ", ".join(str(part) for part in parts if part)


def _greenhouse(client: PoliteClient, query: Query) -> Iterator[Posting]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{query.board}/jobs"
    for job in client.get(url, content="true").json().get("jobs", []):
        title = job.get("title", "")
        location = (job.get("location") or {}).get("name", "")
        if _matches(title, location, query):
            body = html_to_text(html.unescape(job.get("content", "")))
            text = _posting_text(title, location, body, query.board)
            yield Posting(job["absolute_url"], text)


def _lever(client: PoliteClient, query: Query) -> Iterator[Posting]:
    url = f"https://api.lever.co/v0/postings/{query.board}"
    for job in client.get(url, mode="json").json():
        title = job.get("text", "")
        location = (job.get("categories") or {}).get("location", "")
        if _matches(title, location, query):
            body = job.get("descriptionPlain", "")
            yield Posting(job["hostedUrl"], _posting_text(title, location, body, query.board))


def _ashby(client: PoliteClient, query: Query) -> Iterator[Posting]:
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
        if _matches(title, location, query):
            body = job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml", ""))
            yield Posting(job["jobUrl"], _posting_text(title, location, body, query.board))


def _workable(client: PoliteClient, query: Query) -> Iterator[Posting]:
    url = f"https://apply.workable.com/api/v1/widget/accounts/{query.board}"
    data = client.get(url, details="true").json()
    company = data.get("name") or query.board
    for job in data.get("jobs", []):
        title = job.get("title", "")
        places = [
            _join(place.get("city"), place.get("region"), place.get("country"))
            for place in job.get("locations") or []
            if isinstance(place, dict)
        ] or [_join(job.get("city"), job.get("state"), job.get("country"))]
        if job.get("telecommuting"):
            places.append("Remote")
        location = "; ".join(place for place in places if place)
        if _matches(title, location, query):
            body = html_to_text(job.get("description", ""))
            yield Posting(job["url"], _posting_text(title, location, body, company))


def _smartrecruiters(client: PoliteClient, query: Query) -> Iterator[Posting]:
    """The list holds no description, so only postings that pass the filters are opened."""
    base = f"https://api.smartrecruiters.com/v1/companies/{query.board}/postings"
    for page in range(SMARTRECRUITERS_MAX_PAGES):
        offset = page * SMARTRECRUITERS_PAGE_SIZE
        data = client.get(base, limit=str(SMARTRECRUITERS_PAGE_SIZE), offset=str(offset)).json()
        for job in data.get("content", []):
            place = job.get("location") or {}
            location = place.get("fullLocation") or _join(place.get("city"), place.get("country"))
            location = re.sub(r"(?:\s*,)+\s*", ", ", location).strip(" ,")  # drop empty parts
            if place.get("remote"):
                location = f"{location}; Remote" if location else "Remote"
            title = job.get("name", "")
            if not _matches(title, location, query):
                continue
            if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", str(job.get("id", ""))):
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
            yield Posting(url, _posting_text(title, location, body, company))
        if offset + SMARTRECRUITERS_PAGE_SIZE >= data.get("totalFound", 0):
            break


def _remotive(client: PoliteClient, query: Query) -> Iterator[Posting]:
    """Remote jobs from many employers. Remotive's terms ask that it is credited and linked."""
    data = client.get("https://remotive.com/api/remote-jobs", search=query.search).json()
    for job in data.get("jobs", []):
        title = job.get("title", "")
        location = f"Remote ({job.get('candidate_required_location') or 'Worldwide'})"
        if _matches(title, location, query):
            body = f"Found through Remotive.\n\n{html_to_text(job.get('description', ''))}"
            yield Posting(job["url"], _posting_text(title, location, body, job.get("company_name")))


def _adzuna(client: PoliteClient, query: Query) -> Iterator[Posting]:
    """Keyword search across employers in one country. Adzuna returns only a summary."""
    app_id = os.environ.get("ADZUNA_APP_ID", "")
    app_key = os.environ.get("ADZUNA_APP_KEY", "")
    if not (app_id and app_key):
        raise ValueError("Adzuna needs ADZUNA_APP_ID and ADZUNA_APP_KEY")
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
        try:
            results = client.get(url, **params).json().get("results", [])
        except (httpx.HTTPError, ValueError) as exc:
            # The request URL carries the key, so report the failure without it.
            status = getattr(getattr(exc, "response", None), "status_code", type(exc).__name__)
            raise ValueError(f"Adzuna request failed: {status}") from None
        for job in results:
            title = html_to_text(job.get("title", ""))
            location = (job.get("location") or {}).get("display_name", "")
            if _matches(title, location, query):
                summary = html_to_text(job.get("description", ""))
                body = f"{summary}\n\nOnly a short summary of this posting was available."
                company = (job.get("company") or {}).get("display_name", "")
                yield Posting(job["redirect_url"], _posting_text(title, location, body, company))
        if len(results) < ADZUNA_PAGE_SIZE:
            break


def _page(client: PoliteClient, query: Query) -> Iterator[Posting]:
    yield Posting(query.url, html_to_text(client.get(query.url).text))


FETCHERS = {
    "greenhouse": _greenhouse,
    "lever": _lever,
    "ashby": _ashby,
    "workable": _workable,
    "smartrecruiters": _smartrecruiters,
    "remotive": _remotive,
    "adzuna": _adzuna,
    "url": _page,
}


def describe(query: Query) -> str:
    return query.board or query.search or query.url or query.source


def fetch_postings(
    queries: list[Query], client: PoliteClient, fetch_log: Optional[FetchLog] = None
) -> Iterator[Posting]:
    """Yield postings lazily. One failing source is logged and does not stop the others."""
    remotive_allowed: Optional[bool] = None
    for query in queries:
        if query.source == "remotive" and fetch_log is not None:
            if remotive_allowed is None:
                remotive_allowed = fetch_log.allow("remotive", REMOTIVE_MIN_INTERVAL_SECONDS)
            if not remotive_allowed:
                logger.info("skipping Remotive: fetched within the last six hours")
                continue
        try:
            for posting in FETCHERS[query.source](client, query):
                if is_safe_posting_url(posting.url):
                    yield posting
                else:
                    logger.warning("skipping posting with unsafe URL: %r", posting.url)
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as exc:
            logger.error("source %s %s failed: %r", query.source, describe(query), exc)


def make_discover(postings: Iterator[Posting], store: MemoryStore):
    def discover(state: JobSearchState) -> dict:
        for posting in postings:
            if not posting.text.strip():
                logger.warning("skipping empty posting %s", posting.url)
            elif not store.has_url(posting.url):
                store.claim(posting.url, posting.text, state["run_id"])
                return {"posting_url": posting.url, "posting_text": posting.text}
        return {"posting_url": "", "posting_text": ""}

    return discover
