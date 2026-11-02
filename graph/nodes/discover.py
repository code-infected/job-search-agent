"""Discovery: pulls postings from the curated query list. Makes no model call."""

import html
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph.state import JobSearchState
from memory.store import MemoryStore

logger = logging.getLogger(__name__)

MAX_POSTING_CHARS = 20_000
MAX_ATTEMPTS = 4
MAX_RETRY_WAIT_SECONDS = 60.0
USER_AGENT = "job-search-agent/1.0"


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["greenhouse", "lever", "url"]
    board: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    url: Optional[str] = Field(default=None, pattern=r"^https?://")
    keywords: list[str] = []
    exclude_keywords: list[str] = []
    locations: list[str] = []

    @model_validator(mode="after")
    def has_target(self) -> "Query":
        if self.source == "url" and not self.url:
            raise ValueError("url queries need a url")
        if self.source != "url" and not self.board:
            raise ValueError(f"{self.source} queries need a board")
        return self


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_profile: str = Field(min_length=1)
    queries: list[Query] = Field(min_length=1)


def load_search_config(path: Path) -> SearchConfig:
    return SearchConfig.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Posting:
    url: str
    text: str


def html_to_text(markup: str) -> str:
    soup = BeautifulSoup(markup, "html.parser")
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
    ) -> None:
        self._client = client or httpx.Client(
            timeout=20.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._delay = delay_seconds
        self._sleep = sleep
        self._last_request = 0.0

    def get(self, url: str, **params: str) -> httpx.Response:
        for attempt in range(MAX_ATTEMPTS):
            wait = self._delay - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            response = self._client.get(url, params=params)
            if response.status_code != 429 and response.status_code < 500:
                break
            if attempt + 1 < MAX_ATTEMPTS:
                self._sleep(min(self._retry_wait(response, attempt), MAX_RETRY_WAIT_SECONDS))
        response.raise_for_status()
        return response

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
    return not query.locations or any(word.lower() in place for word in query.locations)


def _greenhouse(client: PoliteClient, query: Query) -> Iterator[Posting]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{query.board}/jobs"
    for job in client.get(url, content="true").json().get("jobs", []):
        title = job.get("title", "")
        location = (job.get("location") or {}).get("name", "")
        if _matches(title, location, query):
            body = html_to_text(html.unescape(job.get("content", "")))
            yield Posting(job["absolute_url"], f"{title}\nLocation: {location}\n\n{body}")


def _lever(client: PoliteClient, query: Query) -> Iterator[Posting]:
    url = f"https://api.lever.co/v0/postings/{query.board}"
    for job in client.get(url, mode="json").json():
        title = job.get("text", "")
        location = (job.get("categories") or {}).get("location", "")
        if _matches(title, location, query):
            body = job.get("descriptionPlain", "")[:MAX_POSTING_CHARS]
            yield Posting(job["hostedUrl"], f"{title}\nLocation: {location}\n\n{body}")


def _page(client: PoliteClient, query: Query) -> Iterator[Posting]:
    yield Posting(query.url, html_to_text(client.get(query.url).text))


FETCHERS = {"greenhouse": _greenhouse, "lever": _lever, "url": _page}


def fetch_postings(queries: list[Query], client: PoliteClient) -> Iterator[Posting]:
    """Yield postings lazily. One failing source is logged and does not stop the others."""
    for query in queries:
        try:
            yield from FETCHERS[query.source](client, query)
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("source %s failed: %s", query.board or query.url, exc)


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
