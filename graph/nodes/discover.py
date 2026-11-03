"""Discovery: pulls postings from the configured sources. Makes no model call."""

import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from graph.sources import BOARD_SOURCES, SITE_SOURCES, SOURCES
from graph.sources.ats import workday_site
from graph.sources.common import (
    MAX_MARKUP_CHARS,
    MAX_POSTING_CHARS,
    MAX_RESPONSE_BYTES,
    MAX_URL_CHARS,
    PoliteClient,
    Posting,
    html_to_text,
    is_safe_posting_url,
    reject_internal_address,
)
from graph.sources.directory import SYSTEMS as DIRECTORY_SYSTEMS
from graph.sources.directory import BoardDirectory
from graph.state import JobSearchState
from memory.store import MemoryStore

__all__ = [
    "BOARD_SOURCES",
    "DIRECTORY_SYSTEMS",
    "MAX_MARKUP_CHARS",
    "MAX_POSTING_CHARS",
    "MAX_RESPONSE_BYTES",
    "SITE_SOURCES",
    "FetchLog",
    "PoliteClient",
    "Posting",
    "Query",
    "SearchConfig",
    "describe",
    "fetch_postings",
    "html_to_text",
    "is_safe_posting_url",
    "load_search_config",
    "make_discover",
    "reject_internal_address",
]

logger = logging.getLogger(__name__)

FILTER_FIELDS = ("keywords", "exclude_keywords", "locations")
MAX_TERMS = 10
DEFAULT_BOARDS_PER_RUN = 300
Term = Field(min_length=1, max_length=200)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    board: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$", max_length=100)
    url: Optional[str] = Field(default=None, pattern=r"^https?://", max_length=MAX_URL_CHARS)
    search: Optional[str] = Field(default=None, min_length=1, max_length=200)
    country: Optional[str] = Field(default=None, pattern=r"^[a-z]{2}$")
    where: Optional[str] = Field(default=None, min_length=1, max_length=100)
    systems: list[str] = []
    boards_per_run: int = Field(default=DEFAULT_BOARDS_PER_RUN, ge=1, le=5000)
    keywords: list[str] = []
    exclude_keywords: list[str] = []
    locations: list[str] = []

    @model_validator(mode="after")
    def has_target(self) -> "Query":
        spec = SOURCES.get(self.source)
        if spec is None:
            raise ValueError(f"unknown source {self.source!r}")
        if spec.kind == "url" and not self.url:
            raise ValueError("url queries need a url")
        if spec.kind == "link":
            if not self.url:
                raise ValueError(f"{self.source} queries need a careers link")
            workday_site(self.url)
        if spec.kind == "board" and not self.board:
            raise ValueError(f"{self.source} queries need a board")
        if spec.search == "required" and not self.search:
            raise ValueError(f"{self.source} queries need search terms")
        if spec.needs_country and not self.country:
            raise ValueError(f"{self.source} queries need a two-letter country code")
        if spec.kind == "directory":
            if not self.systems or not set(self.systems) <= set(DIRECTORY_SYSTEMS):
                raise ValueError(f"directory queries need systems from {DIRECTORY_SYSTEMS}")
            if not self.keywords:
                raise ValueError("scanning every company needs title keywords")
        return self


class SearchConfig(BaseModel):
    """What to look for. Top-level filters apply to every query that does not set its own.

    Job sites and directory systems named here are expanded into queries on every run,
    so the same search terms reach every site without repeating them per site.
    """

    model_config = ConfigDict(extra="forbid")

    target_profile: str = Field(min_length=1, max_length=4000)
    keywords: list[str] = []
    exclude_keywords: list[str] = []
    locations: list[str] = []
    queries: list[Query] = []
    job_sites: list[str] = []
    search_terms: list[str] = Field(default=[], max_length=MAX_TERMS)
    country: Optional[str] = Field(default=None, pattern=r"^[a-z]{2}$")
    where: Optional[str] = Field(default=None, min_length=1, max_length=100)
    scan_companies: list[str] = []
    companies_per_run: int = Field(default=DEFAULT_BOARDS_PER_RUN, ge=10, le=5000)

    @field_validator("search_terms")
    @classmethod
    def terms_are_short(cls, terms: list[str]) -> list[str]:
        cleaned = [term.strip() for term in terms if term.strip()]
        if any(len(term) > 200 for term in cleaned):
            raise ValueError("each search term must be under 200 characters")
        return cleaned

    @model_validator(mode="after")
    def names_are_known(self) -> "SearchConfig":
        unknown = set(self.job_sites) - set(SITE_SOURCES)
        if unknown:
            raise ValueError(f"unknown job sites: {sorted(unknown)}")
        unknown = set(self.scan_companies) - set(DIRECTORY_SYSTEMS)
        if unknown:
            raise ValueError(f"unknown board systems: {sorted(unknown)}")
        if self.scan_companies and not self.keywords:
            raise ValueError(
                "scanning every company needs title keywords, or every job on those "
                "systems would be scored"
            )
        return self

    def has_sources(self) -> bool:
        return bool(self.queries or self.job_sites or self.scan_companies)

    def resolved_queries(self) -> list[Query]:
        return [self._inherit(query) for query in self.queries + self._expanded()]

    def _inherit(self, query: Query) -> Query:
        update = {field: getattr(query, field) or getattr(self, field) for field in FILTER_FIELDS}
        # A site without its own search returns everything, so the search terms filter titles.
        if SOURCES[query.source].search == "none" and SOURCES[query.source].kind == "site":
            update["keywords"] = update["keywords"] or self.search_terms
        return Query.model_validate({**query.model_dump(), **update})

    def _expanded(self) -> list[Query]:
        terms = self.search_terms or self.keywords[:MAX_TERMS]
        expanded = []
        for name in self.job_sites:
            spec = SOURCES[name]
            if spec.needs_country and not self.country:
                logger.warning("skipping %s: it needs a country", name)
                continue
            site_terms: list[Optional[str]]
            if spec.search == "none":
                site_terms = [None]
            elif terms:
                site_terms = list(terms)
            elif spec.search == "optional":
                site_terms = [None]
            else:
                logger.warning("skipping %s: it needs search terms", name)
                continue
            # Built from already validated settings, and checked again once filters apply.
            expanded += [
                Query.model_construct(
                    source=name, search=term, country=self.country, where=self.where
                )
                for term in site_terms
            ]
        if self.scan_companies:
            expanded.append(
                Query.model_construct(
                    source="directory",
                    systems=list(self.scan_companies),
                    boards_per_run=self.companies_per_run,
                )
            )
        return expanded


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


def _page(client: PoliteClient, query: Query) -> Iterator[Posting]:
    yield Posting(query.url, html_to_text(client.get(query.url).text))


def describe(query: Query) -> str:
    if query.source == "directory":
        return ", ".join(query.systems)
    return query.board or query.search or query.url or query.source


def fetch_postings(
    queries: list[Query],
    client: PoliteClient,
    fetch_log: Optional[FetchLog] = None,
    directory: Optional[BoardDirectory] = None,
) -> Iterator[Posting]:
    """Yield postings lazily. One failing source is logged and does not stop the others."""
    allowed: dict[str, bool] = {}
    for query in queries:
        spec = SOURCES[query.source]
        if spec.min_interval_seconds and fetch_log is not None:
            # Decided once per run, so several searches on one site all run or all wait.
            if query.source not in allowed:
                allowed[query.source] = fetch_log.allow(query.source, spec.min_interval_seconds)
            if not allowed[query.source]:
                logger.info("skipping %s: fetched recently", spec.label)
                continue
        if query.source == "directory":
            if directory is None:
                logger.warning("skipping the company directory: none is configured")
                continue
            postings = directory.scan(query)
        elif query.source == "url":
            postings = _page(client, query)
        else:
            postings = spec.fetch(client, query)
        try:
            for posting in postings:
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
