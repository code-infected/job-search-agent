"""Every company board on the major board systems, found in Common Crawl's open index.

No board system offers a search across all its companies, so the directory is built
from the public web crawl: every board page the crawl has seen names its company.
The list is stored locally, refreshed monthly, and scanned a slice per run, oldest
first and alternating between systems so each one sees about one request a second.
"""

import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from graph.sources import ats
from graph.sources.common import USER_AGENT, PoliteClient, Posting, reject_internal_address

if TYPE_CHECKING:
    from graph.nodes.discover import Query

logger = logging.getLogger(__name__)

INDEX_LIST_URL = "https://index.commoncrawl.org/collinfo.json"
# URL patterns in the crawl index that hold each system's company boards.
CRAWL_PATTERNS = {
    "greenhouse": ("boards.greenhouse.io/*", "job-boards.greenhouse.io/*"),
    "lever": ("jobs.lever.co/*",),
    "ashby": ("jobs.ashbyhq.com/*",),
    "workable": ("apply.workable.com/*",),
    "smartrecruiters": ("jobs.smartrecruiters.com/*",),
    "recruitee": ("*.recruitee.com",),
    "breezy": ("*.breezy.hr",),
}
SYSTEMS = tuple(CRAWL_PATTERNS)
# Scanning reads titles first where the system allows it, to keep each board cheap.
FETCHERS = {
    "greenhouse": ats.greenhouse_light,
    "lever": ats.lever,
    "ashby": ats.ashby,
    "workable": ats.workable,
    "smartrecruiters": ats.smartrecruiters,
    "recruitee": ats.recruitee,
    "breezy": ats.breezy,
}
REFRESH_SECONDS = 30 * 24 * 3600
RESCAN_SECONDS = 12 * 3600
MISSING_RETRY_SECONDS = 30 * 24 * 3600
MAX_BOARDS_PER_SYSTEM = 50_000
CRAWLS_TO_TRY = 3
PAGE_ATTEMPTS = 3
INDEX_DEADLINE_SECONDS = 300.0
SAVE_EVERY = 25
BOARD_ERRORS = (
    httpx.HTTPError,
    httpx.InvalidURL,
    OSError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
)


def index_client() -> PoliteClient:
    """Common Crawl's index answers slowly and throttles bursts."""
    http = httpx.Client(
        timeout=180.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        event_hooks={"request": [reject_internal_address]},
    )
    return PoliteClient(http, delay_seconds=2.0, deadline_seconds=INDEX_DEADLINE_SECONDS)


class BoardDirectory:
    def __init__(
        self,
        path: Path,
        client: Optional[PoliteClient] = None,
        index: Optional[PoliteClient] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._client = client or PoliteClient(attempts=2)
        self._index = index
        self._clock = clock
        self._data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("built", {})
        data.setdefault("boards", {})
        data.setdefault("scanned", {})
        data.setdefault("missing", {})
        return data

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data), encoding="utf-8")
        temporary.replace(self._path)

    def counts(self) -> dict[str, int]:
        return {system: len(self._data["boards"].get(system, [])) for system in SYSTEMS}

    def built_at(self, system: str) -> float:
        return float(self._data["built"].get(system, 0))

    # Building

    def refresh(self, systems: list[str], force: bool = False) -> dict[str, int]:
        """Read the crawl index for systems never read or read over a month ago."""
        built = self._data["built"]
        now = self._clock()
        stale = [
            system
            for system in systems
            if force or system not in built or now - built[system] > REFRESH_SECONDS
        ]
        if not stale:
            return self.counts()
        index = self._index or index_client()
        crawls = self._crawl_endpoints(index)
        for system in stale:
            found = self._read_crawls(index, crawls, system)
            if found:
                known = {board.lower(): board for board in self._data["boards"].get(system, [])}
                known.update({board.lower(): board for board in found})
                boards = sorted(known.values(), key=str.lower)[:MAX_BOARDS_PER_SYSTEM]
                self._data["boards"][system] = boards
                self._data["built"][system] = now
                logger.info("directory: %d %s boards", len(boards), system)
            else:
                logger.warning("directory: the crawl index returned no %s boards", system)
        self.save()
        return self.counts()

    @staticmethod
    def _crawl_endpoints(index: PoliteClient) -> list[str]:
        crawls = index.get(INDEX_LIST_URL).json()
        return [crawl["cdx-api"] for crawl in crawls[:CRAWLS_TO_TRY] if "cdx-api" in crawl]

    def _read_crawls(self, index: PoliteClient, crawls: list[str], system: str) -> set[str]:
        """The newest crawl that answers. Older ones are tried only if it fails."""
        for endpoint in crawls:
            found: set[str] = set()
            for pattern in CRAWL_PATTERNS[system]:
                found |= self._read_pattern(index, endpoint, pattern, system)
            if found:
                return found
        return set()

    @staticmethod
    def _read_pattern(index: PoliteClient, endpoint: str, pattern: str, system: str) -> set[str]:
        found: set[str] = set()
        size = {"url": pattern, "showNumPages": "true"}
        try:
            pages = int(index.request("GET", endpoint, params=size).json()["pages"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("directory: could not size %s in %s: %r", pattern, endpoint, exc)
            return found
        for page in range(pages):
            params = {"url": pattern, "output": "json", "fl": "url", "page": str(page)}
            lines = None
            # The index often drops connections under load, which the client does not retry.
            for _ in range(PAGE_ATTEMPTS):
                try:
                    lines = index.request("GET", endpoint, params=params).text.splitlines()
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    error = exc
            if lines is None:
                logger.warning("directory: skipped page %d of %s: %r", page, pattern, error)
                continue
            for line in lines:
                try:
                    url = json.loads(line)["url"]
                except (ValueError, KeyError, TypeError):
                    continue
                found.update(slug for source, slug in ats.boards_in_text(url) if source == system)
        return found

    # Scanning

    def due(self, systems: list[str], limit: int) -> list[tuple[str, str]]:
        """Boards not scanned recently, least recently scanned first, alternating systems."""
        now = self._clock()
        scanned, missing = self._data["scanned"], self._data["missing"]

        def waited(log: dict, key: str, seconds: float) -> bool:
            return key not in log or now - log[key] >= seconds

        queues = []
        for system in systems:
            ready = [
                board
                for board in self._data["boards"].get(system, [])
                if waited(scanned, f"{system}/{board}", RESCAN_SECONDS)
                and waited(missing, f"{system}/{board}", MISSING_RETRY_SECONDS)
            ]
            # Never scanned first, then the longest unscanned.
            ready.sort(key=lambda board: scanned.get(f"{system}/{board}", float("-inf")))
            queues.append([(system, board) for board in ready])
        interleaved = [pair for row in _zip_longest(queues) for pair in row]
        return interleaved[:limit]

    def scan(self, query: "Query") -> Iterator[Posting]:
        systems = [system for system in query.systems if system in FETCHERS]
        self.refresh(systems)
        boards = self.due(systems, query.boards_per_run)
        logger.info("directory: scanning %d company boards", len(boards))
        try:
            for done, (system, board) in enumerate(boards, start=1):
                key = f"{system}/{board}"
                board_query = query.model_copy(update={"source": system, "board": board})
                try:
                    yield from FETCHERS[system](self._client, board_query)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (404, 410):
                        self._data["missing"][key] = self._clock()
                    else:
                        logger.debug("directory: %s failed: %r", key, exc)
                except BOARD_ERRORS as exc:
                    logger.debug("directory: %s failed: %r", key, exc)
                # Reached only once every posting from the board was taken.
                self._data["scanned"][key] = self._clock()
                if done % SAVE_EVERY == 0:
                    self.save()
        finally:
            self.save()


def _zip_longest(queues: list[list[tuple[str, str]]]) -> Iterator[list[tuple[str, str]]]:
    for index in range(max((len(queue) for queue in queues), default=0)):
        yield [queue[index] for queue in queues if index < len(queue)]
