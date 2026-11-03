"""Find which job board a company uses, from its name or a careers page link.

Users rarely know that a company posts through Greenhouse as "acme-labs", so this
turns "Acme Labs" or a careers URL into board entries that discovery can read.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Optional

import httpx

from graph.sources import BOARD_SOURCES
from graph.sources.ats import RESERVED, URL_PATTERNS, boards_in_text, workday_links, workday_site
from graph.sources.common import MAX_MARKUP_CHARS, PoliteClient

__all__ = ["RESERVED", "URL_PATTERNS", "BoardMatch", "boards_in_text", "find_boards"]

logger = logging.getLogger(__name__)

MAX_NAME_CHARS = 100


@dataclass(frozen=True)
class BoardMatch:
    source: str
    # The board name, or for Workday the careers link.
    board: str
    open_jobs: int
    sample_titles: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def name_candidates(name: str) -> list[str]:
    """Board names a company is likely to use: "Acme Labs" -> acmelabs, acme-labs, acme."""
    words = re.findall(r"[A-Za-z0-9]+", name[:MAX_NAME_CHARS])
    if not words:
        return []
    lowered = [word.lower() for word in words]
    candidates = ["".join(lowered), "-".join(lowered), "".join(words), lowered[0]]
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _titles(jobs: list, key: str) -> tuple[str, ...]:
    return tuple(str(job.get(key, "")) for job in jobs[:3] if isinstance(job, dict))


def _listing(client: PoliteClient, source: str, board: str) -> tuple[int, tuple[str, ...]]:
    if source == "greenhouse":
        jobs = client.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs").json()
        jobs = jobs.get("jobs", [])
        return len(jobs), _titles(jobs, "title")
    if source == "lever":
        jobs = client.get(f"https://api.lever.co/v0/postings/{board}", mode="json").json()
        return len(jobs), _titles(jobs, "text")
    if source == "ashby":
        url = f"https://api.ashbyhq.com/posting-api/job-board/{board}"
        jobs = client.get(url).json().get("jobs", [])
        return len(jobs), _titles(jobs, "title")
    if source == "workable":
        url = f"https://apply.workable.com/api/v1/widget/accounts/{board}"
        jobs = client.get(url).json().get("jobs", [])
        return len(jobs), _titles(jobs, "title")
    if source == "smartrecruiters":
        url = f"https://api.smartrecruiters.com/v1/companies/{board}/postings"
        data = client.get(url, limit="3").json()
        return int(data.get("totalFound", 0)), _titles(data.get("content", []), "name")
    if source == "recruitee":
        jobs = client.get(f"https://{board.lower()}.recruitee.com/api/offers/").json()
        jobs = jobs.get("offers", [])
        return len(jobs), _titles(jobs, "title")
    if source == "breezy":
        jobs = client.get(f"https://{board.lower()}.breezy.hr/json").json()
        return len(jobs), _titles(jobs, "name")
    if source == "rippling":
        data = client.get(f"https://ats.rippling.com/api/v2/board/{board}/jobs").json()
        return int(data.get("totalItems", 0)), _titles(data.get("items", []), "name")
    if source == "gem":
        jobs = client.get(f"https://api.gem.com/job_board/v0/{board}/job_posts/").json()
        return len(jobs), _titles(jobs, "title")
    if source == "workday":
        host, tenant, site = workday_site(board)
        data = client.request(
            "POST",
            f"https://{host}/wday/cxs/{tenant}/{site}/jobs",
            json={"appliedFacets": {}, "limit": 3, "offset": 0, "searchText": ""},
        ).json()
        return int(data.get("total", 0)), _titles(data.get("jobPostings", []), "title")
    raise ValueError(f"unknown board system {source}")


def probe(client: PoliteClient, source: str, board: str) -> Optional[BoardMatch]:
    """Read one board's listing. None if it does not exist or has no open jobs."""
    try:
        count, titles = _listing(client, source, board)
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, AttributeError, TypeError):
        return None
    if count <= 0:
        return None
    return BoardMatch(source, board, count, titles)


def _candidates_in_text(text: str) -> list[tuple[str, str]]:
    return boards_in_text(text) + [("workday", link) for link in workday_links(text)]


def find_boards(text: str, client: Optional[PoliteClient] = None) -> list[BoardMatch]:
    """Boards with open jobs for a company name, a board URL, or a careers page URL."""
    # One attempt per probe: most candidate names do not exist, and a retry only adds waiting.
    client = client or PoliteClient(delay_seconds=0, attempts=1)
    text = text.strip()
    if re.match(r"^https?://", text, re.IGNORECASE):
        candidates = _candidates_in_text(text)
        if not candidates:
            # A company careers page usually embeds or links to its board.
            try:
                candidates = _candidates_in_text(client.get(text).text[:MAX_MARKUP_CHARS])
            except (httpx.HTTPError, httpx.InvalidURL, ValueError, OSError) as exc:
                logger.warning("could not read careers page %s: %r", text, exc)
                return []
    else:
        candidates = [(source, slug) for slug in name_candidates(text) for source in BOARD_SOURCES]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(lambda pair: probe(client, *pair), candidates)
    # Board names are case-insensitive on most systems, so one board can answer twice.
    unique: dict[tuple[str, str], BoardMatch] = {}
    for match in results:
        if match:
            unique.setdefault((match.source, match.board.lower()), match)
    return sorted(unique.values(), key=lambda match: -match.open_jobs)
