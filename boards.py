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

from graph.nodes.discover import BOARD_SOURCES, MAX_MARKUP_CHARS, PoliteClient

logger = logging.getLogger(__name__)

MAX_NAME_CHARS = 100
SLUG = r"([A-Za-z0-9_-]{1,100})"
# Path segments that belong to the board software, not to a company.
RESERVED = frozenset({"embed", "j", "api", "jobs", "v0", "v1", "job_board", "boards"})
URL_PATTERNS = (
    ("greenhouse", re.compile(r"greenhouse\.io/embed/job_board(?:/js)?\?for=" + SLUG)),
    ("greenhouse", re.compile(r"(?:job-boards|boards)(?:\.eu)?\.greenhouse\.io/" + SLUG)),
    ("lever", re.compile(r"jobs\.lever\.co/" + SLUG)),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/" + SLUG)),
    ("workable", re.compile(r"apply\.workable\.com/" + SLUG)),
    ("smartrecruiters", re.compile(r"(?:jobs|careers)\.smartrecruiters\.com/" + SLUG)),
)


@dataclass(frozen=True)
class BoardMatch:
    source: str
    board: str
    open_jobs: int
    sample_titles: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def boards_in_text(text: str) -> list[tuple[str, str]]:
    """Board references in a URL or a page's markup, in order, without duplicates."""
    found: dict[tuple[str, str], None] = {}
    for source, pattern in URL_PATTERNS:
        for slug in pattern.findall(text):
            if slug.lower() not in RESERVED:
                found[(source, slug)] = None
    return list(found)


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


def probe(client: PoliteClient, source: str, board: str) -> Optional[BoardMatch]:
    """Read one board's listing. None if it does not exist or has no open jobs."""
    try:
        if source == "greenhouse":
            jobs = client.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs").json()
            jobs = jobs.get("jobs", [])
            titles = _titles(jobs, "title")
        elif source == "lever":
            jobs = client.get(f"https://api.lever.co/v0/postings/{board}", mode="json").json()
            titles = _titles(jobs, "text")
        elif source == "ashby":
            url = f"https://api.ashbyhq.com/posting-api/job-board/{board}"
            jobs = client.get(url).json().get("jobs", [])
            titles = _titles(jobs, "title")
        elif source == "workable":
            url = f"https://apply.workable.com/api/v1/widget/accounts/{board}"
            jobs = client.get(url).json().get("jobs", [])
            titles = _titles(jobs, "title")
        else:
            url = f"https://api.smartrecruiters.com/v1/companies/{board}/postings"
            data = client.get(url, limit="3").json()
            jobs = [None] * int(data.get("totalFound", 0))
            titles = _titles(data.get("content", []), "name")
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        return None
    if not isinstance(jobs, list) or not jobs:
        return None
    return BoardMatch(source, board, len(jobs), titles)


def find_boards(text: str, client: Optional[PoliteClient] = None) -> list[BoardMatch]:
    """Boards with open jobs for a company name, a board URL, or a careers page URL."""
    # One attempt per probe: most candidate names do not exist, and a retry only adds waiting.
    client = client or PoliteClient(delay_seconds=0, attempts=1)
    text = text.strip()
    if re.match(r"^https?://", text, re.IGNORECASE):
        candidates = boards_in_text(text)
        if not candidates:
            # A company careers page usually embeds or links to its board.
            try:
                candidates = boards_in_text(client.get(text).text[:MAX_MARKUP_CHARS])
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
