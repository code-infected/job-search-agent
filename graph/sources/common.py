"""What every job source shares: a careful HTTP client, posting text, and title filters."""

import ipaddress
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

MAX_POSTING_CHARS = 20_000
MAX_MARKUP_CHARS = 1_000_000
MAX_RESPONSE_BYTES = 50_000_000
MAX_URL_CHARS = 2000
REQUEST_DEADLINE_SECONDS = 90.0
MAX_ATTEMPTS = 4
MAX_RETRY_WAIT_SECONDS = 60.0
USER_AGENT = "job-search-agent/1.0"
# A posting open to anyone, anywhere, passes any location filter.
ANYWHERE = ("worldwide", "anywhere")


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


def posting_text(title: object, location: object, body: str, company: object = "") -> str:
    title, location, company = (str(part or "").strip() for part in (title, location, company))
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


def join(*parts: object) -> str:
    return ", ".join(str(part) for part in parts if part)


def title_matches(title: str, query: Any) -> bool:
    lowered = title.lower()
    if any(word.lower() in lowered for word in query.exclude_keywords):
        return False
    return not query.keywords or any(word.lower() in lowered for word in query.keywords)


def place_matches(location: str, query: Any) -> bool:
    place = location.lower()
    if not query.locations or any(word in place for word in ANYWHERE):
        return True
    return any(word.lower() in place for word in query.locations)


def matches(title: str, location: str, query: Any) -> bool:
    """Filter before any model call, so a posting the user cannot take costs nothing."""
    return title_matches(title, query) and place_matches(location, query)


class PoliteClient:
    """HTTP with a minimum delay between requests to each host, and backoff on 429 and 5xx.

    The delay is per host, so a scan that alternates between job boards stays polite to
    each one without waiting on the others.
    """

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = MAX_ATTEMPTS,
        deadline_seconds: float = REQUEST_DEADLINE_SECONDS,
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
        self._deadline = deadline_seconds
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, url: str, **params: str) -> httpx.Response:
        return self.request("GET", url, params=params)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        json: Optional[dict] = None,
        auth: Optional[tuple[str, str]] = None,
    ) -> httpx.Response:
        host = urlsplit(url).hostname or ""
        for attempt in range(self._attempts):
            self._wait_turn(host)
            response = self._fetch(method, url, params or {}, headers, json, auth)
            if response.status_code != 429 and response.status_code < 500:
                break
            if attempt + 1 < self._attempts:
                self._sleep(min(self._retry_wait(response, attempt), MAX_RETRY_WAIT_SECONDS))
        response.raise_for_status()
        return response

    def _wait_turn(self, host: str) -> None:
        """Reserve the next slot for this host, so threads sharing the client also wait."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._last_request.get(host, -self._delay) + self._delay)
            self._last_request[host] = slot
        if slot > now:
            self._sleep(slot - now)

    def _fetch(self, method, url, params, headers, json, auth) -> httpx.Response:
        """Read a capped body under an overall deadline, which per-read timeouts do not give."""
        deadline = time.monotonic() + self._deadline
        body = bytearray()
        extra = {"headers": headers, "json": json}
        if auth:
            extra["auth"] = auth
        with self._client.stream(method, url, params=params, **extra) as response:
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError(f"response too large: {url}")
                if time.monotonic() > deadline:
                    raise ValueError(f"response too slow: {url}")
            kept = ("content-type", "retry-after")
            kept_headers = {key: value for key, value in response.headers.items() if key in kept}
            return httpx.Response(
                response.status_code,
                headers=kept_headers,
                content=bytes(body),
                request=response.request,
            )

    @staticmethod
    def _retry_wait(response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("Retry-After", "")
        return float(header) if header.isdigit() else float(2 ** (attempt + 1))
