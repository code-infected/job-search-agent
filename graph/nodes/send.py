"""Stores an approved message as a draft. No code path here delivers anything."""

import base64
import logging
import os
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, Protocol

import httpx

from graph.state import JobSearchState

logger = logging.getLogger(__name__)

GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
GMAIL_DRAFTS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
DRAFT_SUBJECT = "Application outreach draft"


class DraftSink(Protocol):
    def create_draft(self, run_id: str, posting_url: str, message: str) -> str:
        """Store a draft and return a reference to it."""


class FileDraftSink:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def create_draft(self, run_id: str, posting_url: str, message: str) -> str:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{run_id}.md"
        path.write_text(f"Posting: {posting_url}\n\n{message}\n", encoding="utf-8")
        return str(path)


class GmailDraftSink:
    """Creates a Gmail draft with no recipient. The drafts endpoint is the only one used."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._credentials = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        self._client = client or httpx.Client(timeout=20.0)

    def create_draft(self, run_id: str, posting_url: str, message: str) -> str:
        token_response = self._client.post(GMAIL_TOKEN_URL, data=self._credentials)
        token_response.raise_for_status()
        email = EmailMessage()
        email["Subject"] = DRAFT_SUBJECT
        email.set_content(f"Posting: {posting_url}\n\n{message}\n")
        raw = base64.urlsafe_b64encode(email.as_bytes()).decode()
        response = self._client.post(
            GMAIL_DRAFTS_URL,
            headers={"Authorization": f"Bearer {token_response.json()['access_token']}"},
            json={"message": {"raw": raw}},
        )
        response.raise_for_status()
        return response.json()["id"]


def build_draft_sink(backend: str, drafts_dir: Path) -> DraftSink:
    if backend == "file":
        return FileDraftSink(drafts_dir)
    if backend == "gmail":
        return GmailDraftSink(
            os.environ["GMAIL_CLIENT_ID"],
            os.environ["GMAIL_CLIENT_SECRET"],
            os.environ["GMAIL_REFRESH_TOKEN"],
        )
    raise ValueError(f"DRAFT_BACKEND must be file or gmail, got {backend!r}")


def make_send(sink: DraftSink):
    def send(state: JobSearchState) -> dict:
        if state["approval_status"] != "approved":
            raise RuntimeError(f"send reached without approval for run {state['run_id']}")
        try:
            reference = sink.create_draft(
                state["run_id"], state["posting_url"], state["draft_message"] or ""
            )
        except (OSError, httpx.HTTPError, KeyError) as exc:
            logger.error("draft creation failed for %s: %s", state["posting_url"], exc)
            return {"error": f"send: {exc}"}
        logger.info("draft created for %s: %s", state["posting_url"], reference)
        return {}

    return send
