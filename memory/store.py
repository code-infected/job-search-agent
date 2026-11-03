"""Vector store over past postings and decisions, retrieved before scoring a new posting.

Embeddings use Chroma's bundled local model, so stored vectors stay valid when
MODEL_PROVIDER changes and no vendor receives the posting history.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import chromadb
from chromadb.config import Settings as ChromaSettings

from memory.schema import PRECEDENT_STATUSES, DecisionRecord, Status

COLLECTION = "job_search_decisions"
EXCERPT_CHARS = 2000
TRACKING_PARAMS = frozenset({"gh_src", "ref", "source"})


def canonical_url(url: str) -> str:
    """Drop fragments and tracking parameters so one posting has one identity."""
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query)
        if not (key.lower().startswith("utm_") or key.lower() in TRACKING_PARAMS)
    ]
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(query), "")
    )


def _record_id(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


class MemoryStore:
    def __init__(self, path: Path, embedding_function: Any = None) -> None:
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        options = {"embedding_function": embedding_function} if embedding_function else {}
        self._collection = client.get_or_create_collection(COLLECTION, **options)

    def has_url(self, url: str) -> bool:
        return bool(self._collection.get(ids=[_record_id(url)], include=[])["ids"])

    def claim(self, url: str, posting_text: str, run_id: str) -> None:
        """Mark a posting as in progress so a later run cannot pick it up while it is pending."""
        self.write(
            DecisionRecord(
                url=canonical_url(url),
                posting_excerpt=posting_text[:EXCERPT_CHARS],
                status="in_progress",
                run_id=run_id,
                decided_at=datetime.now(UTC).isoformat(),
            )
        )

    def write(self, record: DecisionRecord) -> None:
        metadata = record.model_dump(exclude={"posting_excerpt"}, exclude_none=True)
        self._collection.upsert(
            ids=[_record_id(record.url)],
            documents=[record.posting_excerpt or record.url],
            metadatas=[metadata],
        )

    def similar_decisions(self, posting_text: str, limit: int = 5) -> list[DecisionRecord]:
        where = {"status": {"$in": list(PRECEDENT_STATUSES)}}
        available = len(self._collection.get(where=where, include=[])["ids"])
        if not available:
            return []
        result = self._collection.query(
            query_texts=[posting_text[:EXCERPT_CHARS]],
            n_results=min(limit, available),
            where=where,
            include=["documents", "metadatas"],
        )
        return [
            DecisionRecord(posting_excerpt=document, **metadata)
            for document, metadata in zip(
                result["documents"][0], result["metadatas"][0], strict=True
            )
        ]

    def by_status(self, status: Status) -> list[DecisionRecord]:
        result = self._collection.get(where={"status": status}, include=["documents", "metadatas"])
        return [
            DecisionRecord(posting_excerpt=document, **metadata)
            for document, metadata in zip(result["documents"], result["metadatas"], strict=True)
        ]

    def all_records(self) -> list[DecisionRecord]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            DecisionRecord(posting_excerpt=document, **metadata)
            for document, metadata in zip(result["documents"], result["metadatas"], strict=True)
        ]

    def record_outcome(self, url: str, outcome: str) -> bool:
        """Attach the real-world outcome to a past decision. False if the URL is unknown."""
        result = self._collection.get(ids=[_record_id(url)], include=["documents", "metadatas"])
        if not result["ids"]:
            return False
        record = DecisionRecord(
            posting_excerpt=result["documents"][0], **result["metadatas"][0], strict=True
        )
        self.write(record.model_copy(update={"outcome": outcome}))
        return True

    def release_failed(self, url: str) -> bool:
        """Let discovery retry a dead letter. Decisions and pending claims cannot be released."""
        record_id = _record_id(url)
        result = self._collection.get(ids=[record_id], include=["metadatas"])
        if not result["ids"] or result["metadatas"][0]["status"] != "failed":
            return False
        self._collection.delete(ids=[record_id])
        return True
