"""FastAPI review queue. Supplies the external signal that resumes a paused run."""

import logging
import secrets
import threading
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import Settings
from graph.build import Dependencies, build_graph, open_checkpointer
from graph.nodes.human_review import ReviewDecision
from memory.store import MemoryStore
from runner import (
    build_dependencies,
    is_pending,
    pending_reviews,
    run_batch,
    stale_reviews,
    submit_decision,
)

logger = logging.getLogger(__name__)

MIN_TOKEN_LENGTH = 16
DASHBOARD = Path(__file__).with_name("dashboard.html")
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class OutcomeUpdate(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    outcome: str = Field(min_length=1, max_length=2000)


class DeadLetterRelease(BaseModel):
    url: str = Field(min_length=1, max_length=2000)


def create_app(
    settings: Optional[Settings] = None,
    dependency_factory: Optional[Callable[[MemoryStore], Dependencies]] = None,
    store: Optional[MemoryStore] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if len(settings.review_api_token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"REVIEW_API_TOKEN must be set to at least {MIN_TOKEN_LENGTH} characters: "
            "it is what stops anything but the reviewer from approving a draft"
        )
    build = dependency_factory or (lambda memory: build_dependencies(settings, memory))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        app.state.store = store or MemoryStore(settings.vector_store_path)
        app.state.checkpointer = open_checkpointer(settings.checkpoint_db)
        deps = build(app.state.store)
        deps.postings = iter(())  # the review graph only resumes runs, it never discovers
        app.state.graph = build_graph(deps, app.state.checkpointer)
        yield
        app.state.checkpointer.conn.close()

    app = FastAPI(title="Job Search Agent Review Queue", lifespan=lifespan)
    review_lock = threading.Lock()
    batch_lock = threading.Lock()
    bearer = HTTPBearer(auto_error=False)

    def require_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        supplied = credentials.credentials if credentials else ""
        if not secrets.compare_digest(supplied.encode(), settings.review_api_token.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing token")

    authenticated = [Depends(require_token)]

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return DASHBOARD.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/reviews", dependencies=authenticated)
    def list_reviews(request: Request) -> list[dict]:
        state = request.app.state
        return [asdict(review) for review in pending_reviews(state.graph, state.checkpointer)]

    @app.get("/reviews/digest", dependencies=authenticated)
    def digest(request: Request) -> dict:
        state = request.app.state
        waiting = pending_reviews(state.graph, state.checkpointer)
        stale = stale_reviews(waiting, settings.review_stale_hours)
        return {
            "pending": len(waiting),
            "stale": [
                {
                    "run_id": review.run_id,
                    "created_at": review.created_at,
                    "posting_url": review.payload["posting_url"],
                }
                for review in stale
            ],
        }

    @app.post("/reviews/{run_id}", dependencies=authenticated)
    def decide(run_id: str, decision: ReviewDecision, request: Request) -> dict:
        graph = request.app.state.graph
        with review_lock:
            if not is_pending(graph, run_id):
                raise HTTPException(status.HTTP_409_CONFLICT, "no pending review for this run")
            values = submit_decision(graph, run_id, decision)
        return {"run_id": run_id, "approval_status": values["approval_status"]}

    @app.post("/runs", status_code=status.HTTP_202_ACCEPTED, dependencies=authenticated)
    def start_run(request: Request, background: BackgroundTasks) -> dict:
        if not batch_lock.acquire(blocking=False):
            raise HTTPException(status.HTTP_409_CONFLICT, "a batch is already running")
        state = request.app.state

        def run() -> None:
            try:
                deps = build(state.store)
                results = run_batch(deps, state.checkpointer, settings.max_postings_per_run)
                logger.info("batch finished: %d postings", len(results))
            except Exception:
                logger.exception("batch failed")
            finally:
                batch_lock.release()

        background.add_task(run)
        return {"status": "started"}

    @app.post("/outcomes", dependencies=authenticated)
    def record_outcome(update: OutcomeUpdate, request: Request) -> dict:
        if not request.app.state.store.record_outcome(update.url, update.outcome):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no decision recorded for this URL")
        return {"status": "recorded"}

    @app.get("/dead-letters", dependencies=authenticated)
    def dead_letters(request: Request) -> list[dict]:
        return [record.model_dump() for record in request.app.state.store.by_status("failed")]

    @app.post("/dead-letters/release", dependencies=authenticated)
    def release(update: DeadLetterRelease, request: Request) -> dict:
        request.app.state.store.release(update.url)
        return {"status": "released"}

    return app
