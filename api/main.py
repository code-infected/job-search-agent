"""Local status page: start a run, watch its progress, and browse the jobs found.

Drafts are reviewed in the terminal (`python cli.py pending` and `decide`), so nothing
here can approve or save one. The page has no login because it is a personal tool, so it
only answers requests addressed to this machine, and only starts a run when asked by
the page itself. That stops another website open in the browser from reading the job
list through DNS rebinding or starting paid runs with a cross-site request.
"""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response

from config import Settings
from graph.build import Dependencies, build_graph, open_checkpointer
from jobs import job_rows, to_csv
from memory.store import MemoryStore
from runner import build_dependencies, pending_reviews, run_batch

logger = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
DASHBOARD = Path(__file__).with_name("dashboard.html")
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def _hostname(value: str) -> str:
    return (urlsplit(f"//{value}").hostname or "").lower()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_app(
    settings: Optional[Settings] = None,
    dependency_factory: Optional[Callable[[MemoryStore], Dependencies]] = None,
    store: Optional[MemoryStore] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    build = dependency_factory or (lambda memory: build_dependencies(settings, memory))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        app.state.store = store or MemoryStore(settings.vector_store_path)
        app.state.checkpointer = open_checkpointer(settings.checkpoint_db)
        deps = build(app.state.store)
        deps.postings = iter(())  # only reads paused runs, never discovers
        app.state.graph = build_graph(deps, app.state.checkpointer)
        yield
        app.state.checkpointer.conn.close()

    app = FastAPI(
        title="Job Search Agent",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    batch_lock = threading.Lock()
    progress = {"running": False, "checked": 0, "drafted": 0, "message": "No run yet."}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if _hostname(request.headers.get("host", "")) not in LOCAL_HOSTS:
            return JSONResponse({"detail": "only local requests are served"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method != "GET" and origin and urlsplit(origin).hostname not in LOCAL_HOSTS:
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def all_jobs(state) -> list[dict]:
        pending = pending_reviews(state.graph, state.checkpointer)
        return job_rows(state.store, [review.payload for review in pending])

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return DASHBOARD.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/status")
    def run_status() -> dict:
        return dict(progress)

    @app.get("/jobs")
    def jobs(request: Request) -> list[dict]:
        return all_jobs(request.app.state)

    @app.get("/jobs.csv")
    def jobs_csv(request: Request) -> Response:
        return Response(
            "﻿" + to_csv(all_jobs(request.app.state)),  # BOM so Excel reads UTF-8
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="jobs.csv"'},
        )

    @app.post("/runs", status_code=status.HTTP_202_ACCEPTED)
    def start_run(request: Request, background: BackgroundTasks) -> dict:
        if not batch_lock.acquire(blocking=False):
            raise HTTPException(status.HTTP_409_CONFLICT, "a run is already in progress")
        state = request.app.state
        progress.update(running=True, checked=0, drafted=0, started_at=_now(), finished_at=None)
        progress["message"] = "Searching job boards..."

        def counted(values: dict) -> None:
            progress["checked"] += 1
            progress["drafted"] += values["approval_status"] == "pending"
            progress["message"] = f"Checked {progress['checked']} postings..."

        def run() -> None:
            try:
                deps = build(state.store)
                run_batch(deps, state.checkpointer, settings.max_postings_per_run, counted)
                stopped = " Stopped at the spend ceiling." if deps.budget.exceeded else ""
                progress["message"] = (
                    f"Finished: {progress['checked']} new postings checked, "
                    f"{progress['drafted']} drafts waiting for review.{stopped}"
                )
            except Exception as error:
                logger.exception("run failed")
                progress["message"] = f"Run failed: {error}"
            finally:
                progress.update(running=False, finished_at=_now())
                batch_lock.release()

        background.add_task(run)
        return {"status": "started"}

    return app
