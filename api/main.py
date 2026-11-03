"""Local web app: set up the agent, start or schedule runs, and browse the jobs found.

Drafts are reviewed in the terminal (`python cli.py pending` and `decide`), so nothing
here can approve or save one. The app has no login because it is a personal tool, so it
only answers requests addressed to this machine, and only accepts changes made from its
own pages. That stops another website open in the browser from reading your data
through DNS rebinding, or starting paid runs and changing settings with a cross-site
request.
"""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response

from api.setup import router as setup_router
from api.setup import setup_status
from config import ENV_PATH, Settings
from graph.build import Dependencies, open_checkpointer
from graph.nodes.discover import DIRECTORY_SYSTEMS, FetchLog
from graph.sources.directory import BoardDirectory
from jobs import job_rows, to_csv
from memory.store import MemoryStore
from runner import (
    build_dependencies,
    build_read_only_graph,
    directory_path,
    pending_reviews,
    run_batch,
)

logger = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
PAGES = Path(__file__).parent
SCHEDULER_TICK_SECONDS = 60
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
    env_path: Path = ENV_PATH,
    scheduler: bool = True,
) -> FastAPI:
    progress = {"running": False, "checked": 0, "drafted": 0, "message": "No run yet."}
    batch_lock = threading.Lock()
    stop = threading.Event()

    def build(app: FastAPI) -> Dependencies:
        if dependency_factory:
            return dependency_factory(app.state.store)
        return build_dependencies(app.state.settings, app.state.store)

    def start_batch(app: FastAPI) -> bool:
        if not batch_lock.acquire(blocking=False):
            return False
        progress.update(
            running=True,
            checked=0,
            drafted=0,
            tokens=0,
            cost_usd=None,
            started_at=_now(),
            finished_at=None,
            message="Searching job boards...",
        )

        def run() -> None:
            try:
                deps = build(app)
                settings = app.state.settings

                def counted(values: dict) -> None:
                    progress["checked"] += 1
                    progress["drafted"] += values["approval_status"] == "pending"
                    progress["tokens"] = deps.budget.input_tokens + deps.budget.output_tokens
                    if settings.input_usd_per_mtok or settings.output_usd_per_mtok:
                        progress["cost_usd"] = round(deps.budget.cost_usd, 4)
                    progress["message"] = f"Checked {progress['checked']} postings..."

                run_batch(deps, app.state.checkpointer, settings.max_postings_per_run, counted)
                stopped = " Stopped at the spend limit." if deps.budget.exceeded else ""
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

        app.state.batch_thread = threading.Thread(target=run, daemon=True)
        app.state.batch_thread.start()
        return True

    def start_directory_build(app: FastAPI) -> bool:
        """Read the crawl index for every board system. Shares the run lock, since both write
        the directory file."""
        if not batch_lock.acquire(blocking=False):
            return False
        progress.update(
            running=True,
            started_at=_now(),
            finished_at=None,
            message="Building the list of companies from Common Crawl. This takes a few minutes.",
        )

        def build_list() -> None:
            try:
                directory = BoardDirectory(directory_path(app.state.settings))
                counts = directory.refresh(list(DIRECTORY_SYSTEMS))
                progress["message"] = f"Company list ready: {sum(counts.values())} companies."
            except Exception as error:
                logger.exception("directory build failed")
                progress["message"] = f"Building the company list failed: {error}"
            finally:
                progress.update(running=False, finished_at=_now())
                batch_lock.release()

        app.state.batch_thread = threading.Thread(target=build_list, daemon=True)
        app.state.batch_thread.start()
        return True

    def scheduled_tick(app: FastAPI) -> bool:
        """Start a run if automatic runs are on, setup is complete, and one is due."""
        hours = app.state.settings.auto_run_hours
        if hours <= 0 or not setup_status(app.state.settings)["ready"]:
            return False
        if not app.state.schedule_log.allow("scheduled_run", hours * 3600):
            return False
        return start_batch(app)

    def scheduler_loop(app: FastAPI) -> None:
        while not stop.wait(SCHEDULER_TICK_SECONDS):
            try:
                scheduled_tick(app)
            except Exception:
                logger.exception("scheduled run check failed")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        app.state.store = store or MemoryStore(app.state.settings.vector_store_path)
        app.state.checkpointer = open_checkpointer(app.state.settings.checkpoint_db)
        app.state.graph = build_read_only_graph(app.state.store, app.state.checkpointer)
        app.state.schedule_log = FetchLog(app.state.settings.checkpoint_db.parent / "schedule.json")
        if scheduler:
            threading.Thread(target=scheduler_loop, args=(app,), daemon=True).start()
        yield
        stop.set()
        app.state.checkpointer.conn.close()

    app = FastAPI(
        title="Job Search Agent",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings or Settings.from_env()
    app.state.env_path = env_path
    app.state.start_batch = start_batch
    app.state.start_directory_build = start_directory_build
    app.state.scheduled_tick = scheduled_tick
    app.include_router(setup_router)

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
        return (PAGES / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/setup", response_class=HTMLResponse, include_in_schema=False)
    def setup_page() -> str:
        return (PAGES / "setup.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/status")
    def run_status(request: Request) -> dict:
        settings = request.app.state.settings
        setup = setup_status(settings)
        return {
            **progress,
            "ready": setup["ready"],
            "missing": setup["missing"],
            "auto_run_hours": settings.auto_run_hours,
        }

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
    def start_run(request: Request) -> dict:
        setup = setup_status(request.app.state.settings)
        if not setup["ready"]:
            missing = ", ".join(setup["missing"])
            raise HTTPException(status.HTTP_409_CONFLICT, f"finish setup first: {missing}")
        if not start_batch(request.app):
            raise HTTPException(status.HTTP_409_CONFLICT, "a run is already in progress")
        return {"status": "started"}

    return app
