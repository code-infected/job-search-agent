"""Setup routes: model, resume, facts, what to look for, where to look, and preferences."""

import json
import os
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from boards import find_boards
from config import Settings
from envfile import update_env
from graph.facts import FactsStore, FactsStoreError, load_facts
from graph.llm import Budget, ModelCallError
from graph.nodes.discover import SearchConfig, load_search_config
from models import PROVIDERS, get_chat_model
from resume import MAX_RESUME_BYTES, ResumeError, resume_text, suggest_facts

router = APIRouter(prefix="/setup")
MIN_AUTO_RUN_HOURS = 6


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:/-]+$")
    api_key: str = Field(default="", max_length=500)

    @field_validator("provider")
    @classmethod
    def known_provider(cls, value: str) -> str:
        if value not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        return value


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fit_score_threshold: float = Field(ge=0.05, le=0.95)
    max_postings_per_run: int = Field(ge=1, le=200)
    max_cost_per_run_usd: float = Field(ge=0.01, le=100)
    auto_run_hours: float = Field(ge=0, le=168)
    input_usd_per_mtok: float = Field(ge=0, le=1000)
    output_usd_per_mtok: float = Field(ge=0, le=1000)
    adzuna_app_id: str = Field(default="", max_length=200)
    adzuna_app_key: str = Field(default="", max_length=200)

    @field_validator("auto_run_hours")
    @classmethod
    def sensible_schedule(cls, value: float) -> float:
        if 0 < value < MIN_AUTO_RUN_HOURS:
            raise ValueError(
                f"automatic runs must be off or at least {MIN_AUTO_RUN_HOURS} hours apart"
            )
        return value


class BoardSearch(BaseModel):
    query: str = Field(min_length=1, max_length=500)


def model_status() -> dict:
    provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    key_variable = PROVIDERS.get(provider, (None, None))[1]
    return {
        "provider": provider,
        "name": os.environ.get("MODEL_NAME", "").strip(),
        "key_set": provider in PROVIDERS
        and (not key_variable or bool(os.environ.get(key_variable))),
        "providers": sorted(PROVIDERS),
    }


def setup_status(settings: Settings) -> dict:
    """What is configured, and what is still missing before a run can start."""
    model = model_status()
    missing = []
    if not (model["provider"] in PROVIDERS and model["name"] and model["key_set"]):
        missing.append("model")
    facts, facts_error = None, None
    if settings.facts_path.exists():
        try:
            facts = load_facts(settings.facts_path).model_dump()
        except FactsStoreError as exc:
            facts_error = str(exc)
    if facts is None:
        missing.append("facts")
    search, search_error = None, None
    if settings.search_config_path.exists():
        try:
            search = load_search_config(settings.search_config_path).model_dump(exclude_none=True)
        except (OSError, ValidationError) as exc:
            search_error = str(exc)
    if search is None:
        missing.append("search")
    elif not search["queries"]:
        missing.append("sources")
    return {
        "ready": not missing,
        "missing": missing,
        "model": model,
        "facts": facts,
        "facts_error": facts_error,
        "search": search,
        "search_error": search_error,
        "preferences": {
            "fit_score_threshold": settings.fit_score_threshold,
            "max_postings_per_run": settings.max_postings_per_run,
            "max_cost_per_run_usd": settings.max_cost_per_run_usd,
            "auto_run_hours": settings.auto_run_hours,
            "input_usd_per_mtok": settings.input_usd_per_mtok,
            "output_usd_per_mtok": settings.output_usd_per_mtok,
            "adzuna_set": bool(
                os.environ.get("ADZUNA_APP_ID") and os.environ.get("ADZUNA_APP_KEY")
            ),
        },
    }


def _write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(exclude_none=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_env(request: Request, updates: dict[str, str]) -> None:
    try:
        update_env(request.app.state.env_path, updates)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
    except OSError:
        # Usually a container, where settings come from the environment and .env is read-only.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "settings could not be written to .env here, so set them in your environment instead",
        ) from None


@router.get("/state")
def state(request: Request) -> dict:
    return setup_status(request.app.state.settings)


@router.post("/model")
def save_model(choice: ModelSettings, request: Request) -> dict:
    updates = {"MODEL_PROVIDER": choice.provider, "MODEL_NAME": choice.name}
    key_variable = PROVIDERS[choice.provider][1]
    if choice.api_key and key_variable:
        updates[key_variable] = choice.api_key
    _save_env(request, updates)
    return model_status()


@router.post("/resume")
async def read_resume(request: Request) -> dict:
    """Suggest facts from an uploaded resume. Nothing is saved until the user confirms."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_RESUME_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "the file is larger than 5 MB")
    data = await request.body()
    settings = request.app.state.settings
    try:
        text = resume_text(data)
        model = get_chat_model()
    except (ResumeError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    budget = Budget(
        settings.max_tokens_per_run,
        settings.max_cost_per_run_usd,
        settings.input_usd_per_mtok,
        settings.output_usd_per_mtok,
    )
    try:
        return await run_in_threadpool(suggest_facts, model, text, budget)
    except ModelCallError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"the model call failed: {exc}") from None


@router.put("/facts")
def save_facts(facts: FactsStore, request: Request) -> dict:
    _write_json(request.app.state.settings.facts_path, facts)
    return {"saved": len(facts.facts)}


@router.put("/search")
def save_search(search: SearchConfig, request: Request) -> dict:
    _write_json(request.app.state.settings.search_config_path, search)
    return {"saved": len(search.queries)}


@router.post("/boards")
def boards(search: BoardSearch) -> list[dict]:
    return [match.to_dict() for match in find_boards(search.query)]


@router.put("/preferences")
def save_preferences(preferences: Preferences, request: Request) -> dict:
    updates = {
        "FIT_SCORE_THRESHOLD": str(preferences.fit_score_threshold),
        "MAX_POSTINGS_PER_RUN": str(preferences.max_postings_per_run),
        "MAX_COST_PER_RUN_USD": str(preferences.max_cost_per_run_usd),
        "AUTO_RUN_HOURS": str(preferences.auto_run_hours),
        "MODEL_INPUT_USD_PER_MTOK": str(preferences.input_usd_per_mtok),
        "MODEL_OUTPUT_USD_PER_MTOK": str(preferences.output_usd_per_mtok),
    }
    if preferences.adzuna_app_id:
        updates["ADZUNA_APP_ID"] = preferences.adzuna_app_id
    if preferences.adzuna_app_key:
        updates["ADZUNA_APP_KEY"] = preferences.adzuna_app_key
    _save_env(request, updates)
    request.app.state.settings = replace(
        request.app.state.settings,
        fit_score_threshold=preferences.fit_score_threshold,
        max_postings_per_run=preferences.max_postings_per_run,
        max_cost_per_run_usd=preferences.max_cost_per_run_usd,
        auto_run_hours=preferences.auto_run_hours,
        input_usd_per_mtok=preferences.input_usd_per_mtok,
        output_usd_per_mtok=preferences.output_usd_per_mtok,
    )
    return setup_status(request.app.state.settings)["preferences"]


@router.post("/rescore")
def rescore(request: Request) -> dict:
    return {"released": request.app.state.store.release_discarded()}
