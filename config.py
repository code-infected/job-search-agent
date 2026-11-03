"""Runtime settings, read once from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class Settings:
    vector_store_path: Path
    checkpoint_db: Path
    facts_path: Path
    search_config_path: Path
    drafts_dir: Path
    draft_backend: str
    max_postings_per_run: int
    max_cost_per_run_usd: float
    max_tokens_per_run: int
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    fit_score_threshold: float
    request_delay_seconds: float
    review_stale_hours: float
    auto_run_hours: float

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            vector_store_path=Path(env("VECTOR_STORE_PATH", "./data/chroma")),
            checkpoint_db=Path(env("CHECKPOINT_DB", "./data/checkpoints.sqlite")),
            facts_path=Path(env("FACTS_PATH", "./data/facts.json")),
            search_config_path=Path(env("SEARCH_CONFIG_PATH", "./data/search.json")),
            drafts_dir=Path(env("DRAFTS_DIR", "./data/drafts")),
            draft_backend=env("DRAFT_BACKEND", "file"),
            max_postings_per_run=int(env("MAX_POSTINGS_PER_RUN", "25")),
            max_cost_per_run_usd=float(env("MAX_COST_PER_RUN_USD", "2.00")),
            max_tokens_per_run=int(env("MAX_TOKENS_PER_RUN", "500000")),
            input_usd_per_mtok=float(env("MODEL_INPUT_USD_PER_MTOK", "0")),
            output_usd_per_mtok=float(env("MODEL_OUTPUT_USD_PER_MTOK", "0")),
            fit_score_threshold=float(env("FIT_SCORE_THRESHOLD", "0.7")),
            request_delay_seconds=float(env("REQUEST_DELAY_SECONDS", "1.0")),
            review_stale_hours=float(env("REVIEW_STALE_HOURS", "24")),
            auto_run_hours=float(env("AUTO_RUN_HOURS", "0")),
        )
