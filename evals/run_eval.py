"""Replays past decisions through fit scoring and tailoring and prints the results table.

Every case is scored with no retrieved precedent, so the logged decision cannot leak into
the answer it is being compared against.
"""

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from config import Settings
from evals.dataset import EvalCase, load_cases
from graph.facts import FactsStore, load_facts
from graph.llm import Budget, ModelCallError
from graph.nodes.fit_score import assess_fit
from graph.nodes.tailor import UngroundedDraftError, draft_message
from models import get_chat_model

# Known disqualifier categories, used to check a stated reason names the real one.
DISQUALIFIERS: dict[str, tuple[str, ...]] = {
    "stack mismatch": ("stack", "technolog", "language", "framework", "tool"),
    "experience gap": ("experience", "years", "senior", "seniority", "junior"),
    "compensation": ("comp", "salary", "pay"),
    "location": ("location", "onsite", "on-site", "relocat", "remote", "visa"),
}


def categories(text: str) -> set[str]:
    lowered = text.lower()
    return {
        name
        for name, terms in DISQUALIFIERS.items()
        if any(re.search(rf"\b{re.escape(term)}", lowered) for term in terms)
    }


def reasoning_matches(logged_reason: str, reasoning: str) -> Optional[bool]:
    """Whether the reasoning names a logged disqualifier. None if the log names no category."""
    expected = categories(logged_reason)
    return bool(expected & categories(reasoning)) if expected else None


def langsmith_summary(started_at: datetime) -> Optional[dict]:
    """Cost and latency from LangSmith traces, or None when tracing is not configured."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        return None
    from langsmith import Client

    runs = list(
        Client().list_runs(
            project_name=os.environ.get("LANGSMITH_PROJECT", "job-search-agent"),
            run_type="llm",
            start_time=started_at,
        )
    )
    latencies = [(run.end_time - run.start_time).total_seconds() for run in runs if run.end_time]
    return {
        "llm_calls": len(runs),
        "total_tokens": sum(run.total_tokens or 0 for run in runs),
        "total_cost_usd": float(sum(run.total_cost or 0 for run in runs)),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }


def evaluate(
    model,
    cases: list[EvalCase],
    facts: FactsStore,
    target_profile: str,
    threshold: float,
    budget: Budget,
) -> dict:
    rows = []
    for case in cases:
        try:
            assessment = assess_fit(model, case.posting_text, facts, target_profile, [], budget)
        except ModelCallError as exc:
            rows.append({"url": case.url, "error": str(exc)})
            continue
        predicted = assessment.fit_score >= threshold
        row = {
            "url": case.url,
            "expected_qualified": case.qualified,
            "predicted_qualified": predicted,
            "fit_score": assessment.fit_score,
            "fit_reasoning": assessment.fit_reasoning,
            "reasoning_match": None
            if case.qualified or predicted
            else reasoning_matches(case.reason, assessment.fit_reasoning),
            "draft_verified": None,
        }
        if predicted:
            try:
                row["draft_message"] = draft_message(model, case.posting_text, facts, budget)
                row["draft_verified"] = True
            except UngroundedDraftError as exc:
                row["draft_verified"] = False
                row["draft_problems"] = str(exc)
            except ModelCallError as exc:
                row["draft_error"] = str(exc)
        rows.append(row)

    scored = [row for row in rows if "error" not in row]
    agreed = [row for row in scored if row["expected_qualified"] == row["predicted_qualified"]]
    reasons = [row["reasoning_match"] for row in scored if row["reasoning_match"] is not None]
    drafts = [row["draft_verified"] for row in scored if row["draft_verified"] is not None]
    return {
        "cases": len(cases),
        "scored": len(scored),
        "agreement_rate": len(agreed) / len(scored) if scored else None,
        "reasoning_checked": len(reasons),
        "reasoning_match_rate": sum(reasons) / len(reasons) if reasons else None,
        "drafts_checked": len(drafts),
        "drafts_verified": sum(drafts),
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "rows": rows,
    }


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_table(summary: dict, source: str) -> str:
    lines = [
        f"Dataset: `{source}`, {summary['cases']} cases, model `{summary['model']}`, "
        f"elapsed {summary['elapsed_seconds']:.0f}s",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Fit agreement with logged decision | {_percent(summary['agreement_rate'])} "
        f"({summary['scored']} scored) |",
        f"| Reasoning names the logged disqualifier | {_percent(summary['reasoning_match_rate'])} "
        f"({summary['reasoning_checked']} checked) |",
        f"| Drafts passing the fact check | {summary['drafts_verified']} of "
        f"{summary['drafts_checked']} |",
        f"| Tokens | {summary['input_tokens']} in, {summary['output_tokens']} out |",
    ]
    traces = summary.get("langsmith")
    if traces:
        lines.append(
            f"| LangSmith | {traces['llm_calls']} calls, ${traces['total_cost_usd']:.4f}, "
            f"mean latency {traces['mean_latency_seconds'] or 0:.1f}s |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, default=Path("evals/sample_tracker.csv"))
    parser.add_argument("--facts", type=Path, default=Path("config/facts.example.json"))
    parser.add_argument("--target-profile", default="Any role matching the candidate facts.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="latest", help="results file name, without extension")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/eval_results"),
        help="kept private by default, since real runs contain drafts built from your facts",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    budget = Budget(
        settings.max_tokens_per_run,
        settings.max_cost_per_run_usd,
        settings.input_usd_per_mtok,
        settings.output_usd_per_mtok,
    )
    started_at = datetime.now(UTC)
    began = time.monotonic()
    summary = evaluate(
        get_chat_model(),
        load_cases(args.tracker, args.limit),
        load_facts(args.facts),
        args.target_profile,
        settings.fit_score_threshold,
        budget,
    )
    summary["elapsed_seconds"] = time.monotonic() - began
    summary["model"] = f"{os.environ['MODEL_PROVIDER']}/{os.environ['MODEL_NAME']}"
    summary["langsmith"] = langsmith_summary(started_at)

    table = render_table(summary, args.tracker.name)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / f"{args.output}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.results_dir / f"{args.output}.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
