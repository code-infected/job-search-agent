"""Command line: run a batch, list jobs, review drafts, and record what happened next."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from boards import find_boards
from config import Settings
from graph.build import build_graph, open_checkpointer
from graph.nodes.discover import DIRECTORY_SYSTEMS
from graph.nodes.human_review import ReviewDecision
from graph.sources.directory import BoardDirectory
from jobs import job_rows, to_csv
from memory.store import MemoryStore
from runner import (
    build_dependencies,
    build_read_only_graph,
    directory_path,
    is_pending,
    pending_reviews,
    run_batch,
    stale_reviews,
    submit_decision,
)


def printable(value: object) -> str:
    """Stored text derives from the open web, so it must not reach the terminal as escapes."""
    return "".join(ch if ch.isprintable() else " " for ch in str(value))


def print_jobs(store: MemoryStore, settings: Settings, csv_path: Optional[Path]) -> None:
    checkpointer = open_checkpointer(settings.checkpoint_db)
    waiting = pending_reviews(build_read_only_graph(store, checkpointer), checkpointer)
    rows = job_rows(store, [review.payload for review in waiting])
    if csv_path:
        csv_path.write_text(to_csv(rows), encoding="utf-8-sig")
        print(f"{len(rows)} postings written to {csv_path}")
        return
    for row in rows:
        score = "  -  " if row["fit_score"] is None else f"{row['fit_score']:.2f}"
        where = " | ".join(printable(row[key]) for key in ("company", "location") if row[key])
        print(f"{score}  {printable(row['title'])} | {where}")
        print(f"      {row['status']}  {printable(row['url'])}")
    print(f"{len(rows)} postings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="discover and process postings")
    commands.add_parser("pending", help="list reviews waiting for a decision")
    commands.add_parser("digest", help="list reviews waiting past the stale limit")
    decide = commands.add_parser("decide", help="approve or reject a pending review")
    decide.add_argument("run_id")
    decide.add_argument("decision", choices=["approved", "rejected"])
    decide.add_argument("--reason")
    jobs = commands.add_parser("jobs", help="list every posting found, best fit first")
    jobs.add_argument("--csv", type=Path, help="write the list to this CSV file instead")
    outcome = commands.add_parser("outcome", help="record what happened after applying")
    outcome.add_argument("url")
    outcome.add_argument("outcome", help='for example "interview" or "no reply"')
    release = commands.add_parser("release", help="let a failed posting be tried again")
    release.add_argument("url")
    commands.add_parser("rescore", help="score ruled-out postings again on the next run")
    find = commands.add_parser("find-board", help="find the job board a company uses")
    find.add_argument("company", help="a company name or careers page link")
    directory = commands.add_parser(
        "directory", help="build or show the list of companies on each board system"
    )
    directory.add_argument(
        "--refresh", action="store_true", help="read Common Crawl again even if the list is new"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Request logs would print full URLs, and some sources carry an API key in the URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()

    if args.command == "find-board":
        matches = find_boards(args.company)
        for match in matches:
            print(f"{match.source} {match.board}: {match.open_jobs} open jobs")
            for title in match.sample_titles:
                print(f"    {printable(title)}")
        print(f"{len(matches)} boards found. Add one on the setup page or to data/search.json.")
        return

    if args.command == "directory":
        print("Reading Common Crawl's index. This takes a few minutes.")
        board_directory = BoardDirectory(directory_path(settings))
        counts = board_directory.refresh(list(DIRECTORY_SYSTEMS), force=args.refresh)
        for system, count in counts.items():
            print(f"{system:16} {count} companies")
        print(f"{sum(counts.values())} companies in total")
        return

    store = MemoryStore(settings.vector_store_path)
    if args.command in ("jobs", "outcome", "release", "rescore"):
        if args.command == "rescore":
            count = store.release_discarded()
            print(f"{count} ruled-out postings will be scored again on the next run")
        elif args.command == "outcome":
            if not store.record_outcome(args.url, args.outcome):
                parser.error(f"no decision recorded for {args.url}")
            print("recorded")
        elif args.command == "release":
            if not store.release_failed(args.url):
                parser.error(f"no failed posting recorded for {args.url}")
            print("released, the next run will try it again")
        else:
            print_jobs(store, settings, args.csv)
        return

    deps = build_dependencies(settings, store)
    checkpointer = open_checkpointer(settings.checkpoint_db)

    if args.command == "run":
        for values in run_batch(deps, checkpointer, settings.max_postings_per_run):
            print(values["approval_status"], values["fit_score"], printable(values["posting_url"]))
        return

    deps.postings = iter(())
    graph = build_graph(deps, checkpointer)
    if args.command == "decide":
        if not is_pending(graph, args.run_id):
            parser.error(f"no pending review for {args.run_id}")
        decision = ReviewDecision(decision=args.decision, reason=args.reason)
        print(submit_decision(graph, args.run_id, decision)["approval_status"])
        return

    reviews = pending_reviews(graph, checkpointer)
    if args.command == "digest":
        reviews = stale_reviews(reviews, settings.review_stale_hours)
    for review in reviews:
        print(review.run_id, review.created_at, review.payload["fit_score"])
        print(f"  {printable(review.payload['posting_url'])}")
        print(f"  {printable(review.payload['fit_reasoning'])}")
        print(f"  Draft: {printable(review.payload['draft_message'])}")
    print(f"{len(reviews)} waiting")


if __name__ == "__main__":
    main()
