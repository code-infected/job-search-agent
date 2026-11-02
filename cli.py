"""Command line: run a batch, list reviews, decide one, or print the daily digest."""

import argparse
import logging

from config import Settings
from graph.build import build_graph, open_checkpointer
from graph.nodes.human_review import ReviewDecision
from runner import (
    build_dependencies,
    is_pending,
    pending_reviews,
    run_batch,
    stale_reviews,
    submit_decision,
)


def printable(value: object) -> str:
    """Stored text derives from the open web, so it must not reach the terminal as escapes."""
    return "".join(ch if ch.isprintable() else " " for ch in str(value))


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    deps = build_dependencies(settings)
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
