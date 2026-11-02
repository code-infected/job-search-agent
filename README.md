# job-search-agent

A LangGraph agent that runs a job search end to end. It discovers postings, scores fit with visible reasoning, drafts outreach grounded only in verified facts, stops at a human approval gate, and writes every outcome back into memory so later scoring has real precedent.

It is built for judgment, not volume. It never submits anything on its own.

## How it works

```
discover -> fit_score -> score passes threshold?
                           |no                |yes
                     mark_discarded         tailor -> fact check passes?
                           |                  |no            |yes
                           |             dead_letter    human_review (paused)
                           |                  |              |
                           |                  |        approved? -- no ------+
                           |                  |              |yes            |
                           |                  |            send (draft only) |
                           v                  v              v               v
                                         memory_write
                                              |
                     retrieved by the next run's fit_score
```

Each run handles one posting, in its own checkpointed thread. A batch repeats runs until discovery is exhausted or a cap or spend ceiling is reached.

| Guarantee | How it is enforced |
| --- | --- |
| Nothing sends itself | `send` only creates a local file or a Gmail draft with no recipient, and refuses to run without approval |
| No invented substance | `tailor` sees only the posting and the facts store, and `graph/claims.py` checks every sentence deterministically |
| Human review is a real gate | `human_review` is a LangGraph `interrupt` on a SQLite checkpointer. In-memory checkpointing is refused |
| Every score has reasoning | `FitAssessment` rejects a missing or blank reason, and the call is retried |
| Provider-agnostic | All model access goes through `models.py`, and a test fails if `graph/`, `memory/`, `evals/`, or `api/` imports a provider SDK |
| Untrusted input stays data | Posting text is wrapped in a random per-call tag and never enters a system prompt |

### The fact check

`graph/claims.py` uses no model, so a posting cannot influence it. It fails closed. Only a closed set of greetings, sign-offs, and closings is exempt, plus one opening template whose role and employer must appear in the posting. Every other sentence is a claim, including any statement about the employer. Claims must follow these rules:

- Words must come from the facts store. Terms found only in the posting do not count, because borrowing a posting's requirements is the usual way a draft invents experience.
- Phrases such as "at Acme" or "Bachelor of Law" must appear in a single fact.
- A sentence with a figure must stay inside one fact holding that figure, and keep any hedge such as "estimated" or "targeting".
- Links, email addresses, and hidden or symbol characters are rejected unless the facts contain them.

A draft that fails gets one rewrite with the violations as feedback. If it still fails it goes to the dead-letter list and never reaches a reviewer. The cost of this strictness is that drafts stay close to the wording of your facts and say nothing about the employer beyond naming the role.

This is a vocabulary check, not a semantic proof. "I built PostgreSQL and React." uses only fact vocabulary and passes, even though it misstates both. That is one reason the human gate exists.

## Setup

Requires Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock
cp .env.example .env             # set MODEL_PROVIDER, MODEL_NAME, the matching key, REVIEW_API_TOKEN
mkdir -p data
cp config/facts.example.json data/facts.json
cp config/search.example.json data/search.json
```

Edit both files in `data/`, which is gitignored:

- `facts.json` is the only thing the agent may say about you. Write each fact the way you would be happy to see it quoted.
- `search.json` holds `target_profile`, a plain-language description of the roles you want, and `queries`. Each query is a public Greenhouse or Lever board or a direct `url`, with optional `keywords`, `exclude_keywords`, and `locations` filters applied before any model call.

## Running

Review in the browser:

```bash
uvicorn api.main:create_app --factory --port 8000
```

Open `http://127.0.0.1:8000`, enter your `REVIEW_API_TOKEN`, start a run, and approve or reject each draft. A rejection needs a reason, because it becomes precedent.

Or from the command line:

```bash
python cli.py run
python cli.py pending
python cli.py decide <run_id> approved
python cli.py decide <run_id> rejected --reason "Too much on-call"
python cli.py digest              # reviews waiting past REVIEW_STALE_HOURS
```

Drafts land in `data/drafts/`. With `DRAFT_BACKEND=gmail` and OAuth credentials using the `gmail.compose` scope, they become Gmail drafts instead.

To run in a container, mount your data and pass your settings. The image contains neither:

```bash
docker build -t job-search-agent .
docker run -p 127.0.0.1:8000:8000 --env-file .env -v "$PWD/data:/app/data" job-search-agent
```

| Route | Purpose |
| --- | --- |
| `GET /reviews`, `POST /reviews/{run_id}` | List pending reviews, and approve or reject one |
| `GET /reviews/digest` | Pending count and reviews past the stale limit |
| `POST /runs` | Start a batch |
| `POST /outcomes` | Attach a real-world outcome to a past decision |
| `GET /dead-letters`, `POST /dead-letters/release` | Inspect failures, and allow one to be retried |

All except `GET /health` and the dashboard page require `Authorization: Bearer <REVIEW_API_TOKEN>`.

## Switching providers

Change `MODEL_PROVIDER` and `MODEL_NAME` in `.env`, set the matching key, and restart. Embeddings use a local model bundled with Chroma, so stored memory stays valid across vendors.

## Evaluation

```bash
python -m evals.run_eval --output sample_run --results-dir evals/results
python -m evals.run_eval --tracker path/to/tracker.csv --facts data/facts.json --target-profile "..."
```

A tracker is a CSV of past human decisions with the columns `url`, `posting_text`, `decision`, and `reason`. It is only an answer key for measuring accuracy. The agent does not need one to run. Results go to `data/eval_results/` by default, which stays private, since a real run contains drafts built from your facts.

Latest run, on the synthetic sample in `evals/sample_tracker.csv`:

| Measure | Result |
| --- | --- |
| Fit agreement with logged decision | 100% (10 scored) |
| Reasoning names the logged disqualifier | 100% (6 checked) |
| Drafts passing the fact check | 4 of 4 |
| Tokens | 23079 in, 3637 out |
| LangSmith | 15 calls, $0.0767, mean latency 3.1s |

Model: `anthropic/claude-sonnet-5`. Raw output is in `evals/results/`.

Read this as a smoke test, not an accuracy figure. The sample has 10 hand-written, clear-cut cases. A meaningful number needs a real decision history replayed with `--tracker`.

Run the unit tests with `pytest`. They use a scripted model and need no keys or network. Run the eval before shipping any prompt change.

## Failure handling

| Failure | Behaviour |
| --- | --- |
| Model call fails or returns malformed output | Three attempts with backoff, then dead-letter. The batch continues |
| Posting seen before | Skipped by canonical URL before any model call |
| Restart during a pending review | The review is restored from SQLite and resumes without repeating earlier nodes |
| Review left unanswered | Listed by the digest after `REVIEW_STALE_HOURS` |
| Spend ceiling reached | The batch stops. `MAX_COST_PER_RUN_USD` needs `MODEL_*_USD_PER_MTOK` set, otherwise only `MAX_TOKENS_PER_RUN` applies |
| Job board rate limit | Minimum delay between requests, backoff on 429 and 5xx, honouring `Retry-After` |
| Hostile or broken source | Non-public addresses are refused on every redirect hop, responses are size and time capped, posting URLs are validated, and one bad source does not stop the others |

## Known limitations

- **Discovery** reads public Greenhouse and Lever boards and direct URLs. General web search is not included, since it needs a paid search API.
- **Stuck claims:** a process killed between discovery and a terminal state leaves its posting marked in progress, so it will not be rediscovered until released.
- **ChromaDB advisories:** `pip-audit` reports PYSEC-2026-311, 3813, 3814, and 3815 against `chromadb`, with no fixed release. All four concern Chroma's HTTP server. This project uses the embedded client and never starts that server, so they are not reachable here. Do not expose a Chroma server with this version.
- **Tracing** sends posting text and candidate facts to LangSmith. Set `LANGSMITH_TRACING=false` to keep them local.
- **Address filtering** resolves a host before connecting, so a small DNS-rebinding window remains.
- **Hashes:** the lockfile pins versions but not hashes, because it is resolved on one platform and used on another.

## Deliberately out of scope

| Not built | Why |
| --- | --- |
| Alerting and paging | The digest covers a single-user queue |
| Horizontal scaling and multi-worker concurrency | One user, one process. SQLite and the in-process locks assume this |
| Escalation beyond the daily digest | Not needed at this scale |
| Automated red-teaming of the injection boundary | The boundary and fact check are tested against known attacks, not fuzzed |
| Cost and latency dashboards | Raw LangSmith traces are enough |
| Multi-user support | Single-tenant by design, with one shared review token |
