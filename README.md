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

## Quick start

Requires Python 3.14. From this folder, run `start.bat` on Windows or `./start.sh` on macOS and Linux. The first start creates a virtual environment and installs dependencies, then every start opens the agent in your browser. `python app.py` does the same once the environment exists.

The first visit goes to the setup page, which walks through four steps:

1. **Model.** Pick a provider and model and paste its key. The key is written to `.env` and never shown again.
2. **About you.** Upload your resume as PDF or text. The model suggests facts copied from it, and any fact using a name or number not found in the resume is flagged. Nothing is saved until you have checked, edited, and confirmed the list. Email addresses and phone numbers are left out. These facts are the only things a draft may say about you.
3. **What you're looking for.** A plain-language description of the roles you want, plus keyword, excluded keyword, and location filters. Filters run before any model call, so they save money.
4. **Where to look.** Any mix of four kinds of source, all filtered by step 3 before any model call:
   - **Job sites** listing many employers, searched with your search terms on every run. Twelve need no account, and five more need a free key you can paste on the same page.
   - **Every company on a board system**, such as every Greenhouse or Workable board the agent knows of. See [Scanning every company](#scanning-every-company).
   - **A specific company**, found from its name or careers page link, including Workday careers sites.
   - **A single posting link.**

A last section sets the fit threshold, how many postings one run may check, a spend limit, model prices for cost tracking, and an optional schedule.

Everything stays on your machine in `data/` and `.env`, both gitignored. The only data that leaves is what goes to your model provider, and to LangSmith if tracing is on.

### Setting up by hand

The setup page writes two files you can also edit directly. Copy `config/facts.example.json` and `config/search.example.json` into `data/`, and `.env.example` to `.env`.

`search.json` takes these fields:

| Field | Meaning |
| --- | --- |
| `target_profile` | The roles you want, in plain words. Every posting is scored against it |
| `keywords`, `exclude_keywords`, `locations` | Title and location filters every source inherits, unless a query sets its own. Locations containing "worldwide" or "anywhere" always pass |
| `job_sites` | Names of job sites to search on every run, from the table below |
| `search_terms` | Up to ten terms sent to each job site's search. Defaults to `keywords`. Sites without a search filter their titles by these terms instead |
| `country`, `where` | A two-letter country code and a city or region, for sites that search one place |
| `scan_companies` | Board systems whose every known company is checked, such as `["greenhouse", "workable"]`. Needs `keywords` |
| `companies_per_run` | How many of those companies to check per run. Default 300 |
| `queries` | Individual sources: a company board, a Workday careers link, or a posting link |

Job sites, by name:

| Name | Covers | Key |
| --- | --- | --- |
| `remotive`, `remoteok`, `weworkremotely`, `himalayas`, `jobicy`, `workingnomads` | Remote jobs, worldwide | None |
| `hackernews` | Each month's "Who is hiring?" thread, mostly startups | None |
| `fourdayweek` | Jobs with a four-day week | None |
| `arbeitnow` | Germany and Europe | None |
| `landingjobs` | Technology jobs, Portugal and Europe | None |
| `getonbrd` | Technology jobs, Latin America | None |
| `jobtech` | Sweden's public employment service | None |
| `adzuna` | One country at a time, about 20 supported. Summaries only | `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` |
| `jooble` | Most countries. Summaries only | `JOOBLE_API_KEY` |
| `usajobs` | United States government | `USAJOBS_API_KEY`, `USAJOBS_EMAIL` |
| `reed` | United Kingdom. Summaries only | `REED_API_KEY` |
| `findwork` | Software jobs | `FINDWORK_API_KEY` |

Each job site asks to be credited and linked, so every posting says where it was found and keeps that site's link. Sites that ask for infrequent requests are fetched at most hourly, Remotive and Hacker News at most every six hours.

Queries, by source:

| Source | Fields | Notes |
| --- | --- | --- |
| `greenhouse`, `lever`, `ashby`, `workable`, `smartrecruiters`, `recruitee`, `breezy`, `rippling`, `gem` | `board` | A company's public board. `python cli.py find-board "Company"` finds it |
| `workday` | `url`, optional `search` | A Workday careers link, such as `https://acme.wd5.myworkdayjobs.com/Careers` |
| any job site | `search`, `country`, `where` | One search on one site, if you want it separate from `job_sites` |
| `url` | `url` | One posting page |

### Scanning every company

No board system offers a search across all its companies, so the agent builds its own list from [Common Crawl](https://commoncrawl.org), a free public archive of the web: every board page the crawl has seen names its company. The list covers Greenhouse, Lever, Ashby, Workable, SmartRecruiters, Recruitee, and Breezy HR, typically several thousand companies in all, and is stored in `data/board_directory.json` and refreshed monthly.

Build it with the button on the setup page or `python cli.py directory`. Otherwise the first run that needs it builds it. Building takes a few minutes, because Common Crawl's index is slow and often busy. A page it fails to answer is retried and then skipped, so counts vary between builds, and the next refresh adds what was missed.

Each run checks the companies not checked in the last 12 hours, those never checked first, alternating between systems. Titles are read first, and only postings that pass your filters are opened. Each board system sees about one request a second. A company whose board has gone is skipped for a month.

## Running

Start a run from the browser page, or from the terminal:

```bash
python cli.py run                 # search, score, and draft
python cli.py jobs                # every posting found, best fit first
python cli.py jobs --csv jobs.csv # the same list as a spreadsheet
python cli.py pending             # drafts waiting for your decision
python cli.py decide <run_id> approved
python cli.py decide <run_id> rejected --reason "Too much on-call"
python cli.py digest              # drafts waiting past REVIEW_STALE_HOURS
python cli.py outcome <url> "interview"
python cli.py release <url>       # retry a posting that failed
python cli.py rescore             # score ruled-out postings again, after changing your profile
python cli.py find-board "Acme"   # find a company's job board by name or careers link
python cli.py directory           # build or show the list of companies on each board system
```

A rejection needs a reason, because it becomes precedent for future scoring. Approved drafts land in `data/drafts/`. With `DRAFT_BACKEND=gmail` and OAuth credentials using the `gmail.compose` scope, they become Gmail drafts instead.

### Browser page

The jobs page starts a run and follows its progress, including tokens used and, once prices are set, the cost so far. It lists every posting found with its company, location, score, the reasoning behind the score, and the site it came from. Postings new since your last visit are marked, and the tab title shows how many drafts wait for review. The list downloads as a CSV that opens in Excel.

Reviewing drafts stays in the terminal, so the page cannot approve anything.

With a schedule set on the setup page, the agent also starts a run on its own every 6 to 168 hours while it is open. Runs never overlap, and each run stops at the spend limit.

The page has no login, since it is a personal tool. Instead it only answers requests addressed to `127.0.0.1` or `localhost`, and only accepts changes made from its own pages. That keeps other websites open in your browser from reading your data, changing your settings, or starting paid runs. Do not put it behind a proxy or expose it on a network.

To run it in a container, mount your data and pass your settings. The image contains neither:

```bash
docker build -t job-search-agent .
docker run -p 127.0.0.1:8000:8000 --env-file .env -v "$PWD/data:/app/data" job-search-agent
```

In a container the setup page can save facts, search, and sources, which live in `data/`, but not the model or preferences, which come from your env file. Review drafts with `docker exec -it <container> python cli.py pending`.

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
| Tokens | 23379 in, 3690 out |
| LangSmith | 14 calls, $0.0780, mean latency 3.1s |

Model: `anthropic/claude-sonnet-5`. Raw output is in `evals/results/`.

Read this as a smoke test, not an accuracy figure. The sample has 10 hand-written, clear-cut cases. A meaningful number needs a real decision history replayed with `--tracker`.

Run the unit tests with `pytest`. They use a scripted model and need no keys or network. Run the eval before shipping any prompt change.

## Failure handling

| Failure | Behaviour |
| --- | --- |
| Model call fails or returns malformed output | Three attempts with backoff, then dead-letter, shown as Failed by `cli.py jobs`. The batch continues |
| Posting seen before | Skipped by canonical URL before any model call |
| Restart during a pending review | The review is restored from SQLite and resumes without repeating earlier nodes |
| Review left unanswered | Listed by the digest after `REVIEW_STALE_HOURS` |
| Spend ceiling reached | The batch stops. `MAX_COST_PER_RUN_USD` needs `MODEL_*_USD_PER_MTOK` set, otherwise only `MAX_TOKENS_PER_RUN` applies |
| Job board rate limit | Minimum delay between requests to each host, backoff on 429 and 5xx, honouring `Retry-After` |
| Job site key missing or refused | That site is skipped and logged without the key. The others continue |
| Hostile or broken source | Non-public addresses are refused on every redirect hop, responses are size and time capped, posting URLs are validated, and one bad source does not stop the others |

## Known limitations

- **Sites not covered:** LinkedIn, Indeed, Naukri, Glassdoor, Monster, and Wellfound are not read, because their terms forbid automated collection or they offer no public API. iCIMS, Taleo, SuccessFactors, and Personio boards are not read either, for lack of a public feed this project could verify. General web search is not included, since it needs a paid search API.
- **Summaries only:** Adzuna, Jooble, and Reed share a short summary of each posting, so scores from them rest on less text. The posting text tells the model this.
- **Company list coverage** depends on what Common Crawl has seen. A company whose board page the crawl missed is not in the list until a later crawl finds it, but it can still be added by name. Lever boards in particular are often missing, because the index frequently fails to answer for them.
- **Keyed sites** were built from each provider's documented response format and tested against recorded responses, but not against the live service, since that needs a personal key.
- **Board finder** cannot read careers pages rendered by JavaScript. Use the company name instead, or open the page and copy the link to its job board.
- **Outreach drafts** have no recipient. The agent does not look up hiring managers or email addresses.
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
| Multi-user support | A personal tool, run on your own machine |
