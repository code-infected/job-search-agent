"""Every job source the agent can read, and what each one needs.

Sites behind a login or whose terms forbid automated collection, such as LinkedIn,
Indeed, Naukri, and Glassdoor, are deliberately absent.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal, Optional

from graph.sources import ats, feeds, keyed

HOUR = 3600.0


@dataclass(frozen=True)
class Source:
    label: str
    # board: one company's board by name. link: one company's careers link.
    # site: a job site listing many employers. url: one posting page.
    # directory: every company found on the listed board systems.
    kind: Literal["board", "link", "site", "url", "directory"]
    fetch: Optional[Callable] = None
    region: str = ""
    search: Literal["required", "optional", "none"] = "none"
    needs_country: bool = False
    env: tuple[str, ...] = ()
    key_url: str = ""
    # Some sites ask for infrequent fetches. Zero means every run.
    min_interval_seconds: float = 0.0
    summary_only: bool = False

    def describe(self, name: str) -> dict:
        details = asdict(self)
        details.pop("fetch")
        return {"name": name, **details}


SOURCES: dict[str, Source] = {
    # One company's board on an applicant tracking system.
    "greenhouse": Source("Greenhouse", "board", ats.greenhouse),
    "lever": Source("Lever", "board", ats.lever),
    "ashby": Source("Ashby", "board", ats.ashby),
    "workable": Source("Workable", "board", ats.workable),
    "smartrecruiters": Source("SmartRecruiters", "board", ats.smartrecruiters),
    "recruitee": Source("Recruitee", "board", ats.recruitee),
    "breezy": Source("Breezy HR", "board", ats.breezy),
    "rippling": Source("Rippling", "board", ats.rippling),
    "gem": Source("Gem", "board", ats.gem),
    "workday": Source("Workday", "link", ats.workday, search="optional"),
    "url": Source("Posting link", "url"),
    # Job sites listing many employers, free and without an account.
    "remotive": Source(
        "Remotive",
        "site",
        feeds.remotive,
        "Remote, worldwide",
        search="required",
        min_interval_seconds=6 * HOUR,  # Remotive asks for at most four fetches a day
    ),
    "remoteok": Source(
        "Remote OK", "site", feeds.remoteok, "Remote, worldwide", min_interval_seconds=HOUR
    ),
    "weworkremotely": Source(
        "We Work Remotely",
        "site",
        feeds.we_work_remotely,
        "Remote, worldwide",
        min_interval_seconds=HOUR,
    ),
    "himalayas": Source(
        "Himalayas",
        "site",
        feeds.himalayas,
        "Remote, worldwide",
        search="optional",
        min_interval_seconds=HOUR,
    ),
    "jobicy": Source(
        "Jobicy",
        "site",
        feeds.jobicy,
        "Remote, worldwide",
        search="optional",
        min_interval_seconds=HOUR,
    ),
    "workingnomads": Source(
        "Working Nomads",
        "site",
        feeds.working_nomads,
        "Remote, worldwide",
        min_interval_seconds=HOUR,
    ),
    "hackernews": Source(
        "Hacker News: Who is hiring?",
        "site",
        feeds.hacker_news,
        "Startups, worldwide, posted monthly",
        min_interval_seconds=6 * HOUR,
    ),
    "fourdayweek": Source(
        "4 Day Week",
        "site",
        feeds.four_day_week,
        "Four-day weeks, worldwide",
        min_interval_seconds=HOUR,
    ),
    "arbeitnow": Source(
        "Arbeitnow", "site", feeds.arbeitnow, "Germany and Europe", min_interval_seconds=HOUR
    ),
    "landingjobs": Source(
        "Landing.jobs",
        "site",
        feeds.landing_jobs,
        "Technology, Portugal and Europe",
        min_interval_seconds=HOUR,
    ),
    "getonbrd": Source(
        "Get on Board",
        "site",
        feeds.get_on_board,
        "Technology, Latin America",
        search="optional",
        min_interval_seconds=HOUR,
    ),
    "jobtech": Source(
        "Arbetsförmedlingen",
        "site",
        feeds.jobtech,
        "Sweden",
        search="optional",
        min_interval_seconds=HOUR,
    ),
    # Job search APIs that need a free key.
    "adzuna": Source(
        "Adzuna",
        "site",
        keyed.adzuna,
        "One country, about 20 supported",
        search="required",
        needs_country=True,
        env=("ADZUNA_APP_ID", "ADZUNA_APP_KEY"),
        key_url="https://developer.adzuna.com/",
        summary_only=True,
    ),
    "jooble": Source(
        "Jooble",
        "site",
        keyed.jooble,
        "Most countries",
        search="required",
        env=("JOOBLE_API_KEY",),
        key_url="https://jooble.org/api/about",
        summary_only=True,
    ),
    "usajobs": Source(
        "USAJobs",
        "site",
        keyed.usajobs,
        "United States government",
        search="required",
        env=("USAJOBS_API_KEY", "USAJOBS_EMAIL"),
        key_url="https://developer.usajobs.gov/apirequest/",
    ),
    "reed": Source(
        "Reed",
        "site",
        keyed.reed,
        "United Kingdom",
        search="required",
        env=("REED_API_KEY",),
        key_url="https://www.reed.co.uk/developers/jobseeker",
        summary_only=True,
    ),
    "findwork": Source(
        "Findwork",
        "site",
        keyed.findwork,
        "Software jobs, worldwide",
        search="required",
        env=("FINDWORK_API_KEY",),
        key_url="https://findwork.dev/developers/",
    ),
    "directory": Source("Every company on a board system", "directory"),
}

BOARD_SOURCES = tuple(name for name, source in SOURCES.items() if source.kind == "board")
SITE_SOURCES = tuple(name for name, source in SOURCES.items() if source.kind == "site")
KEY_NAMES = frozenset(name for source in SOURCES.values() for name in source.env)
