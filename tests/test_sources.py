"""The job sources beyond Greenhouse and Lever, shared filters, and the board finder."""

import logging
from pathlib import Path

import httpx
import pytest

from boards import boards_in_text, find_boards, name_candidates
from graph.nodes.discover import (
    FetchLog,
    PoliteClient,
    Query,
    SearchConfig,
    fetch_postings,
)


def client_for(handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return PoliteClient(http, delay_seconds=0, sleep=lambda _: None)


def fetch(query, handler, fetch_log=None):
    return list(fetch_postings([query], client_for(handler), fetch_log))


def test_ashby_postings_carry_company_and_remote_locations():
    def handler(request):
        assert request.url.path == "/posting-api/job-board/acme"
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Backend Engineer",
                        "location": "City A",
                        "secondaryLocations": [{"location": "City B"}],
                        "isRemote": True,
                        "isListed": True,
                        "jobUrl": "https://jobs.example.com/acme/1",
                        "descriptionPlain": "Build services",
                    },
                    {"title": "Hidden", "isListed": False, "jobUrl": "https://jobs.example.com/2"},
                ]
            },
        )

    [posting] = fetch(Query(source="ashby", board="acme"), handler)
    assert posting.url == "https://jobs.example.com/acme/1"
    assert posting.text.startswith(
        "Backend Engineer\nLocation: City A; City B; Remote\nCompany: acme\n\nBuild services"
    )


def test_workable_postings_use_the_account_name_and_strip_markup():
    def handler(request):
        assert request.url.params["details"] == "true"
        return httpx.Response(
            200,
            json={
                "name": "Acme Labs",
                "jobs": [
                    {
                        "title": "Developer",
                        "url": "https://apply.example.com/j/1",
                        "locations": [{"city": "City A", "country": "Country A"}],
                        "telecommuting": False,
                        "description": "<p>Write code</p><script>x()</script>",
                    }
                ],
            },
        )

    [posting] = fetch(Query(source="workable", board="acme"), handler)
    assert "Location: City A, Country A\nCompany: Acme Labs" in posting.text
    assert "Write code" in posting.text
    assert "x()" not in posting.text


def test_smartrecruiters_opens_only_postings_that_pass_the_filters():
    opened = []

    def handler(request):
        if request.url.path.endswith("/postings"):
            return httpx.Response(
                200,
                json={
                    "totalFound": 3,
                    "content": [
                        {
                            "id": "111",
                            "name": "Software Engineer",
                            "location": {"fullLocation": "street, , City A, , Country A"},
                        },
                        {
                            "id": "222",
                            "name": "Senior Engineer",
                            "location": {"fullLocation": "City A"},
                        },
                        {
                            "id": "../x",
                            "name": "Software Engineer",
                            "location": {"fullLocation": "City A"},
                        },
                    ],
                },
            )
        opened.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "postingUrl": "https://jobs.example.com/acme/111",
                "jobAd": {"sections": {"jobDescription": {"text": "<p>Ship features</p>"}}},
            },
        )

    query = Query(source="smartrecruiters", board="acme", exclude_keywords=["senior"])
    [posting] = fetch(query, handler)
    assert opened == ["/v1/companies/acme/postings/111"]
    assert posting.url == "https://jobs.example.com/acme/111"
    assert "Location: street, City A, Country A\n" in posting.text
    assert "Ship features" in posting.text


def test_remotive_credits_the_source_and_passes_worldwide_postings():
    def handler(request):
        assert request.url.params["search"] == "python"
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Python Developer",
                        "company_name": "Acme",
                        "candidate_required_location": "Worldwide",
                        "url": "https://remote.example.com/1",
                        "description": "<p>Remote work</p>",
                    },
                    {
                        "title": "Python Developer",
                        "company_name": "Other",
                        "candidate_required_location": "Region B only",
                        "url": "https://remote.example.com/2",
                        "description": "",
                    },
                ]
            },
        )

    query = Query(source="remotive", search="python", locations=["region a"])
    [posting] = fetch(query, handler)
    assert posting.url == "https://remote.example.com/1"
    assert "Location: Remote (Worldwide)\nCompany: Acme" in posting.text
    assert "Found through Remotive." in posting.text


def test_remotive_is_fetched_at_most_once_per_interval(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"jobs": []})

    now = [1000.0]
    log = FetchLog(tmp_path / "fetches.json", clock=lambda: now[0])
    queries = [Query(source="remotive", search="python"), Query(source="remotive", search="go")]
    list(fetch_postings(queries, client_for(handler), log))
    assert len(calls) == 2  # both searches in the first run
    list(fetch_postings(queries, client_for(handler), FetchLog(log._path, clock=lambda: now[0])))
    assert len(calls) == 2  # a second run soon after is skipped, even from a new process
    now[0] += 6 * 3600
    list(fetch_postings(queries, client_for(handler), FetchLog(log._path, clock=lambda: now[0])))
    assert len(calls) == 4


def test_adzuna_needs_keys_and_marks_postings_as_summaries(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    query = Query(source="adzuna", search="developer", country="in")
    assert fetch(query, lambda request: httpx.Response(200)) == []

    monkeypatch.setenv("ADZUNA_APP_ID", "id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "key")

    def handler(request):
        assert request.url.path == "/v1/api/jobs/in/search/1"
        assert request.url.params["what"] == "developer"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "<strong>Developer</strong>",
                        "location": {"display_name": "City A"},
                        "company": {"display_name": "Acme"},
                        "description": "Short summary",
                        "redirect_url": "https://ads.example.com/1",
                    }
                ]
            },
        )

    [posting] = fetch(query, handler)
    assert posting.text.startswith("Developer\nLocation: City A\nCompany: Acme")
    assert "Only a short summary" in posting.text


def test_adzuna_errors_never_log_the_key(monkeypatch, caplog):
    monkeypatch.setenv("ADZUNA_APP_ID", "id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "very-secret-key")
    query = Query(source="adzuna", search="developer", country="in")
    with caplog.at_level(logging.ERROR):
        assert fetch(query, lambda request: httpx.Response(401)) == []
    assert "Adzuna request failed: 401" in caplog.text
    assert "very-secret-key" not in caplog.text


def test_the_example_search_config_is_valid():
    path = Path(__file__).parents[1] / "config" / "search.example.json"
    config = SearchConfig.model_validate_json(path.read_text(encoding="utf-8"))
    assert {query.source for query in config.resolved_queries()} >= {
        "workday",
        "remotive",
        "adzuna",
        "directory",
    }


def test_top_level_filters_apply_unless_a_query_sets_its_own():
    config = SearchConfig(
        target_profile="p",
        exclude_keywords=["senior"],
        locations=["region a"],
        queries=[
            Query(source="lever", board="a"),
            Query(source="lever", board="b", locations=["region b"]),
        ],
    )
    first, second = config.resolved_queries()
    assert first.exclude_keywords == ["senior"]
    assert first.locations == ["region a"]
    assert second.locations == ["region b"]
    assert second.exclude_keywords == ["senior"]


# Board finder


@pytest.mark.parametrize(
    "text, expected",
    [
        ("https://boards.greenhouse.io/acme", [("greenhouse", "acme")]),
        ("https://job-boards.greenhouse.io/acme/jobs/123", [("greenhouse", "acme")]),
        (
            '<script src="https://boards.greenhouse.io/embed/job_board/js?for=acme">',
            [("greenhouse", "acme")],
        ),
        ("https://jobs.lever.co/acme-labs/abc", [("lever", "acme-labs")]),
        ("https://jobs.ashbyhq.com/acme", [("ashby", "acme")]),
        ("https://apply.workable.com/acme/j/ABC123", [("workable", "acme")]),
        ("https://apply.workable.com/j/ABC123", []),
        ("https://jobs.smartrecruiters.com/AcmeGroup/123", [("smartrecruiters", "AcmeGroup")]),
        ("https://example.com/careers", []),
    ],
)
def test_board_references_are_read_from_links_and_markup(text, expected):
    assert boards_in_text(text) == expected


def test_company_names_become_likely_board_names():
    assert name_candidates("Acme Labs") == ["acmelabs", "acme-labs", "AcmeLabs", "acme"]
    assert name_candidates("  ") == []


def test_find_boards_by_name_keeps_only_boards_with_jobs():
    def handler(request):
        host = request.url.host
        if host == "api.lever.co" and request.url.path.endswith("/acme"):
            return httpx.Response(200, json=[{"text": "Engineer"}, {"text": "Designer"}])
        if host == "api.smartrecruiters.com":
            return httpx.Response(200, json={"totalFound": 0, "content": []})
        return httpx.Response(404)

    matches = find_boards("Acme", client_for(handler))
    assert [(m.source, m.board, m.open_jobs) for m in matches] == [("lever", "acme", 2)]
    assert matches[0].sample_titles == ("Engineer", "Designer")


def test_one_board_found_under_two_spellings_is_listed_once():
    def handler(request):
        if request.url.host == "api.lever.co":
            return httpx.Response(200, json=[{"text": "Engineer"}])
        return httpx.Response(404)

    matches = find_boards("Acme", client_for(handler))
    assert [(m.source, m.board) for m in matches] == [("lever", "acme")]


def test_a_missing_board_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = PoliteClient(http, delay_seconds=0, sleep=lambda _: None, attempts=1)
    assert find_boards("https://jobs.lever.co/acme", client) == []
    assert len(calls) == 1


def test_find_boards_reads_a_careers_page_for_its_board():
    def handler(request):
        if request.url.host == "careers.example.com":
            return httpx.Response(
                200, text='<a href="https://jobs.ashbyhq.com/acme">Open roles</a>'
            )
        return httpx.Response(200, json={"jobs": [{"title": "Engineer"}]})

    [match] = find_boards("https://careers.example.com/", client_for(handler))
    assert (match.source, match.board) == ("ashby", "acme")
