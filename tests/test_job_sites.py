"""Job sites, more board systems, keyed search APIs, and the company directory.

Payloads follow the shape each live API returned when these sources were added.
"""

import json
import logging

import httpx
import pytest

from boards import boards_in_text, find_boards
from graph.nodes.discover import FetchLog, PoliteClient, Query, SearchConfig, fetch_postings
from graph.sources import SITE_SOURCES, SOURCES
from graph.sources.ats import workday_links, workday_site
from graph.sources.directory import RESCAN_SECONDS, BoardDirectory


def client_for(handler, sleep=lambda _: None, delay=0):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return PoliteClient(http, delay_seconds=delay, sleep=sleep)


def fetch(query, handler, **kwargs):
    return list(fetch_postings([query], client_for(handler), **kwargs))


def engineer(source, **fields):
    return Query(source=source, keywords=["engineer"], **fields)


# Job sites without an account


def test_arbeitnow_filters_titles_and_follows_pages():
    pages = []

    def handler(request):
        page = request.url.params["page"]
        pages.append(page)
        jobs = [
            {
                "title": "Backend Engineer",
                "company_name": "Acme",
                "location": "City A",
                "remote": True,
                "url": f"https://jobs.example.com/{page}",
                "description": "<p>Build</p>",
            },
            {"title": "Marketing Lead", "url": "https://jobs.example.com/x", "description": ""},
        ]
        return httpx.Response(200, json={"data": jobs, "links": {"next": page == "1" or None}})

    postings = fetch(engineer("arbeitnow"), handler)
    assert pages == ["1", "2"]
    assert [p.url for p in postings] == ["https://jobs.example.com/1", "https://jobs.example.com/2"]
    assert "Location: City A; Remote\nCompany: Acme" in postings[0].text
    assert "Found through Arbeitnow." in postings[0].text


def test_remoteok_skips_its_terms_notice():
    def handler(request):
        return httpx.Response(
            200,
            json=[
                {"legal": "Please link back"},
                {
                    "position": "Platform Engineer",
                    "company": "Acme",
                    "location": "",
                    "url": "https://remote.example.com/1",
                    "description": "<p>Run things</p>",
                },
            ],
        )

    [posting] = fetch(engineer("remoteok"), handler)
    assert "Location: Remote (Worldwide)\nCompany: Acme" in posting.text
    assert "Found through Remote OK." in posting.text


def test_jobicy_and_himalayas_send_the_search_terms():
    def handler(request):
        if request.url.host == "jobicy.com":
            assert request.url.params["tag"] == "python"
            jobs = [
                {
                    "jobTitle": "Python Engineer",
                    "companyName": "Acme",
                    "jobGeo": "Region A",
                    "url": "https://jobicy.example.com/1",
                    "jobDescription": "<p>Code</p>",
                }
            ]
            return httpx.Response(200, json={"jobs": jobs})
        assert request.url.params["q"] == "python"
        jobs = [
            {
                "title": "Python Engineer",
                "companyName": "Beta",
                "locationRestrictions": ["Region A", "Region B"],
                "applicationLink": "https://himalayas.example.com/1",
                "guid": "g",
                "description": "",
            }
        ]
        return httpx.Response(200, json={"jobs": jobs})

    [jobicy] = fetch(engineer("jobicy", search="python"), handler)
    assert "Location: Remote (Region A)" in jobicy.text
    [himalayas] = fetch(engineer("himalayas", search="python"), handler)
    assert "Location: Remote (Region A, Region B)\nCompany: Beta" in himalayas.text


def test_we_work_remotely_reads_company_and_role_from_the_rss_title():
    rss = """<?xml version="1.0"?><rss><channel>
      <item><title>Acme: Senior Engineer</title><region>Anywhere in the World</region>
        <link>https://wwr.example.com/1</link><description>&lt;p&gt;Do work&lt;/p&gt;</description></item>
      <item><title>Beta: Software Engineer</title><region>Region A</region>
        <link>https://wwr.example.com/2</link><description>Build</description></item>
    </channel></rss>"""

    query = engineer("weworkremotely", exclude_keywords=["senior"])
    [posting] = fetch(query, lambda request: httpx.Response(200, text=rss))
    assert posting.url == "https://wwr.example.com/2"
    assert posting.text.startswith("Software Engineer\nLocation: Remote (Region A)\nCompany: Beta")


def test_rss_entity_expansion_is_refused():
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaaaaaaaa">'
        '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]><rss><channel><item>'
        "<title>&b;</title></item></channel></rss>"
    )
    assert fetch(engineer("weworkremotely"), lambda request: httpx.Response(200, text=bomb)) == []


def test_hacker_news_reads_replies_to_the_latest_hiring_thread():
    def handler(request):
        if request.url.path.endswith("search_by_date"):
            hits = [
                {"objectID": "2", "title": "Ask HN: Who wants to be hired? (Month)"},
                {"objectID": "1", "title": "Ask HN: Who is hiring? (Month)"},
            ]
            return httpx.Response(200, json={"hits": hits})
        assert request.url.path == "/api/v1/items/1"
        children = [
            {"id": 11, "text": "Acme | Backend Engineer | Remote (Worldwide)<p>We build tools."},
            {"id": 12, "text": "Beta | Designer | City A<p>Draw things."},
            {"id": 13, "text": None},
        ]
        return httpx.Response(200, json={"children": children})

    [posting] = fetch(engineer("hackernews"), handler)
    assert posting.url == "https://news.ycombinator.com/item?id=11"
    assert posting.text.startswith("Acme | Backend Engineer | Remote (Worldwide)\n")
    assert "Company: Acme" in posting.text
    assert "We build tools." in posting.text


@pytest.mark.parametrize(
    "source, payload, url, location",
    [
        (
            "getonbrd",
            {
                "data": [
                    {
                        "attributes": {
                            "title": "Data Engineer",
                            "countries": ["Country A"],
                            "remote": True,
                            "description": "<p>Pipelines</p>",
                        },
                        "links": {"public_url": "https://gob.example.com/1"},
                    }
                ]
            },
            "https://gob.example.com/1",
            "Country A; Remote",
        ),
        (
            "landingjobs",
            [
                {
                    "title": "Java Engineer",
                    "locations": [{"city": "City A", "country_code": "PT"}],
                    "remote": False,
                    "url": "https://lj.example.com/1",
                    "role_description": "<p>Services</p>",
                }
            ],
            "https://lj.example.com/1",
            "City A, PT",
        ),
        (
            "jobtech",
            {
                "hits": [
                    {
                        "headline": "AI Engineer",
                        "workplace_address": {"municipality": "City A", "country": "Sverige"},
                        "webpage_url": "https://jt.example.com/1",
                        "description": {"text": "Models"},
                        "employer": {"name": "Acme AB"},
                    }
                ]
            },
            "https://jt.example.com/1",
            "City A, Sverige",
        ),
        (
            "fourdayweek",
            {
                "jobs": [
                    {
                        "title": "QA Engineer",
                        "url": "https://4dw.example.com/1",
                        "is_remote": True,
                        "location_original": "City A",
                        "description": "Tests",
                        "company_name": "Acme",
                    }
                ],
                "has_more": False,
            },
            "https://4dw.example.com/1",
            "City A; Remote",
        ),
        (
            "workingnomads",
            [
                {
                    "title": "Support Engineer",
                    "location": "Europe",
                    "url": "https://wn.example.com/1",
                    "company_name": "Acme",
                    "description": "Help",
                }
            ],
            "https://wn.example.com/1",
            "Remote (Europe)",
        ),
    ],
)
def test_regional_and_remote_sites_are_parsed(source, payload, url, location):
    [posting] = fetch(engineer(source), lambda request: httpx.Response(200, json=payload))
    assert posting.url == url
    assert f"Location: {location}\n" in posting.text
    assert "Found through" in posting.text


# More board systems


def test_recruitee_skips_unpublished_offers():
    def handler(request):
        assert request.url.host == "acme.recruitee.com"
        offers = [
            {
                "title": "Software Engineer",
                "location": "City A",
                "remote": True,
                "careers_url": "https://careers.example.com/o/1",
                "company_name": "Acme",
                "description": "<p>Build</p>",
                "requirements": "<p>Python</p>",
                "status": "published",
            },
            {"title": "Engineer", "status": "draft", "careers_url": "https://x.example.com"},
        ]
        return httpx.Response(200, json={"offers": offers})

    [posting] = fetch(engineer("recruitee", board="Acme"), handler)
    assert "Location: City A; Remote\nCompany: Acme" in posting.text
    assert "Build" in posting.text and "Python" in posting.text


def test_breezy_and_rippling_open_only_matching_postings():
    opened = []

    def handler(request):
        if request.url.host == "acme.breezy.hr":
            if request.url.path == "/json":
                jobs = [
                    {
                        "name": "Field Engineer",
                        "url": "https://acme.breezy.hr/p/1",
                        "location": {"name": "City A", "is_remote": False},
                        "company": {"name": "Acme"},
                    },
                    {"name": "Recruiter", "url": "https://acme.breezy.hr/p/2", "location": {}},
                    {"name": "Engineer", "url": "https://elsewhere.example.com/p", "location": {}},
                ]
                return httpx.Response(200, json=jobs)
            opened.append(request.url.path)
            return httpx.Response(200, text="<main>Fix machines</main>")
        if request.url.path.endswith("/jobs"):
            items = [
                {
                    "id": "abc-1",
                    "name": "Data Engineer",
                    "url": "https://ats.example.com/acme/jobs/abc-1",
                    "locations": [{"name": "City B", "workplaceType": "REMOTE"}],
                },
                {"id": "../x", "name": "Engineer", "url": "https://ats.example.com/x"},
                {"id": "abc-2", "name": "Recruiter", "url": "https://ats.example.com/2"},
            ]
            return httpx.Response(200, json={"items": items, "totalPages": 1})
        opened.append(request.url.path)
        detail = {"description": {"role": "<p>Pipelines</p>"}, "companyName": "Acme Inc"}
        return httpx.Response(200, json=detail)

    [breezy] = fetch(engineer("breezy", board="acme"), handler)
    assert "Fix machines" in breezy.text
    [rippling] = fetch(engineer("rippling", board="acme"), handler)
    assert "Location: City B; Remote\nCompany: Acme Inc" in rippling.text
    assert opened == ["/p/1", "/api/v2/board/acme/jobs/abc-1"]


def test_gem_postings_are_parsed():
    def handler(request):
        jobs = [
            {
                "title": "Backend Engineer",
                "location": {"name": "City A"},
                "location_type": "remote",
                "absolute_url": "https://jobs.example.com/acme/1",
                "content_plain": "Ship it",
            }
        ]
        return httpx.Response(200, json=jobs)

    [posting] = fetch(engineer("gem", board="acme"), handler)
    assert "Location: City A; Remote" in posting.text


def test_workday_searches_and_checks_unnamed_locations_on_the_posting():
    searched = []

    def handler(request):
        if request.method == "POST":
            searched.append(json.loads(request.content))
            postings = [
                {
                    "title": "Software Engineer",
                    "externalPath": "/job/City-A/Software-Engineer_R1",
                    "locationsText": "City A",
                },
                {
                    "title": "Software Engineer",
                    "externalPath": "/job/Multi/Software-Engineer_R2",
                    "locationsText": "2 Locations",
                },
                {
                    "title": "Software Engineer",
                    "externalPath": "/job/City-C/Software-Engineer_R3",
                    "locationsText": "City C",
                },
                {"title": "Software Engineer", "externalPath": "https://evil.example.com"},
            ]
            return httpx.Response(200, json={"total": 4, "jobPostings": postings})
        locations = {"R1": ("City A", []), "R2": ("City C", ["City D"])}
        place, more = locations[request.url.path.rsplit("_", 1)[1]]
        info = {
            "jobDescription": "<p>Write code</p>",
            "location": place,
            "additionalLocations": more,
            "externalUrl": f"https://acme.wd5.myworkdayjobs.com/Careers{request.url.path}",
        }
        return httpx.Response(200, json={"jobPostingInfo": info})

    query = Query(
        source="workday",
        url="https://acme.wd5.myworkdayjobs.com/en-US/Careers",
        search="software",
        keywords=["engineer"],
        locations=["city a", "city d"],
    )
    postings = fetch(query, handler)
    assert searched[0]["searchText"] == "software"
    assert [p.url.rsplit("_", 1)[1] for p in postings] == ["R1", "R2"]
    assert "Location: City C; City D\nCompany: acme" in postings[1].text


@pytest.mark.parametrize(
    "url", ["https://example.com/Careers", "https://acme.wd5.myworkdayjobs.com/", "ftp://x"]
)
def test_only_workday_careers_links_are_accepted(url):
    with pytest.raises(ValueError):
        Query(source="workday", url=url)


def test_workday_links_are_read_with_their_site():
    assert workday_site("https://acme.wd5.myworkdayjobs.com/en-US/Careers/job/x") == (
        "acme.wd5.myworkdayjobs.com",
        "acme",
        "Careers",
    )
    page = '<a href="https://acme.wd1.myworkdayjobs.com/External">Jobs</a>'
    assert workday_links(page) == ["https://acme.wd1.myworkdayjobs.com/External"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("https://acme.recruitee.com/o/role", [("recruitee", "acme")]),
        ("https://www.recruitee.com/pricing", []),
        ("https://acme.breezy.hr/p/1", [("breezy", "acme")]),
        ("https://ats.rippling.com/acme/jobs", [("rippling", "acme")]),
        ("https://jobs.gem.com/acme/123", [("gem", "acme")]),
    ],
)
def test_new_board_systems_are_recognised_in_links(text, expected):
    assert boards_in_text(text) == expected


def test_board_finder_reports_a_workday_site_from_its_link():
    def handler(request):
        assert request.url.path == "/wday/cxs/acme/Careers/jobs"
        return httpx.Response(200, json={"total": 42, "jobPostings": [{"title": "Engineer"}]})

    [match] = find_boards("https://acme.wd5.myworkdayjobs.com/Careers", client_for(handler))
    assert (match.source, match.board, match.open_jobs) == (
        "workday",
        "https://acme.wd5.myworkdayjobs.com/Careers",
        42,
    )


# Keyed search APIs


@pytest.mark.parametrize("source", ["jooble", "usajobs", "reed", "findwork"])
def test_keyed_sites_without_a_key_are_skipped(source, monkeypatch, caplog):
    for name in SOURCES[source].env:
        monkeypatch.delenv(name, raising=False)
    with caplog.at_level(logging.ERROR):
        assert fetch(Query(source=source, search="developer"), lambda r: httpx.Response(200)) == []
    assert "needs" in caplog.text


def test_keyed_sites_send_keys_where_each_api_expects_them(monkeypatch):
    monkeypatch.setenv("JOOBLE_API_KEY", "jk")
    monkeypatch.setenv("USAJOBS_API_KEY", "uk")
    monkeypatch.setenv("USAJOBS_EMAIL", "me@example.com")
    monkeypatch.setenv("REED_API_KEY", "rk")
    monkeypatch.setenv("FINDWORK_API_KEY", "fk")

    def handler(request):
        host = request.url.host
        if host == "jooble.org":
            assert request.method == "POST" and request.url.path == "/api/jk"
            assert json.loads(request.content)["location"] == "City A"
            job = {
                "title": "Engineer",
                "location": "City A",
                "snippet": "Short",
                "link": "https://j.example.com/1",
                "company": "Acme",
            }  # noqa: E501
            return httpx.Response(200, json={"jobs": [job]})
        if host == "data.usajobs.gov":
            assert request.headers["Authorization-Key"] == "uk"
            assert request.headers["User-Agent"] == "me@example.com"
            job = {
                "PositionTitle": "IT Engineer",
                "PositionURI": "https://u.example.com/1",
                "OrganizationName": "Agency",
                "PositionLocationDisplay": "City A",
                "UserArea": {"Details": {"JobSummary": "Serve"}},
            }
            items = [{"MatchedObjectDescriptor": job}]
            return httpx.Response(200, json={"SearchResult": {"SearchResultItems": items}})
        if host == "www.reed.co.uk":
            assert request.headers["Authorization"].startswith("Basic ")
            assert request.url.params["locationName"] == "City A"
            job = {
                "jobTitle": "Engineer",
                "employerName": "Acme",
                "locationName": "City A",
                "jobDescription": "Short",
                "jobUrl": "https://r.example.com/1",
            }  # noqa: E501
            return httpx.Response(200, json={"results": [job]})
        assert request.headers["Authorization"] == "Token fk"
        job = {
            "role": "Engineer",
            "company_name": "Acme",
            "location": "",
            "remote": True,
            "url": "https://f.example.com/1",
            "text": "Long",
        }  # noqa: E501
        return httpx.Response(200, json={"results": [job]})

    found = {}
    for source in ("jooble", "usajobs", "reed", "findwork"):
        [posting] = fetch(engineer(source, search="engineer", where="City A"), handler)
        found[source] = posting.text
    assert "Only a short summary" in found["jooble"]
    assert "Company: Agency" in found["usajobs"] and "Serve" in found["usajobs"]
    assert "Only a short summary" in found["reed"]
    assert "Location: Remote" in found["findwork"]


def test_keyed_failures_never_repeat_the_key(monkeypatch, caplog):
    monkeypatch.setenv("JOOBLE_API_KEY", "very-secret-key")
    with caplog.at_level(logging.ERROR):
        assert fetch(Query(source="jooble", search="x"), lambda r: httpx.Response(403)) == []
    assert "Jooble request failed: 403" in caplog.text
    assert "very-secret-key" not in caplog.text


# Search settings shared by every site


def test_job_sites_are_searched_with_the_shared_terms():
    config = SearchConfig(
        target_profile="p",
        keywords=["engineer"],
        search_terms=["python developer", "data engineer"],
        country="in",
        where="City A",
        job_sites=["remotive", "remoteok", "adzuna"],
    )
    queries = [(q.source, q.search, q.country, q.where) for q in config.resolved_queries()]
    assert queries == [
        ("remotive", "python developer", "in", "City A"),
        ("remotive", "data engineer", "in", "City A"),
        ("remoteok", None, "in", "City A"),
        ("adzuna", "python developer", "in", "City A"),
        ("adzuna", "data engineer", "in", "City A"),
    ]
    assert all(q.keywords == ["engineer"] for q in config.resolved_queries())


def test_sites_without_search_filter_titles_by_the_search_terms():
    config = SearchConfig(target_profile="p", search_terms=["python"], job_sites=["remoteok"])
    [query] = config.resolved_queries()
    assert query.keywords == ["python"]


def test_sites_that_need_a_country_or_terms_are_skipped_without_them():
    config = SearchConfig(target_profile="p", job_sites=["adzuna", "remotive", "himalayas"])
    assert [(q.source, q.search) for q in config.resolved_queries()] == [("himalayas", None)]


def test_unknown_sites_and_unfiltered_company_scans_are_refused():
    with pytest.raises(ValueError, match="unknown job sites"):
        SearchConfig(target_profile="p", job_sites=["linkedin"])
    with pytest.raises(ValueError, match="title keywords"):
        SearchConfig(target_profile="p", scan_companies=["greenhouse"])


def test_every_site_source_has_a_label_and_known_kind():
    assert len(SITE_SOURCES) >= 15
    for name in SITE_SOURCES:
        assert SOURCES[name].label and SOURCES[name].fetch


def test_requests_wait_per_host_so_alternating_boards_do_not_queue():
    waits = []
    client = client_for(lambda request: httpx.Response(200), sleep=waits.append, delay=5)
    client.get("https://a.example.com/1")
    client.get("https://b.example.com/1")
    assert waits == []
    client.get("https://a.example.com/2")
    assert len(waits) == 1


def test_any_site_with_an_interval_is_fetched_at_most_that_often(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(200, json=[{"legal": "x"}])

    log = FetchLog(tmp_path / "fetches.json", clock=lambda: 1000.0)
    fetch(engineer("remoteok"), handler, fetch_log=log)
    fetch(engineer("remoteok"), handler, fetch_log=log)
    assert calls == ["remoteok.com"]


# Company directory


def crawl_handler(lines, calls=None):
    def handler(request):
        if calls is not None:
            calls.append(str(request.url))
        if request.url.path == "/collinfo.json":
            return httpx.Response(200, json=[{"cdx-api": "https://index.example.com/c1"}])
        if request.url.params.get("showNumPages"):
            return httpx.Response(200, json={"pages": 1})
        pattern = request.url.params["url"]
        found = [line for line in lines if pattern.split("/")[0].lstrip("*.") in line]
        return httpx.Response(200, text="\n".join(json.dumps({"url": u}) for u in found))

    return handler


def test_directory_is_built_from_the_crawl_index(tmp_path):
    lines = [
        "https://boards.greenhouse.io/acme/jobs/1",
        "https://job-boards.greenhouse.io/Acme/jobs/2",
        "https://job-boards.greenhouse.io/beta",
        "https://boards.greenhouse.io/embed/job_board?for=gamma",
        "https://jobs.lever.co/delta/abc",
    ]
    index = client_for(crawl_handler(lines))
    directory = BoardDirectory(tmp_path / "d.json", index=index, clock=lambda: 1000.0)
    assert directory.refresh(["greenhouse", "lever"])["greenhouse"] == 3
    reloaded = BoardDirectory(tmp_path / "d.json")
    assert reloaded.counts()["lever"] == 1


def test_a_dropped_index_page_is_tried_again(tmp_path):
    answer = crawl_handler(["https://jobs.lever.co/delta"])
    dropped = []

    def handler(request):
        if request.url.params.get("output") and not dropped:
            dropped.append(request)
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return answer(request)

    directory = BoardDirectory(tmp_path / "d.json", index=client_for(handler), clock=lambda: 1.0)
    assert directory.refresh(["lever"])["lever"] == 1


def test_directory_is_not_rebuilt_within_a_month(tmp_path):
    calls = []
    index = client_for(crawl_handler(["https://jobs.lever.co/delta"], calls))
    directory = BoardDirectory(tmp_path / "d.json", index=index, clock=lambda: 1000.0)
    directory.refresh(["lever"])
    made = len(calls)
    directory.refresh(["lever"])
    assert len(calls) == made


def seeded_directory(tmp_path, client, now):
    path = tmp_path / "d.json"
    boards = {"greenhouse": ["g1", "g2", "g3"], "lever": ["l1"]}
    built = {"greenhouse": now[0], "lever": now[0]}
    path.write_text(json.dumps({"boards": boards, "built": built}), encoding="utf-8")
    return BoardDirectory(path, client=client, clock=lambda: now[0])


def test_scans_alternate_systems_and_skip_recent_and_missing_boards(tmp_path):
    now = [10_000.0]

    def handler(request):
        if "g2" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json={"jobs": []} if "greenhouse" in request.url.host else [])

    directory = seeded_directory(tmp_path, client_for(handler), now)
    assert directory.due(["greenhouse", "lever"], 10) == [
        ("greenhouse", "g1"),
        ("lever", "l1"),
        ("greenhouse", "g2"),
        ("greenhouse", "g3"),
    ]
    query = Query(source="directory", systems=["greenhouse", "lever"], keywords=["engineer"])
    list(directory.scan(query))
    assert directory.due(["greenhouse", "lever"], 10) == []
    now[0] += RESCAN_SECONDS
    assert ("greenhouse", "g2") not in directory.due(["greenhouse", "lever"], 10)
    assert len(directory.due(["greenhouse", "lever"], 10)) == 3


def test_a_board_left_mid_scan_is_scanned_again(tmp_path):
    now = [10_000.0]

    def handler(request):
        if request.url.path.endswith("/jobs"):
            jobs = [
                {"id": n, "title": "Engineer", "absolute_url": f"https://e.example.com/{n}"}
                for n in (1, 2)
            ]
            return httpx.Response(200, json={"jobs": jobs})
        return httpx.Response(200, json={"content": "Body"})

    directory = seeded_directory(tmp_path, client_for(handler), now)
    query = Query(source="directory", systems=["greenhouse"], keywords=["engineer"])
    postings = fetch_postings([query], client_for(handler), directory=directory)
    assert next(postings).url == "https://e.example.com/1"
    postings.close()
    assert ("greenhouse", "g1") in directory.due(["greenhouse"], 10)


def test_scan_reads_titles_first_and_opens_only_matches(tmp_path):
    opened = []

    def handler(request):
        if request.url.path.endswith("/jobs"):
            assert "content" not in request.url.params
            jobs = [
                {"id": 1, "title": "Engineer", "absolute_url": "https://e.example.com/1"},
                {"id": 2, "title": "Recruiter", "absolute_url": "https://e.example.com/2"},
            ]
            return httpx.Response(200, json={"jobs": jobs})
        opened.append(request.url.path)
        return httpx.Response(200, json={"content": "&lt;p&gt;Body&lt;/p&gt;"})

    directory = seeded_directory(tmp_path, client_for(handler), [10_000.0])
    query = Query(
        source="directory", systems=["greenhouse"], keywords=["engineer"], boards_per_run=1
    )
    [posting] = fetch_postings([query], client_for(handler), directory=directory)
    assert opened == ["/v1/boards/g1/jobs/1"]
    assert "Body" in posting.text
