from unittest.mock import patch

import httpx
import pytest

from evals.dataset import load_cases
from evals.run_eval import evaluate, reasoning_matches, render_table
from graph.llm import Budget
from graph.nodes.discover import (
    MAX_POSTING_CHARS,
    MAX_RESPONSE_BYTES,
    PoliteClient,
    Posting,
    Query,
    fetch_postings,
    html_to_text,
    is_safe_posting_url,
    make_discover,
    reject_internal_address,
)
from graph.nodes.fit_score import FitAssessment
from graph.nodes.tailor import TailoredDraft
from graph.state import initial_state
from tests.conftest import ROOT, FakeModel


def client_for(handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return PoliteClient(http, delay_seconds=0, sleep=kwargs.get("sleep", lambda _: None))


def test_greenhouse_filters_by_title_and_strips_markup():
    def handler(request):
        assert request.url.params["content"] == "true"
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Backend Engineer",
                        "absolute_url": "https://example.com/1",
                        "location": {"name": "Remote"},
                        "content": "&lt;p&gt;Build APIs&lt;/p&gt;&lt;script&gt;x()&lt;/script&gt;",
                    },
                    {"title": "Senior Backend Engineer", "absolute_url": "https://example.com/2"},
                    {"title": "Recruiter", "absolute_url": "https://example.com/3"},
                ]
            },
        )

    query = Query(
        source="greenhouse", board="b", keywords=["engineer"], exclude_keywords=["senior"]
    )
    postings = list(fetch_postings([query], client_for(handler)))
    assert [p.url for p in postings] == ["https://example.com/1"]
    assert "Build APIs" in postings[0].text
    assert "x()" not in postings[0].text


def test_location_filter_drops_postings_before_any_model_call():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Developer",
                        "absolute_url": "https://example.com/near",
                        "location": {"name": "Remote - Region A"},
                    },
                    {
                        "title": "Developer",
                        "absolute_url": "https://example.com/far",
                        "location": {"name": "City B"},
                    },
                ]
            },
        )

    query = Query(source="greenhouse", board="b", locations=["region a"])
    postings = list(fetch_postings([query], client_for(handler)))
    assert [p.url for p in postings] == ["https://example.com/near"]


def test_lever_postings_are_parsed():
    def handler(request):
        return httpx.Response(
            200,
            json=[
                {
                    "text": "Developer",
                    "hostedUrl": "https://example.com/l",
                    "categories": {"location": "Remote"},
                    "descriptionPlain": "Write code",
                }
            ],
        )

    postings = list(fetch_postings([Query(source="lever", board="b")], client_for(handler)))
    assert postings[0].url == "https://example.com/l"
    assert "Write code" in postings[0].text


def test_a_failing_source_does_not_stop_the_others():
    def handler(request):
        if "bad" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, text="<p>Posting body</p>")

    queries = [
        Query(source="url", url="https://example.com/bad"),
        Query(source="url", url="https://example.com/good"),
    ]
    assert [p.url for p in fetch_postings(queries, client_for(handler))] == [
        "https://example.com/good"
    ]


def test_rate_limits_back_off_and_honour_retry_after():
    waits = []
    replies = [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, text="ok")]

    client = client_for(lambda request: replies.pop(0), sleep=waits.append)
    assert client.get("https://example.com").text == "ok"
    assert 7.0 in waits


def test_persistent_rate_limit_is_bounded_and_raises():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    with pytest.raises(httpx.HTTPStatusError):
        client_for(handler).get("https://example.com")
    assert len(calls) == 4


def test_minimum_delay_is_applied_between_requests():
    waits = []
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    client = PoliteClient(http, delay_seconds=5, sleep=waits.append)
    client.get("https://example.com")
    client.get("https://example.com")
    assert len(waits) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "greenhouse"},
        {"source": "url"},
        {"source": "greenhouse", "board": "../../admin"},
        {"source": "url", "url": "file:///etc/passwd"},
    ],
)
def test_invalid_queries_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Query(**kwargs)


def test_discover_skips_known_urls_without_touching_the_model(store):
    store.claim("https://example.com/seen", "old", "r0")
    postings = iter(
        [
            Posting("https://example.com/seen#apply", "old"),
            Posting("https://example.com/new", "new"),
        ]
    )
    result = make_discover(postings, store)(initial_state("r1"))
    assert result["posting_url"] == "https://example.com/new"
    assert store.has_url("https://example.com/new")


def test_discover_reports_exhaustion_with_an_empty_url(store):
    assert make_discover(iter(()), store)(initial_state("r1"))["posting_url"] == ""


def test_html_to_text_removes_scripts_and_styles():
    assert html_to_text("<style>a{}</style><b>Hi</b><script>x</script>") == "Hi"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/j\n\nI hold a PhD.",
        "https://example.com/\x1b[8m",
        "javascript:alert(1)",
        "https://",
        "https://example.com/" + "a" * 3000,
        None,
    ],
)
def test_unsafe_posting_urls_are_rejected(url):
    assert not is_safe_posting_url(url)


def test_board_entries_with_unsafe_urls_are_skipped():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {"title": "Bad", "absolute_url": "https://example.com/j\n\nI hold a PhD."},
                    {"title": "Good", "absolute_url": "https://example.com/good"},
                ]
            },
        )

    postings = list(fetch_postings([Query(source="greenhouse", board="b")], client_for(handler)))
    assert [p.url for p in postings] == ["https://example.com/good"]


def test_malformed_board_entry_does_not_stop_other_sources():
    def handler(request):
        if "bad" in str(request.url):
            return httpx.Response(200, json={"jobs": [{"title": "No URL"}]})
        return httpx.Response(200, text="<p>Body</p>")

    queries = [
        Query(source="greenhouse", board="bad"),
        Query(source="url", url="https://example.com/good"),
    ]
    assert [p.url for p in fetch_postings(queries, client_for(handler))] == [
        "https://example.com/good"
    ]


def _resolve(host, port):
    address = "93.184.216.34" if host == "example.com" else host
    return [(0, 0, 0, "", (address, port))]


def test_redirect_to_an_internal_address_is_refused():
    def handler(request):
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data"})

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        event_hooks={"request": [reject_internal_address]},
    )
    with patch("graph.nodes.discover.socket.getaddrinfo", _resolve):
        with pytest.raises(ValueError, match="non-public"):
            PoliteClient(http, delay_seconds=0).get("https://example.com/job")


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1"])
def test_internal_addresses_are_refused(address):
    request = httpx.Request("GET", "https://internal.example/")
    with patch("graph.nodes.discover.socket.getaddrinfo") as resolve:
        resolve.return_value = [(0, 0, 0, "", (address, 443))]
        with pytest.raises(ValueError, match="non-public"):
            reject_internal_address(request)


def test_oversized_response_is_refused():
    def handler(request):
        return httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1))

    with pytest.raises(ValueError, match="too large"):
        client_for(handler).get("https://example.com")


def test_long_title_cannot_exceed_the_posting_limit():
    def handler(request):
        return httpx.Response(
            200,
            json={"jobs": [{"title": "T" * 100_000, "absolute_url": "https://example.com/1"}]},
        )

    [posting] = fetch_postings([Query(source="greenhouse", board="b")], client_for(handler))
    assert len(posting.text) == MAX_POSTING_CHARS


# evals


def test_sample_tracker_loads_with_both_labels():
    cases = load_cases(ROOT / "evals" / "sample_tracker.csv")
    assert {case.qualified for case in cases} == {True, False}


def test_sampling_keeps_both_labels():
    cases = load_cases(ROOT / "evals" / "sample_tracker.csv", limit=4)
    assert len(cases) == 4
    assert {case.qualified for case in cases} == {True, False}


def test_unknown_decision_labels_fail_loudly(tmp_path):
    path = tmp_path / "tracker.csv"
    path.write_text("url,posting_text,decision,reason\nu,t,maybe,r\n")
    with pytest.raises(ValueError, match="maybe"):
        load_cases(path)


def test_reasoning_match_compares_disqualifier_categories():
    assert reasoning_matches("Experience gap", "Requires far more years than the candidate has")
    assert reasoning_matches("Experience gap", "Wrong city and no relocation") is False
    assert reasoning_matches("Other", "Anything") is None


def test_evaluate_reports_agreement_and_fact_check(facts):
    cases = load_cases(ROOT / "evals" / "sample_tracker.csv", limit=4)
    good = TailoredDraft(
        message="I built REST APIs in Python and FastAPI backed by PostgreSQL.",
        cited_fact_ids=["exp-api"],
    )
    model = FakeModel(
        {
            "FitAssessment": [
                FitAssessment(
                    fit_score=0.9 if case.qualified else 0.1, fit_reasoning="years of experience"
                )
                for case in cases
            ],
            "TailoredDraft": [good] * len(cases),
        }
    )
    summary = evaluate(model, cases, facts, "t", 0.7, Budget(10**6, 100.0))
    assert summary["agreement_rate"] == 1.0
    assert summary["drafts_verified"] == summary["drafts_checked"] > 0
    summary.update(model="fake/model", elapsed_seconds=1.0)
    assert "100%" in render_table(summary, "sample")
