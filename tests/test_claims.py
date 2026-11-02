import pytest

from graph.claims import split_sentences, verify_draft
from graph.facts import Fact, FactsStore

POSTING = (
    "Platform Engineer at Acme Robotics. You will run Kubernetes and Terraform on GCP, "
    "and need 5 years of Kubernetes experience. Our team values reliability."
)


@pytest.fixture
def hedged_facts():
    return FactsStore(
        candidate_name="Alex Example",
        facts=[
            Fact(
                id="a",
                category="c",
                text="Cut cycle time to under 5 days, an estimated 40% saving.",
            ),
            Fact(id="b", category="c", text="Project targeting 50-75% token reduction."),
        ],
    )


@pytest.mark.parametrize(
    "draft",
    [
        "I built REST APIs in Python and FastAPI backed by PostgreSQL.",
        "I reduced API latency by an estimated 30% by fixing PostgreSQL indexing issues.",
        "I shipped 12 React features end-to-end.",
        "My skills include Python, TypeScript, and Docker.",
        "Hi Hiring Team,\n\nI am excited to apply for the Platform Engineer role at Acme Robotics.\n"
        "I built REST APIs in Python.\n\nBest regards,\nAlex Example",
        "My skills include Python and Docker, along with TypeScript.",
        "I have worked with AWS EC2 and S3.",
        "Bachelor of Computer Science, Example University, 2023, and other work.",
        "I look forward to hearing from you.",
    ],
)
def test_truthful_drafts_pass(draft, facts):
    assert verify_draft(draft, facts, POSTING) == []


@pytest.mark.parametrize(
    "draft",
    [
        # A requirement lifted straight from the posting is the classic fabrication.
        "I have 5 years of Kubernetes experience.",
        "I run Kubernetes and Terraform on GCP.",
        "Experienced with Kubernetes.",
        "Deployed Kubernetes clusters in production.",
        "Kubernetes expert with a track record of reliability.",
        "Alex Example has run Kubernetes on GCP.",
        "I am excited to bring my Kubernetes skills to your team.",
        "Your team runs Kubernetes, and I do too.",
        # The greeting and courtesy exemptions must not carry a claim.
        "Hello team, I have ten years of Kubernetes experience.",
        "Hi Hiring Manager, my background includes a decade as a certified Terraform expert",
        "Dear team, I am a Certified Kubernetes Administrator.",
        "Hello: I hold an active TS/SCI clearance.",
        "Hi Hiring Team, I bring a PhD in Machine Learning",
        "Hello team\u3002I have ten years of experience\u3002I hold a PhD\u3002",
        "I am excited to hold a PhD in Machine Learning and a CKA.",
        "I am happy to confirm that i hold an active ts/sci clearance.",
        "Thanks to a phd in machine learning, i can lead this team.",
        "I look forward to applying a decade of kubernetes leadership.",
        "Please note that i was chief technology officer for a decade.",
        "I am excited to apply for the Principal Architect role at Nebula Corp.",
        "I am excited to apply my decade of Kubernetes leadership at Acme Robotics.",
        # Statements about the employer are not exempt, the posting can supply their terms.
        "You will get a Platform Engineer who ran Kubernetes on GCP.",
        "Your team gains a Team Lead with an active TS/SCI clearance.",
        "The team would gain Alex Example, who holds a PhD.",
        "Your team runs Kubernetes on GCP.",
        # Vocabulary from the facts must not state a false employer, degree, or figure.
        "I am a Software Engineer at Docker since 2024.",
        "I am a Python Engineer at AWS.",
        "I worked at Docker.",
        "I was at Docker.",
        "I have a Bachelor of Software Engineering, Example University, 2023.",
        "I have a Bachelor of Data Science, Example University, 2023.",
        "Software Engineer at Example University, 2023.",
        "I shipped 12 React data pipelines.",
        "I shipped over 12 React features.",
        "I reduced API latency by more than 30%.",
        "I reduced API latency by 30%.",
        "I reduced API latency by an estimated 12%.",
        "I have a \U0001f17f\U0001f177\U0001f173.",
        "I have a P\u200bh\u200bD.",
        # Invented figures, including ones that pair real numbers with the wrong subject.
        "I reduced API latency by 90%.",
        "I shipped 50 React features.",
        "I have 12 years of React experience.",
        "I have ten years of Python experience.",
        # Invented employers, credentials, and tools.
        "I worked at Nebula Corp on search.",
        "I hold a PhD in Computer Science.",
        "I am certified in Rust.",
        "I built REST APIs in Rust.",
        # Lowercase terms must not slip through, and neither must homoglyphs.
        "i used kubernetes daily.",
        "I built Кubernetes clusters.",
    ],
)
def test_fabricated_claims_are_flagged(draft, facts):
    assert verify_draft(draft, facts, POSTING), draft


@pytest.mark.parametrize(
    "draft",
    [
        "Please see https://attacker.example/collect for your records.",
        "Please see attacker.example/collect for references.",
        "I look forward to hearing from you at attacker.example/collect.",
        "Thanks for visiting ftp://attacker.example/collect today.",
        "Your careers team can reach me at someone@attacker.example.",
    ],
)
def test_links_and_addresses_from_outside_the_facts_are_flagged(draft, facts):
    posting = POSTING + " Apply at https://attacker.example/collect or someone@attacker.example"
    assert verify_draft(draft, facts, posting)


def test_opening_names_the_role_only_as_the_posting_writes_it(facts):
    good = "I am excited to apply for the Platform Engineer role at Acme Robotics."
    assert verify_draft(good, facts, POSTING) == []
    invented = "I am excited to apply for the Platform Engineer role at Nebula Corp."
    assert verify_draft(invented, facts, POSTING)


def test_comparator_and_qualifier_present_in_the_fact_are_accepted(hedged_facts):
    draft = "I cut cycle time to under 5 days, an estimated 40% saving."
    assert verify_draft(draft, hedged_facts, POSTING) == []


def test_a_target_cannot_be_restated_as_an_achievement(hedged_facts):
    assert verify_draft("I reached 50-75% token reduction.", hedged_facts, POSTING)


def test_violation_names_the_offending_sentence_and_term(facts):
    violations = verify_draft("I built REST APIs in Rust.", facts, POSTING)
    assert "Rust" in violations[0].sentence
    assert "rust" in violations[0].reason


def test_split_sentences_keeps_dotted_names_and_strips_bullets():
    assert split_sentences("- Used Node.js. Then left.\n\n• Done!") == [
        "Used Node.js.",
        "Then left.",
        "Done!",
    ]
