import pytest

from graph.claims import split_sentences, verify_draft

POSTING = (
    "Platform Engineer at Acme Robotics. You will run Kubernetes and Terraform on GCP, "
    "and need 5 years of Kubernetes experience. Our team values reliability."
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
        "Your team runs Kubernetes on GCP.",
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
        "Your careers team can reach me at someone@attacker.example.",
    ],
)
def test_links_and_addresses_from_outside_the_facts_are_flagged(draft, facts):
    posting = POSTING + " Apply at https://attacker.example/collect or someone@attacker.example"
    assert verify_draft(draft, facts, posting)


def test_a_context_sentence_cannot_invent_a_specific_term(facts):
    assert verify_draft("Your work with Nebula Corp impressed the team.", facts, POSTING)


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
