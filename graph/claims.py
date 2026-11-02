"""Deterministic claim extraction: traces a draft back to the verified facts store.

No model is involved, so nothing in a scraped posting can influence the result. The check
fails closed. Only a short, closed set of greetings, sign-offs, and closings is exempt, plus
one opening template whose role and employer must appear in the posting. Every other sentence
is a claim about the candidate, and may only use vocabulary from the facts store. Terms found
only in the posting never support a claim, because borrowing a posting's requirements is the
usual way a draft invents experience.

This is a vocabulary check, not a semantic proof, so the human gate still reads every draft.
"""

import re
import unicodedata
from dataclasses import dataclass

from graph.facts import FactsStore

WORD_RE = re.compile(r"\w[\w+#]*(?:[.\-/]\w[\w+#]*)*")
NUMBER_RE = re.compile(r"(?<![^\W\d_])\d+(?:[.,]\d+)*")
LINK_RE = re.compile(
    r"[a-z][a-z0-9+.-]*://\S+|\bwww\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+"
    r"|(?<![\w@])(?:[\w-]+\.)+[a-z]{2,}(?:/\S*)?",
    re.I,
)

GREETING_RE = re.compile(
    r"^(hi|hello|dear)( there| all| team| hiring team| hiring manager| recruiter"
    r"| recruiting team)?[,:!.]?$",
    re.I,
)
SIGN_OFF_RE = re.compile(
    r"^(best|best regards|kind regards|warm regards|regards|sincerely|thanks|thank you|cheers"
    r"|yours truly)[,.!]?$",
    re.I,
)
CLOSING_RE = re.compile(
    r"^((i look forward|looking forward) to hearing from you"
    r"|(thank you|thanks) for your (time|consideration))[.!]?$",
    re.I,
)
OPENING_RE = re.compile(
    r"^(?:i am|i'm) (?:excited|writing|keen|eager|happy|glad|delighted|interested) to apply for "
    r"(?:the )?(?P<role>[^,;:]{1,80}?)(?: role| position| opening)?"
    r"(?: at (?P<employer>[^,;:]{1,60}?))?[.!]?$",
    re.I,
)
COMPARATOR_RE = re.compile(
    r"\b(over|under|above|below|at least|at most|more than|less than|fewer than|nearly|almost"
    r"|up to)\s+\d",
    re.I,
)
# A figure's fact may hedge it. Dropping the hedge would turn an estimate into a claim.
QUALIFIERS = ("estimated", "approximately", "roughly", "around", "targeting")
DISALLOWED_CATEGORIES = frozenset({"Cc", "Cf", "Cn", "Co", "So"})
ALLOWED_CONTROLS = frozenset("\n\r\t")

STOPWORDS = frozenset(
    "a about above across after all also am an and any are as at be because been being both "
    "but by can could did do does for from had has have having he her here his how i if in "
    "into is it its just me more most my no not of on once only or other our out over own "
    "she so some such than that the their them then there these they this those through to "
    "too under until up us very was we were what when where which while who whom why will "
    "with would you your yours i'm i've i'd i'll we've".split()
)
# Connective words a truthful rephrase needs. None of them can carry a credential on their
# own: skills, tools, employers, and durations must still come from the facts store.
GENERIC_WORDS = frozenset(
    "ability ably along alongside apply applying background bring bringing building "
    "closely combine combined contributing currently directly drawing end ensuring "
    "excited experience focus focused gained hands hands-on help helped helping include "
    "includes including interest interested involved leverage leveraging like make making "
    "many much need needs now particularly previously recent recently relevant role "
    "skills solid specifically strong successfully there think use used using well work "
    "worked working".split()
)


@dataclass(frozen=True)
class Violation:
    sentence: str
    reason: str

    def __str__(self) -> str:
        return f"{self.reason}: {self.sentence!r}"


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


GENERIC_STEMS = frozenset(_stem(word) for word in GENERIC_WORDS | STOPWORDS)


def _words(text: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFKC", text))


def _parts(word: str) -> list[str]:
    """Split a compound such as full-stack, or a figure such as 50-75%, into its stems."""
    pieces = [piece for piece in re.split(r"[\-/]", word.lower().rstrip(".")) if piece]
    return [_stem(piece) for piece in pieces]


def _vocabulary(text: str) -> set[str]:
    vocabulary = {number.replace(",", "") for number in NUMBER_RE.findall(text)}
    for word in _words(text):
        vocabulary.update(_parts(word))
    return vocabulary


def _claim_terms(text: str) -> list[str]:
    terms = []
    for word in _words(text):
        if word.lower() in STOPWORDS or NUMBER_RE.fullmatch(word):
            continue
        terms += [part for part in _parts(word) if not part.isdigit()]
    return [term for term in terms if term not in GENERIC_STEMS]


def _phrase(text: str) -> str:
    """Whole words joined by spaces, for matching one phrase inside another."""
    return " " + " ".join(word.lower() for word in _words(text)) + " "


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in unicodedata.normalize("NFKC", text).splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line:
            parts = re.split(r"(?<=[.!?])\s+|(?<=。)\s*", line)
            sentences += [part.strip() for part in parts if part.strip()]
    return sentences


def _is_exempt(sentence: str, posting_phrase: str) -> bool:
    if GREETING_RE.match(sentence) or SIGN_OFF_RE.match(sentence) or CLOSING_RE.match(sentence):
        return True
    opening = OPENING_RE.match(sentence)
    if not opening:
        return False
    names = [opening["role"], opening["employer"]]
    return all(_phrase(name) in posting_phrase for name in names if name)


def _preposition_links(text: str) -> set[tuple[str, ...]]:
    """Links that name an employer, school, or subject, such as 'at Acme' or 'degree of Law'."""
    words = [word.lower() for word in _words(text)]
    links: set[tuple[str, ...]] = set()
    for index, word in enumerate(words):
        if word not in ("at", "from", "of"):
            continue
        following = [item for item in words[index + 1 :] if item not in STOPWORDS]
        if not following or not _parts(following[0]):
            continue
        target = _parts(following[0])[0]
        if word != "of":
            links.add((word, target))
        elif index and words[index - 1] not in STOPWORDS and _parts(words[index - 1]):
            links.add((_parts(words[index - 1])[-1], word, target))
    return links


def _figure_problems(sentence: str, facts: FactsStore, vocabularies: list[set[str]]) -> list[str]:
    """A figure's sentence must stay inside one fact that holds it, with the fact's hedges."""
    lowered = sentence.lower()
    terms = set(_claim_terms(sentence)) - _vocabulary(facts.candidate_name)
    problems = []
    for number in {found.replace(",", "") for found in NUMBER_RE.findall(sentence)}:
        owners = [
            fact.text.lower()
            for fact, vocabulary in zip(facts.facts, vocabularies, strict=True)
            if number in vocabulary and terms <= vocabulary
        ]
        if not owners:
            problems.append(f"figure {number} is not supported by a single fact")
        elif not any(
            all(
                word in lowered or not re.search(rf"\b{word}\s+\S*{re.escape(number)}", owner)
                for word in QUALIFIERS
            )
            for owner in owners
        ):
            problems.append(f"figure {number} dropped its qualifier")
    fact_text = " ".join(fact.text.lower() for fact in facts.facts)
    for match in COMPARATOR_RE.finditer(sentence):
        if match.group().lower() not in fact_text:
            problems.append(f"comparator not in facts: {match.group()!r}")
    return problems


def verify_draft(draft: str, facts: FactsStore, posting_text: str) -> list[Violation]:
    """Return every sentence of `draft` not traceable to the facts store. Empty means clean."""
    vocabularies = [_vocabulary(fact.text) for fact in facts.facts]
    allowed_terms = set().union(*vocabularies) | _vocabulary(facts.candidate_name)
    allowed_links = set().union(*(_preposition_links(fact.text) for fact in facts.facts))
    fact_text = " ".join(fact.text for fact in facts.facts).lower()
    posting_phrase = _phrase(posting_text)

    violations = []
    normalized = unicodedata.normalize("NFKC", draft)
    hidden = sorted(
        {
            f"U+{ord(ch):04X}"
            for ch in normalized
            if unicodedata.category(ch) in DISALLOWED_CATEGORIES and ch not in ALLOWED_CONTROLS
        }
    )
    if hidden:
        violations.append(Violation(draft[:60], f"disallowed characters: {hidden}"))

    for sentence in split_sentences(draft):
        for link in LINK_RE.findall(sentence):
            if link.rstrip(".,)").lower() not in fact_text:
                violations.append(Violation(sentence, f"link or address not in facts: {link}"))
        if _is_exempt(sentence, posting_phrase):
            continue

        unknown = sorted({term for term in _claim_terms(sentence) if term not in allowed_terms})
        if unknown:
            violations.append(Violation(sentence, f"claim terms not in facts: {unknown}"))
        unlinked = sorted(" ".join(link) for link in _preposition_links(sentence) - allowed_links)
        if unlinked:
            violations.append(Violation(sentence, f"phrases not in any fact: {unlinked}"))
        for problem in _figure_problems(sentence, facts, vocabularies):
            violations.append(Violation(sentence, problem))
    return violations
