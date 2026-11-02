"""Deterministic claim extraction: traces a draft back to the verified facts store.

No model is involved, so nothing in a scraped posting can influence the result. The check
fails closed: a sentence counts as a claim about the candidate unless it is recognisably a
greeting, a sign-off, a courtesy, or a statement about the employer. A claim may only use
vocabulary from the facts store. Terms found only in the posting do not support a claim,
because borrowing the posting's requirements is the usual way a draft invents experience.
"""

import re
import unicodedata
from dataclasses import dataclass

from graph.facts import FactsStore

WORD_RE = re.compile(r"\w[\w+#]*(?:[.\-/]\w[\w+#]*)*")
NUMBER_RE = re.compile(r"(?<![^\W\d_])\d+(?:[.,]\d+)*")
URL_OR_EMAIL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself|we|we've|our)\b", re.I)
EMPLOYER_RE = re.compile(
    r"\b(your|you|you're|this (role|position|opening)|the (role|position|team|company))\b", re.I
)
GREETING_RE = re.compile(r"^(hi|hello|dear|greetings)\b[^.!?\d]*[,:!.]?$", re.I)
SIGN_OFF_RE = re.compile(
    r"^(best|best regards|kind regards|warm regards|regards|sincerely|thanks|thank you|cheers"
    r"|yours truly)[,.!]?$",
    re.I,
)
COURTESY_RE = re.compile(
    r"^((i am|i'm) (writing|reaching out|excited|interested|keen|eager|happy|glad|delighted)\b"
    r"|i would (love|welcome|appreciate|be (glad|happy))\b"
    r"|i'd (love|welcome|appreciate|be (glad|happy))\b"
    r"|i look forward\b|looking forward\b|thank you\b|thanks\b|please\b)",
    re.I,
)
# Words that turn a courtesy or employer sentence into an assertion about the candidate.
CREDENTIAL_RE = re.compile(
    r"\b(my|experience\w*|expert\w*|proficien\w*|skill\w*|background|years?|built|led|shipped"
    r"|managed|designed|developed|deployed|certifi\w*|degree|track record|candidate|applicant"
    r"|familiar\w*|knowledge|worked)\b",
    re.I,
)

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


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in unicodedata.normalize("NFKC", text).splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line:
            sentences += [part.strip() for part in SENTENCE_SPLIT_RE.split(line) if part.strip()]
    return sentences


def _is_context(sentence: str) -> bool:
    if CREDENTIAL_RE.search(sentence):
        return False
    if COURTESY_RE.search(sentence):
        return True
    return bool(EMPLOYER_RE.search(sentence)) and not FIRST_PERSON_RE.search(sentence)


def _unknown_context_terms(sentence: str, allowed: set[str]) -> list[str]:
    """Numbers and proper or technical terms in a context sentence must exist somewhere."""
    terms = [number.replace(",", "") for number in NUMBER_RE.findall(sentence)]
    for index, word in enumerate(_words(sentence)):
        looks_specific = any(ch.isupper() for ch in word[1:]) or (word[0].isupper() and index > 0)
        if looks_specific and word.lower() not in STOPWORDS:
            terms += _parts(word)
    return [term for term in terms if term not in allowed]


def _unsupported_numbers(sentence: str, fact_vocabularies: list[set[str]]) -> list[str]:
    """A figure must sit in one fact alongside the word it quantifies in the sentence."""
    unsupported = []
    for match in NUMBER_RE.finditer(sentence):
        number = match.group().replace(",", "")
        clause_end = re.search(r"[,;:]", sentence[match.end() :])
        rest = sentence[match.end() :]
        after = _claim_terms(rest[: clause_end.start()] if clause_end else rest)
        before = _claim_terms(sentence[: match.start()])
        anchors = after[:1] or before[-1:]
        supported = any(
            number in vocabulary and (not anchors or anchors[0] in vocabulary)
            for vocabulary in fact_vocabularies
        )
        if not supported:
            unsupported.append(number)
    return unsupported


def _claim_terms(text: str) -> list[str]:
    terms = []
    for word in _words(text):
        if word.lower() in STOPWORDS or NUMBER_RE.fullmatch(word):
            continue
        terms += [part for part in _parts(word) if not part.isdigit()]
    generic = {_stem(word) for word in GENERIC_WORDS} | {_stem(word) for word in STOPWORDS}
    return [term for term in terms if term not in generic]


def verify_draft(draft: str, facts: FactsStore, posting_text: str) -> list[Violation]:
    """Return every sentence of `draft` not traceable to the facts store. Empty means clean."""
    fact_vocabularies = [_vocabulary(fact.text) for fact in facts.facts]
    fact_vocabulary = set().union(*fact_vocabularies) | _vocabulary(facts.candidate_name)
    context_vocabulary = fact_vocabulary | _vocabulary(posting_text)
    fact_text = " ".join(fact.text for fact in facts.facts)

    violations = []
    for sentence in split_sentences(draft):
        for link in URL_OR_EMAIL_RE.findall(sentence):
            if link.rstrip(".,)") not in fact_text:
                violations.append(Violation(sentence, f"link or address not in facts: {link}"))

        if GREETING_RE.match(sentence) or SIGN_OFF_RE.match(sentence):
            continue
        if _is_context(sentence):
            unknown = _unknown_context_terms(sentence, context_vocabulary)
            if unknown:
                violations.append(
                    Violation(
                        sentence, f"terms in neither posting nor facts: {sorted(set(unknown))}"
                    )
                )
            continue

        unknown = [term for term in _claim_terms(sentence) if term not in fact_vocabulary]
        if unknown:
            violations.append(
                Violation(sentence, f"claim terms not in facts: {sorted(set(unknown))}")
            )
        numbers = _unsupported_numbers(sentence, fact_vocabularies)
        if numbers:
            violations.append(Violation(sentence, f"figures not supported by a fact: {numbers}"))
    return violations
