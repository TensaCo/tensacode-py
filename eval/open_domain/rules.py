"""The no-model arm: tensorcode's own machinery on open-domain questions.

This is deliberately a real attempt, not a strawman:

* evidence selection is `tc.rank` backed by the repo's BM25 implementation, the same
  op the HotpotQA context work used;
* each retrieved sentence becomes a claim in a `tc.Store` with provenance, so a cited
  answer can be traced to the sentence it came from;
* the answer type is read off the question (who / when / where / how many / which),
  and only spans of that type are considered;
* when the evidence is weak or no span of the right type exists, the arm returns
  `Unknown` rather than guessing. That is the behaviour this project claims as a
  virtue, and SQuAD 2.0 is the benchmark that can price it.

What it is not: it has no world knowledge, no arithmetic planner beyond one step, and
no way to answer a question whose answer is not literally in the passage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tensorcode as tc
from tensorcode.backends.builtin import BM25Ranker

from .data import Item

RUNTIME = tc.Runtime([BM25Ranker()], policy=tc.Policy(localities=frozenset({"in_process"}), allow_egress=False, cache=False))

STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be", "been",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "many", "much", "did", "does", "do",
    "this", "that", "these", "those", "it", "its", "his", "her", "their", "there", "as", "by", "with", "from",
    "first", "name", "also", "known", "between", "during", "after", "before", "than", "then",
}
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
NUM = re.compile(r"\b\d[\d,.]*\b")
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
DATE = re.compile(rf"\b(?:{MONTHS})\b\s+\d{{1,2}},?\s*(?:1[0-9]{{3}}|20[0-9]{{2}})?|\b(?:{MONTHS})\b\s+(?:1[0-9]{{3}}|20[0-9]{{2}})", re.I)
PROPER = re.compile(r"\b(?:[A-Z][\w.'-]*)(?:\s+(?:of|the|de|van|von|and)\s+[A-Z][\w.'-]*|\s+[A-Z][\w.'-]*)*\b")
NP_AFTER = re.compile(r"\b(?:is|are|was|were|called|named|known as)\b\s+(?:the\s+|a\s+|an\s+)?([a-z][\w'-]*(?:\s+[a-z][\w'-]*){0,3})")


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]


def answer_type(question: str) -> str:
    q = question.lower().strip()
    if re.search(r"\bhow (?:many|much)\b", q):
        return "number"
    if q.startswith("when") or re.search(r"\b(?:what year|which year|in what year)\b", q):
        return "date"
    if q.startswith("who") or re.search(r"\bwhose\b", q):
        return "person"
    if q.startswith("where") or re.search(r"\b(?:what (?:city|country|state|place)|which (?:city|country|state|place))\b", q):
        return "place"
    if re.search(r"\b(?:is|was|are|were|does|did|do|has|have|can|could|will|would)\b", q.split()[0] if q.split() else ""):
        return "boolean"
    return "entity"


@dataclass
class Answer:
    text: str | tc.Unknown
    evidence: list[tuple[str, str]]     # (title, sentence) actually used
    score: float
    reason: str = ""


def _candidates(sentence: str, kind: str, question: str) -> list[str]:
    asked = set(_words(question))
    out: list[str] = []
    if kind == "number":
        out = [m.group(0).rstrip(".,") for m in NUM.finditer(sentence)]
    elif kind == "date":
        out = [m.group(0).strip(" ,") for m in DATE.finditer(sentence)] + [m.group(0) for m in YEAR.finditer(sentence)]
    elif kind in ("person", "place", "entity"):
        out = [m.group(0).strip() for m in PROPER.finditer(sentence)]
        if kind == "entity":
            out += [m.group(1).strip() for m in NP_AFTER.finditer(sentence)]
    # a span that just repeats the question is not an answer
    out = [c for c in out if c and not set(_words(c)) <= asked or kind in ("number", "date")]
    seen, uniq = set(), []
    for c in out:
        k = c.lower()
        if k not in seen and len(c) > 1:
            seen.add(k)
            uniq.append(c)
    return uniq


def answer_extractive(item: Item, *, min_score: float, top_k: int = 3, store: tc.Store | None = None) -> Answer:
    """Rank the passage's sentences, then take the best span of the asked-for type."""
    sentences = [s for _, s in item.passages]
    if not sentences:
        return Answer(tc.Unknown("no_passage", "nothing to read"), [], 0.0, "no passage")
    with tc.use(RUNTIME):
        ranked = tc.rank(item.question, sentences, limit=top_k)
    if isinstance(ranked, tc.Unknown):
        return Answer(ranked, [], 0.0, "rank abstained")
    kind = answer_type(item.question)
    used: list[tuple[str, str]] = []
    best: tuple[float, str, str] | None = None
    for sent, score in ranked:
        title = next((t for t, s in item.passages if s == sent), "")
        used.append((title, sent))
        if store is not None:
            ref = tc.Ref(f"sentence:{abs(hash(sent)) % 10**10}")
            store.tell(
                tc.Claim(ref, "reads", sent),
                tc.Evidence(tc.Ref(f"passage:{title}"), __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                            method="bm25-rank@1", confidence=tc.Score(float(score.value), "bm25")),
            )
        for cand in _candidates(sent, kind, item.question):
            if best is None or score.value > best[0]:
                best = (float(score.value), cand, sent)
        if best is not None:
            break  # the top sentence that yields a span of the right type wins
    top = float(ranked[0][1].value)
    if top < min_score:
        return Answer(tc.Unknown("weak_evidence", f"best bm25 {top:.2f} < {min_score:.2f}"), used, top, "weak evidence")
    if best is None:
        return Answer(tc.Unknown("no_span_of_type", f"no {kind} span in the top {top_k} sentences"), used, top, "no span of type")
    return Answer(best[1], used, top)


# ----------------------------------------------------------------- arithmetic

OPS = [
    (re.compile(r"\b(?:how many|how much)\b.*\b(?:in total|altogether|combined)\b", re.I), "sum"),
    (re.compile(r"\b(?:how many .* (?:left|remain|remaining)|how much .* (?:left|remain))\b", re.I), "difference"),
]


def answer_arithmetic(item: Item) -> Answer:
    """One-step arithmetic only: a sum or a difference over the numbers stated.

    Anything needing two or more steps abstains. GSM8K is overwhelmingly multi-step,
    so this is expected to abstain on almost everything; the point is to measure where
    the boundary is rather than to guess.
    """
    nums = [float(m.group(0).replace(",", "")) for m in NUM.finditer(item.question)]
    if len(nums) < 2:
        return Answer(tc.Unknown("not_enough_quantities", f"{len(nums)} numbers found"), [], 0.0)
    kind = next((k for pat, k in OPS if pat.search(item.question)), None)
    if kind is None:
        return Answer(tc.Unknown("no_single_step_pattern", "no total/remainder pattern"), [], 0.0)
    sentences = re.split(r"(?<=[.!?])\s+", item.question)
    if len(sentences) > 3:
        return Answer(tc.Unknown("multi_step", f"{len(sentences)} sentences: needs a plan"), [], 0.0)
    value = sum(nums) if kind == "sum" else nums[0] - sum(nums[1:])
    text = str(int(value)) if float(value).is_integer() else str(value)
    return Answer(text, [], 1.0, kind)


# ------------------------------------------------------- multiple choice

def answer_multiple_choice(item: Item, *, guess_by_overlap: bool) -> Answer:
    """No knowledge source exists, so the honest default is to abstain.

    `guess_by_overlap` enables the classic weak heuristic (pick the option sharing the
    most words with the question) purely so the report can show what that buys over
    chance. It is reported separately and never as the arm's headline.
    """
    if not guess_by_overlap:
        return Answer(tc.Unknown("no_knowledge_source", "multiple choice needs world knowledge; nothing to read"), [], 0.0)
    asked = set(_words(item.question))
    scored = [(len(asked & set(_words(text))), label) for label, text in item.options.items()]
    scored.sort(key=lambda x: (-x[0], x[1]))
    if not scored or scored[0][0] == 0:
        return Answer(tc.Unknown("no_lexical_overlap", "no option shares a content word"), [], 0.0)
    return Answer(scored[0][1], [], float(scored[0][0]), "lexical overlap")
