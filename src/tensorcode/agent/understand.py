"""Text in: sentences, parses, and what each sentence does (tell, ask, request).

Nothing here decides what to *do*. A message is split into sentences by punctuation
and line breaks — never by guessing where a new command starts — and each sentence is
parsed by the chart parser. What a sentence does comes from its grammatical mood:
an imperative is a request, an interrogative a question, a declarative something told.
How much of the sentence the parse covered (words skipped, words guessed) travels
with it, so the agent can say what it did not follow instead of acting on a fragment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..language import Entity, Frame, Grammar, Question, Request, understand
from ..language.chart import Reading, tokenize
from ..language.conventions import RequestConventions, RequestInterpretation, interpret_request, request_conventions

#: Sentence-final punctuation. Brackets nest and double quotes or backticks toggle, and
#: anything inside them stays in one sentence; an apostrophe ("don't") is not a quote.
ENDERS = ".!?"
BRACKETS = {"(": ")", "[": "]", "{": "}"}
TOGGLES = "\"`“”"


def sentences(text: str) -> list[str]:
    """Split a message into sentences: at . ! ? followed by a space or end, and at line breaks.

    Each line of a list ("the artifact itself" under "design:") is its own sentence;
    the agent still reads them as one message and answers once.
    """
    out: list[str] = []
    buf: list[str] = []
    closing: list[str] = []
    quoted = False

    def flush() -> None:
        s = "".join(buf).strip()
        if s:
            out.append(s)
        buf.clear()

    for i, ch in enumerate(text):
        if ch in TOGGLES:
            quoted = not quoted
        elif not quoted and ch in BRACKETS:
            closing.append(BRACKETS[ch])
        elif not quoted and closing and ch == closing[-1]:
            closing.pop()
        inside = quoted or bool(closing)
        if ch == "\n" and not inside:
            flush()
            continue
        buf.append(ch)
        if ch in ENDERS and not inside and (i + 1 == len(text) or text[i + 1].isspace()):
            flush()
    flush()
    return out


@dataclass(frozen=True)
class Act:
    """What one meaning in a sentence does."""

    kind: str          # "request" | "question" | "tell" | "fragment" | "mention"
    meaning: Any       # Request | Question | Frame | other constituent
    frame: Frame | None
    interpretation: RequestInterpretation | None = None

    def describe(self) -> str:
        m = self.meaning
        return f"{self.kind}: " + (m.describe() if hasattr(m, "describe") else repr(m))


@dataclass(frozen=True)
class Sentence:
    text: str
    tokens: tuple[str, ...]
    reading: Reading | None
    acts: tuple[Act, ...]
    skipped: tuple[str, ...] = field(default=())
    guessed: tuple[tuple[str, str], ...] = field(default=())
    parse_ms: float = 0.0

    @property
    def coverage(self) -> float:
        """Share of word tokens the parse used (punctuation does not count against it)."""
        words = [t for t in self.tokens if any(c.isalnum() for c in t)]
        missed = [t for t in self.skipped if any(c.isalnum() for c in t)]
        return 1.0 if not words else round(1 - len(missed) / len(words), 3)


def indirect_request(q: Question, conventions: RequestConventions | None = None) -> Request | None:
    """Compatibility wrapper for the language-owned, defeasible request convention."""
    interpretation = interpret_request(q, conventions)
    return interpretation.request if interpretation is not None else None


def acts_of(meaning: Any, conventions: RequestConventions | None = None) -> list[Act]:
    """One act per meaning; coordinated requests ("do x and do y") are several."""
    if isinstance(meaning, tuple):
        return [a for m in meaning for a in acts_of(m, conventions)]
    return [act_of(meaning, conventions)]


def named_object(frame: Frame) -> Frame:
    """"make a folder called x" may hang "called x" on the verb or on the noun; either way
    it names the object, so it is folded into the object's description."""
    name, obj = frame.roles.get("name"), frame.roles.get("object")
    if name is None or not isinstance(obj, Entity) or obj.features.get("name") is not None:
        return frame
    roles = {k: v for k, v in frame.roles.items() if k != "name"}
    roles["object"] = Entity(obj.kind, obj.text, {**obj.features, "name": name}, obj.ref, obj.candidates)
    return Frame(frame.predicate, roles, frame.features)


def act_of(meaning: Any, conventions: RequestConventions | None = None) -> Act:
    if isinstance(meaning, Request):
        frame = named_object(meaning.frame)
        return Act("request", Request(frame), frame)
    if isinstance(meaning, Question):
        interpretation = interpret_request(meaning, conventions)
        if interpretation is not None:
            request = interpretation.request
            return Act("request", request, request.frame, interpretation)
        return Act("question", meaning, meaning.frame)
    if isinstance(meaning, Frame):
        mood = meaning.mood
        if mood == "imperative":
            return Act("request", Request(meaning), meaning)
        return Act("tell", meaning, meaning)
    return Act("fragment", meaning, None)


QUOTES = {'"': '"', "“": "”", "'": "'", "‘": "’", "`": "`"}


def quoted(s: str) -> str | None:
    """The inside of a sentence that is wholly a quotation, else None."""
    t = s.strip()
    if len(t) > 2 and t[0] in QUOTES and t.rstrip(".!?")[-1:] == QUOTES[t[0]]:
        return t[1:].rstrip(".!?").rstrip(QUOTES[t[0]]).strip()
    return None


def parse_one(grammar: Grammar, s: str, *, mention: bool = False, conventions: RequestConventions | None = None) -> Sentence:
    u = understand(grammar, s)
    r = u.readings[0] if u.readings else None
    acts = tuple(a for m in r.meanings for a in acts_of(m, conventions)) if r else ()
    if mention:
        # quoted language is mentioned, not used: an example, a report, a spec — never a
        # request addressed to the agent
        acts = tuple(Act("mention", a.meaning, a.frame, a.interpretation) for a in acts)
    return Sentence(s, tuple(u.tokens), r, acts, tuple(w for _, w in r.skipped) if r else tuple(tokenize(s)),
                    r.guessed if r else (), round(u.ms, 1))


def read(grammar: Grammar, text: str, *, conventions: RequestConventions | None = None) -> list[Sentence]:
    """Every sentence of ``text``, parsed, with its acts in order.

    Two pieces of text structure are read, both general:

    * a sentence that is wholly a quotation is parsed as language, and its acts are
      *mentions* — the use/mention distinction;
    * a sentence ending in a colon, followed by lines that are noun phrases, is one
      request whose object is those lines ("design: the artifact itself / its bom").
    """
    conventions = request_conventions(conventions)
    parts = sentences(text)
    out: list[Sentence] = []
    i = 0
    while i < len(parts):
        s = parts[i]
        inner = quoted(s)
        if inner:
            out.append(parse_one(grammar, inner, mention=True, conventions=conventions))
            i += 1
            continue
        if s.endswith(":") and i + 1 < len(parts):
            head = parse_one(grammar, s[:-1].strip(), conventions=conventions)
            items, j = [], i + 1
            while j < len(parts) and not parts[j].endswith(":"):
                np = understand(grammar, parts[j], starts=("NP",))
                r = np.readings[0] if np.readings else None
                if r is None or len(r.meanings) != 1 or r.skipped and len([w for _, w in r.skipped if w.isalnum()]) > len(np.tokens) // 2:
                    break
                items.append((parts[j], r, tuple(np.tokens)))
                j += 1
            request = next((a for a in head.acts if a.kind == "request" and a.frame is not None and "object" not in a.frame.roles), None)
            if items and request is not None:
                objects = tuple(r.meanings[0] for _, r, _ in items)
                frame = request.frame.filled(object=objects if len(objects) > 1 else objects[0])
                skipped = tuple(w for _, r, _ in items for _, w in r.skipped)
                guessed = tuple(g for _, r, _ in items for g in r.guessed)
                text_all = s + " " + "; ".join(t for t, _, _ in items)
                tokens = head.tokens + tuple(tok for _, _, toks in items for tok in toks)
                out.append(Sentence(text_all, tokens, head.reading, (act_of(Request(frame)),),
                                    head.skipped + skipped, head.guessed + guessed, head.parse_ms))
                i = j
                continue
        out.append(parse_one(grammar, s, conventions=conventions))
        i += 1
    return out


# ------------------------------------------------------------------ the learned reader


class LearnedReader:
    """Reads with the treebank-trained tagger and parser instead of the hand-written grammar.

    Same output as :func:`read`: sentences with acts. What changes is where the knowledge
    comes from — a treebank and its counts, rather than productions and weights written
    here. ``skipped`` is empty by construction: a dependency parse attaches every word, so
    "coverage" is no longer a measure of how much was understood, and a caller that wants
    to know whether the parse is any good has to look at the treebank scores instead.
    """

    def __init__(self, model_path=None, *, conventions: RequestConventions | None = None) -> None:
        from pathlib import Path

        from ..language.deps_semantics import Reader
        from ..language.learned_parser import load_model
        from ..language.treebank import lemma_table, lemmatize, load as load_treebank

        path = model_path or Path.home() / ".cache" / "tensorcode" / "models" / "ud_ewt_parser.pickle"
        got = load_model(path)
        if got is None:
            raise FileNotFoundError(f"no trained parser at {path}: run eval/parsing/train_ud.py")
        self.tagger, self.parser = got
        self.table = lemma_table(load_treebank("train"))
        self.lemmatize = lemmatize
        self.reader = Reader()
        self.conventions = request_conventions(conventions)

    def read(self, text: str) -> list[Sentence]:
        import time

        out = []
        for raw in sentences(text):
            inner = quoted(raw)
            t0 = time.perf_counter()
            words = list(tokenize(inner or raw))
            if not words:
                continue
            tags = self.tagger.tag(words)
            lemmas = [self.lemmatize(w, t, self.table) for w, t in zip(words, tags)]
            heads, labels = self.parser.parse(words, tags)
            meanings = self.reader.read(words, tags, lemmas, heads, labels)
            acts = tuple(a for m in meanings for a in acts_of(m, self.conventions))
            if inner:
                acts = tuple(Act("mention", a.meaning, a.frame, a.interpretation) for a in acts)
            out.append(Sentence(raw, tuple(words), None, acts, (), (), round((time.perf_counter() - t0) * 1000, 1)))
        return out
