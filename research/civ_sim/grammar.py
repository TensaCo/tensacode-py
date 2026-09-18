"""A symbolic grammar for the people's speech: one grammar, parsed and generated, no model.

    understanding   tokenize -> morphology -> Earley chart parse over a feature grammar with
                    unification -> semantic frames -> claim graphs (with provenance)
    speaking        the same rules run top-down from a frame to a sentence, in the speaker's dialect

It handles rather more than templates: embedded clauses ("Anem said the north field failed"),
negation, tense, modality ("might", "must"), quantifiers ("everyone", "some of the grain"),
questions and imperatives, coordination, comparatives, and pronouns resolved against the
conversation. Ambiguity is real: the chart keeps every parse and the ranker picks one, which is
where mishearing comes from, along with unknown dialect words.

Nothing here is statistical and nothing calls a model: it is a lexicon, an affix table, five
dozen rules and a chart.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Iterable

# ----------------------------------------------------------------- features


AGREEMENT = ("num", "person", "tense", "gender", "q", "modality", "wh", "rel", "amount", "place", "time", "cmp", "aspect", "def")
# what a completed phrase passes up to its parent: tense-like features only, so an object's
# person cannot clash with the subject's (this grammar reads meaning, it does not police agreement)
PROPAGATE = ("tense", "aspect", "modality", "wh", "rel", "amount", "place", "time", "cmp", "q")


def unify(a: dict, b: dict) -> dict | None:
    """Merge two feature dicts, or fail if they disagree on a shared feature."""
    out = dict(a)
    for k, v in b.items():
        if k in out and out[k] != v and out[k] is not None and v is not None:
            return None
        out[k] = v
    return out


@dataclass(frozen=True)
class Frame:
    """A semantic frame: what was said, in a form claims can be read off."""

    pred: str
    args: tuple  # ((role, value), ...) where value is a str or a nested Frame
    tense: str = "present"
    mood: str = "declare"  # declare | ask | command
    modality: str | None = None  # might | must | should
    negated: bool = False
    quantifier: str | None = None  # all | some | none | most

    def get(self, role: str):
        return next((v for r, v in self.args if r == role), None)

    def describe(self) -> str:
        bits = [f"{self.pred}(" + ", ".join(f"{r}={v.pred + '(...)' if isinstance(v, Frame) else v}" for r, v in self.args) + ")"]
        if self.negated:
            bits.append("not")
        if self.modality:
            bits.append(self.modality)
        if self.quantifier:
            bits.append(f"quant={self.quantifier}")
        if self.tense != "present":
            bits.append(self.tense)
        if self.mood != "declare":
            bits.append(self.mood)
        return " ".join(bits)


# ------------------------------------------------------------------ lexicon

# (surface, category, features, semantics)
BASE_LEXICON: tuple = (
    # determiners and quantifiers
    ("the", "DET", {"def": True}, None), ("a", "DET", {"def": False}, None), ("an", "DET", {"def": False}, None),
    ("my", "DET", {"def": True, "poss": "speaker"}, None), ("our", "DET", {"def": True, "poss": "group"}, None),
    ("this", "DET", {"def": True, "near": True}, None), ("that", "DET", {"def": True, "near": False}, None),
    ("every", "QUANT", {"q": "all"}, None), ("all", "QUANT", {"q": "all"}, None), ("some", "QUANT", {"q": "some"}, None),
    ("no", "QUANT", {"q": "none"}, None), ("most", "QUANT", {"q": "most"}, None), ("few", "QUANT", {"q": "few"}, None),
    ("everyone", "NP", {"q": "all", "kind": "people"}, "people"), ("nobody", "NP", {"q": "none", "kind": "people"}, "people"),
    ("someone", "NP", {"q": "some", "kind": "people"}, "people"),
    # nouns (kind tells the claim builder what sort of thing it is)
    ("granary", "N", {"kind": "store"}, "store"), ("store", "N", {"kind": "store"}, "store"), ("loft", "N", {"kind": "store"}, "store"),
    ("hoard", "N", {"kind": "store"}, "store"), ("field", "N", {"kind": "place"}, "field"), ("fields", "N", {"kind": "place", "num": "pl"}, "field"),
    ("food", "N", {"kind": "good", "mass": True}, "food"), ("grain", "N", {"kind": "good", "mass": True}, "food"),
    ("bread", "N", {"kind": "good", "mass": True}, "food"), ("forage", "N", {"kind": "good", "mass": True}, "food"),
    ("wood", "N", {"kind": "good", "mass": True}, "wood"), ("timber", "N", {"kind": "good", "mass": True}, "wood"),
    ("stone", "N", {"kind": "good", "mass": True}, "stone"), ("tool", "N", {"kind": "good"}, "tool"), ("tools", "N", {"kind": "good", "num": "pl"}, "tool"),
    ("raiders", "N", {"kind": "people", "num": "pl"}, "raiders"), ("child", "N", {"kind": "person"}, "child"),
    ("children", "N", {"kind": "person", "num": "pl"}, "child"), ("winter", "N", {"kind": "time"}, "winter"),
    ("rain", "N", {"kind": "weather", "mass": True}, "rain"), ("snow", "N", {"kind": "weather", "mass": True}, "snow"),
    ("frost", "N", {"kind": "weather", "mass": True}, "frost"), ("sky", "N", {"kind": "sky"}, "sky"),
    ("moon", "N", {"kind": "sky"}, "moon"), ("eclipse", "N", {"kind": "sky"}, "eclipse"),
    ("conjunction", "N", {"kind": "sky"}, "conjunction"), ("solar", "N", {"kind": "sky"}, "solar"), ("lunar", "N", {"kind": "sky"}, "lunar"),
    ("omen", "N", {"kind": "sky"}, "omen"), ("feast", "N", {"kind": "rite"}, "feast"), ("price", "N", {"kind": "econ"}, "price"),
    ("debt", "N", {"kind": "econ"}, "debt"), ("market", "N", {"kind": "place"}, "market"),
    # pronouns
    ("i", "PRON", {"person": 1, "num": "sg"}, "speaker"), ("me", "PRON", {"person": 1, "num": "sg"}, "speaker"),
    ("we", "PRON", {"person": 1, "num": "pl"}, "group"), ("you", "PRON", {"person": 2}, "listener"),
    ("he", "PRON", {"person": 3, "gender": "m"}, "anaphor"), ("she", "PRON", {"person": 3, "gender": "f"}, "anaphor"),
    ("they", "PRON", {"person": 3, "num": "pl"}, "anaphor"), ("it", "PRON", {"person": 3, "gender": "n"}, "anaphor"),
    ("him", "PRON", {"person": 3, "gender": "m"}, "anaphor"), ("her", "PRON", {"person": 3, "gender": "f"}, "anaphor"),
    # verbs: (lemma, frame predicate, valency)
    ("hold", "V", {"valency": 2, "pred": "has_amount"}, "has_amount"), ("holds", "V", {"valency": 2, "pred": "has_amount", "tense": "present"}, "has_amount"),
    ("have", "V", {"valency": 2, "pred": "has_amount"}, "has_amount"), ("has", "V", {"valency": 2, "pred": "has_amount"}, "has_amount"),
    ("keep", "V", {"valency": 2, "pred": "keeps_back"}, "keeps_back"), ("keeps", "V", {"valency": 2, "pred": "keeps_back"}, "keeps_back"),
    ("die", "V", {"valency": 1, "pred": "died"}, "died"), ("died", "V", {"valency": 1, "pred": "died", "tense": "past"}, "died"),
    ("fail", "V", {"valency": 1, "pred": "failed"}, "failed"), ("failed", "V", {"valency": 1, "pred": "failed", "tense": "past"}, "failed"),
    ("come", "V", {"valency": 1, "pred": "coming"}, "coming"), ("comes", "V", {"valency": 1, "pred": "coming"}, "coming"),
    ("came", "V", {"valency": 1, "pred": "coming", "tense": "past"}, "coming"),
    ("raid", "V", {"valency": 2, "pred": "raided"}, "raided"), ("raided", "V", {"valency": 2, "pred": "raided", "tense": "past"}, "raided"),
    ("trade", "V", {"valency": 2, "pred": "trades"}, "trades"), ("trades", "V", {"valency": 2, "pred": "trades"}, "trades"),
    ("owe", "V", {"valency": 2, "pred": "owes"}, "owes"), ("owes", "V", {"valency": 2, "pred": "owes"}, "owes"),
    ("say", "V", {"valency": 2, "pred": "say", "clausal": True}, "say"), ("says", "V", {"valency": 2, "pred": "say", "clausal": True}, "say"),
    ("said", "V", {"valency": 2, "pred": "say", "clausal": True, "tense": "past"}, "say"),
    ("hear", "V", {"valency": 2, "pred": "hear", "clausal": True}, "hear"), ("heard", "V", {"valency": 2, "pred": "hear", "clausal": True, "tense": "past"}, "hear"),
    ("know", "V", {"valency": 2, "pred": "know", "clausal": True}, "know"), ("knows", "V", {"valency": 2, "pred": "know", "clausal": True}, "know"),
    ("show", "V", {"valency": 2, "pred": "showed"}, "showed"), ("showed", "V", {"valency": 2, "pred": "showed", "tense": "past"}, "showed"),
    ("give", "V", {"valency": 3, "pred": "gives"}, "gives"), ("gave", "V", {"valency": 3, "pred": "gives", "tense": "past"}, "gives"),
    ("bring", "V", {"valency": 2, "pred": "bring"}, "bring"), ("go", "V", {"valency": 1, "pred": "go"}, "go"),
    ("work", "V", {"valency": 1, "pred": "work"}, "work"), ("help", "V", {"valency": 2, "pred": "help"}, "help"),
    ("be", "V", {"valency": 2, "pred": "is", "copula": True}, "is"), ("is", "V", {"valency": 2, "pred": "is", "copula": True}, "is"),
    ("are", "V", {"valency": 2, "pred": "is", "copula": True, "num": "pl"}, "is"), ("was", "V", {"valency": 2, "pred": "is", "copula": True, "tense": "past"}, "is"),
    ("were", "V", {"valency": 2, "pred": "is", "copula": True, "tense": "past", "num": "pl"}, "is"),
    # auxiliaries, modals, negation, complementizer, conjunctions, prepositions
    ("do", "AUX", {}, None), ("does", "AUX", {}, None), ("did", "AUX", {"tense": "past"}, None),
    ("will", "MODAL", {"modality": None, "tense": "future"}, None), ("might", "MODAL", {"modality": "might"}, None),
    ("must", "MODAL", {"modality": "must"}, None), ("should", "MODAL", {"modality": "should"}, None),
    ("can", "MODAL", {"modality": "can"}, None),
    ("not", "NEG", {}, None), ("never", "NEG", {}, None), ("no", "NEG", {}, None),
    ("that", "COMP", {}, None), ("and", "CONJ", {}, None), ("but", "CONJ", {}, None), ("or", "CONJ", {"disj": True}, None),
    ("of", "P", {"rel": "of"}, None), ("at", "P", {"rel": "at"}, None), ("in", "P", {"rel": "in"}, None),
    ("from", "P", {"rel": "from"}, None), ("to", "P", {"rel": "to"}, None), ("with", "P", {"rel": "with"}, None),
    ("for", "P", {"rel": "for"}, None), ("than", "P", {"rel": "than"}, None),
    # adjectives, amounts, comparatives, adverbs
    ("much", "AMOUNT", {"amount": "much"}, "much"), ("plenty", "AMOUNT", {"amount": "much"}, "much"),
    ("heaps", "AMOUNT", {"amount": "much"}, "much"), ("little", "AMOUNT", {"amount": "little"}, "little"),
    ("scarcely", "AMOUNT", {"amount": "little"}, "little"), ("hardly", "AMOUNT", {"amount": "little"}, "little"),
    ("thin", "AMOUNT", {"amount": "little"}, "little"), ("none", "AMOUNT", {"amount": "none"}, "none"),
    ("empty", "AMOUNT", {"amount": "none"}, "none"), ("full", "AMOUNT", {"amount": "much"}, "much"),
    ("dead", "ADJ", {"pred": "died"}, "dead"), ("hungry", "ADJ", {"pred": "hungry"}, "hungry"),
    ("cold", "ADJ", {"pred": "cold"}, "cold"), ("ill", "ADJ", {"pred": "ill"}, "ill"), ("sick", "ADJ", {"pred": "ill"}, "ill"),
    ("good", "ADJ", {"pred": "good"}, "good"), ("bad", "ADJ", {"pred": "bad"}, "bad"),
    ("trustworthy", "ADJ", {"pred": "trustworthy"}, "trustworthy"), ("dear", "ADJ", {"pred": "expensive"}, "expensive"),
    ("cheap", "ADJ", {"pred": "cheap"}, "cheap"), ("more", "COMPAR", {"cmp": "more"}, "more"), ("less", "COMPAR", {"cmp": "less"}, "less"),
    ("dearer", "COMPAR", {"cmp": "more", "about": "price"}, "dearer"),
    ("here", "ADV", {"place": "here"}, "here"), ("there", "ADV", {"place": "there"}, "there"),
    ("tomorrow", "ADV", {"time": "future"}, "tomorrow"), ("yesterday", "ADV", {"time": "past"}, "yesterday"),
    ("please", "POLITE", {}, None), ("why", "WH", {"wh": "why"}, None), ("where", "WH", {"wh": "where"}, None),
    ("who", "WH", {"wh": "who"}, None), ("what", "WH", {"wh": "what"}, None), ("how", "WH", {"wh": "how"}, None),
)

SUFFIXES = (("ing", {"aspect": "progressive"}), ("ed", {"tense": "past"}), ("s", {}))
IRREGULAR = {"fell": "fall", "gave": "give", "brought": "bring", "went": "go", "told": "tell", "spoke": "speak"}


@dataclass
class Rule:
    lhs: str
    rhs: tuple
    build: object = None  # (children_semantics, features) -> semantics
    feats: dict = field(default_factory=dict)


def _s(children, i):
    return children[i] if i < len(children) else None


# The grammar. Each rule's ``build`` turns child semantics into a Frame or a referent string.
GRAMMAR: tuple = (
    Rule("S", ("NP", "VP"), lambda c, f: _attach_subject(_s(c, 1), _s(c, 0))),
    Rule("S", ("VP",), lambda c, f: _mood(_s(c, 0), "command")),
    Rule("S", ("POLITE", "S"), lambda c, f: _mood(_s(c, 1), "command")),
    Rule("S", ("AUX", "NP", "VP"), lambda c, f: _mood(_attach_subject(_s(c, 2), _s(c, 1)), "ask")),
    Rule("S", ("MODAL", "NP", "VP"), lambda c, f: _mood(_attach_subject(_s(c, 2), _s(c, 1)), "ask")),
    Rule("S", ("WH", "AUX", "NP", "VP"), lambda c, f: _wh(_attach_subject(_s(c, 3), _s(c, 2)), f)),
    Rule("S", ("WH", "VP"), lambda c, f: _wh(_s(c, 1), f)),
    Rule("S", ("S", "CONJ", "S"), lambda c, f: _conj(_s(c, 0), _s(c, 2))),
    Rule("VP", ("V",), lambda c, f: _s(c, 0)),
    Rule("VP", ("V", "NP"), lambda c, f: _arg(_s(c, 0), "object", _s(c, 1))),
    Rule("VP", ("V", "NP", "NP"), lambda c, f: _arg(_arg(_s(c, 0), "recipient", _s(c, 1)), "object", _s(c, 2))),
    Rule("VP", ("V", "AMOUNT", "NP"), lambda c, f: _arg(_arg(_s(c, 0), "amount", _s(c, 1)), "object", _s(c, 2))),
    Rule("VP", ("V", "NP", "PP"), lambda c, f: _pp(_arg(_s(c, 0), "object", _s(c, 1)), _s(c, 2))),
    Rule("VP", ("V", "PP"), lambda c, f: _pp(_s(c, 0), _s(c, 1))),
    Rule("VP", ("V", "ADJ"), lambda c, f: _predicative(_s(c, 0), _s(c, 1))),
    Rule("VP", ("V", "AMOUNT"), lambda c, f: _arg(_s(c, 0), "amount", _s(c, 1))),
    Rule("VP", ("V", "COMP", "S"), lambda c, f: _arg(_s(c, 0), "content", _s(c, 2))),
    Rule("VP", ("V", "S"), lambda c, f: _arg(_s(c, 0), "content", _s(c, 1))),
    Rule("VP", ("V", "PP", "COMP", "S"), lambda c, f: _arg(_pp(_s(c, 0), _s(c, 1)), "content", _s(c, 3))),
    Rule("VP", ("V", "PP", "S"), lambda c, f: _arg(_pp(_s(c, 0), _s(c, 1)), "content", _s(c, 2))),
    Rule("VP", ("MODAL", "VP"), lambda c, f: _modal(_s(c, 1), f)),
    Rule("VP", ("AUX", "NEG", "VP"), lambda c, f: _negate(_s(c, 2))),
    Rule("VP", ("V", "NEG", "NP"), lambda c, f: _negate(_arg(_s(c, 0), "object", _s(c, 2)))),
    Rule("VP", ("NEG", "VP"), lambda c, f: _negate(_s(c, 1))),
    Rule("VP", ("V", "NEG", "ADJ"), lambda c, f: _negate(_predicative(_s(c, 0), _s(c, 2)))),
    Rule("VP", ("V", "NEG", "AMOUNT"), lambda c, f: _negate(_arg(_s(c, 0), "amount", _s(c, 2)))),
    Rule("VP", ("V", "AMOUNT", "P", "NP"), lambda c, f: _arg(_arg(_s(c, 0), "amount", _s(c, 1)), "object", _s(c, 3))),
    Rule("VP", ("VP", "ADV"), lambda c, f: _adv(_s(c, 0), f)),
    Rule("VP", ("VP", "PP"), lambda c, f: _pp(_s(c, 0), _s(c, 1))),
    Rule("VP", ("V", "COMPAR", "PP"), lambda c, f: _compare(_s(c, 0), _s(c, 1), _s(c, 2))),
    Rule("NP", ("PN",), lambda c, f: _s(c, 0)),
    Rule("NP", ("PRON",), lambda c, f: _s(c, 0)),
    Rule("NP", ("N",), lambda c, f: _s(c, 0)),
    Rule("NP", ("DET", "N"), lambda c, f: _s(c, 1)),
    Rule("NP", ("DET", "ADJ", "N"), lambda c, f: f"{_s(c, 1)}:{_s(c, 2)}"),
    Rule("NP", ("QUANT", "N"), lambda c, f: _s(c, 1)),
    Rule("NP", ("AMOUNT", "N"), lambda c, f: _s(c, 1)),
    Rule("NP", ("AMOUNT", "P", "NP"), lambda c, f: _s(c, 2)),
    Rule("NP", ("QUANT", "P", "NP"), lambda c, f: _s(c, 2)),
    Rule("NP", ("NP", "PP"), lambda c, f: _of(_s(c, 0), _s(c, 1))),
    Rule("NP", ("PN", "POSS", "N"), lambda c, f: f"{_s(c, 2)}@{_s(c, 0)}"),
    Rule("NP", ("NP", "CONJ", "NP"), lambda c, f: f"{_s(c, 0)}+{_s(c, 2)}"),
    Rule("PP", ("P", "NP"), lambda c, f: (f.get("rel", "of"), _s(c, 1))),
)


# ------------------------------------------------------- semantic assembly


def _as_frame(x) -> Frame:
    return x if isinstance(x, Frame) else Frame(pred=str(x), args=())


def _attach_subject(vp, subject) -> Frame:
    fr = _as_frame(vp)
    if fr.get("subject") is not None:
        return fr
    return Frame(fr.pred, (("subject", subject),) + fr.args, fr.tense, fr.mood, fr.modality, fr.negated, fr.quantifier)


def _arg(v, role: str, value) -> Frame:
    fr = _as_frame(v)
    return Frame(fr.pred, fr.args + ((role, value),), fr.tense, fr.mood, fr.modality, fr.negated, fr.quantifier)


def _pp(v, pp) -> Frame:
    fr = _as_frame(v)
    rel, obj = pp if isinstance(pp, tuple) else ("of", pp)
    return Frame(fr.pred, fr.args + ((rel, obj),), fr.tense, fr.mood, fr.modality, fr.negated, fr.quantifier)


def _of(head, pp):
    rel, obj = pp if isinstance(pp, tuple) else ("of", pp)
    return f"{head}@{obj}" if rel in ("of", "at", "in") else head


def _predicative(v, adj) -> Frame:
    fr = _as_frame(v)
    return Frame(str(adj), fr.args, fr.tense, fr.mood, fr.modality, fr.negated, fr.quantifier)


def _negate(v) -> Frame:
    fr = _as_frame(v)
    return Frame(fr.pred, fr.args, fr.tense, fr.mood, fr.modality, True, fr.quantifier)


def _modal(v, feats) -> Frame:
    fr = _as_frame(v)
    return Frame(fr.pred, fr.args, feats.get("tense", fr.tense), fr.mood, feats.get("modality") or fr.modality, fr.negated, fr.quantifier)


def _mood(v, mood: str) -> Frame:
    fr = _as_frame(v)
    return Frame(fr.pred, fr.args, fr.tense, mood, fr.modality, fr.negated, fr.quantifier)


def _wh(v, feats) -> Frame:
    fr = _mood(v, "ask")
    return Frame(fr.pred, fr.args + (("wh", feats.get("wh", "what")),), fr.tense, "ask", fr.modality, fr.negated, fr.quantifier)


def _adv(v, feats) -> Frame:
    fr = _as_frame(v)
    tense = "future" if feats.get("time") == "future" else "past" if feats.get("time") == "past" else fr.tense
    args = fr.args + ((("place", feats["place"]),) if "place" in feats else ())
    return Frame(fr.pred, args, tense, fr.mood, fr.modality, fr.negated, fr.quantifier)


def _conj(a, b) -> Frame:
    return Frame("and", (("first", _as_frame(a)), ("second", _as_frame(b))))


def _compare(v, cmp, pp) -> Frame:
    fr = _as_frame(v)
    rel, obj = pp if isinstance(pp, tuple) else ("than", pp)
    return Frame("compare", fr.args + (("direction", str(cmp)), (rel, obj)), fr.tense, fr.mood, fr.modality, fr.negated, fr.quantifier)


# ------------------------------------------------------------ the chart parser


@dataclass
class Lexicon:
    """A dialect's words. ``say`` maps a concept to this dialect's surface form."""

    entries: dict = field(default_factory=dict)  # surface -> [(category, feats, sem)]
    forms: dict = field(default_factory=dict)  # concept -> surface (for generation)

    @classmethod
    def base(cls) -> "Lexicon":
        lex = cls()
        for surface, cat, feats, sem in BASE_LEXICON:
            lex.entries.setdefault(surface, []).append((cat, dict(feats), sem))
            if sem is not None and sem not in lex.forms:
                lex.forms[sem] = surface
        for concept, surface in (("food", "food"), ("store", "granary"), ("much", "much"), ("little", "little"), ("none", "no")):
            lex.forms[concept] = surface
        return lex

    def copy(self) -> "Lexicon":
        out = Lexicon({k: list(v) for k, v in self.entries.items()}, dict(self.forms))
        return out

    def add(self, surface: str, cat: str, feats: dict, sem) -> None:
        self.entries.setdefault(surface, []).append((cat, dict(feats), sem))
        self.forms.setdefault(sem, surface)

    def lookup(self, word: str) -> list:
        w = word.lower().strip(".,!?;:'\"")
        if w in self.entries:
            return [(c, dict(f), s) for c, f, s in self.entries[w]]
        if w in IRREGULAR and IRREGULAR[w] in self.entries:
            return [(c, dict(f) | {"tense": "past"}, s) for c, f, s in self.entries[IRREGULAR[w]]]
        for suffix, extra in SUFFIXES:  # morphology: strip an affix and try the lemma
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                stem = w[: -len(suffix)]
                for cand in (stem, stem + "e"):
                    if cand in self.entries:
                        return [(c, dict(f) | extra, s) for c, f, s in self.entries[cand]]
        return []

    def say(self, concept: str, default: str | None = None) -> str:
        return self.forms.get(concept, default if default is not None else concept)


@dataclass
class Parse:
    frame: Frame
    score: float
    unknown: tuple
    ambiguity: int


class _Item:
    __slots__ = ("rule", "dot", "start", "feats", "kids")

    def __init__(self, rule: Rule, dot: int, start: int, feats: dict, kids: tuple) -> None:
        self.rule, self.dot, self.start, self.feats, self.kids = rule, dot, start, feats, kids

    @property
    def done(self) -> bool:
        return self.dot >= len(self.rule.rhs)

    def next_symbol(self):
        return self.rule.rhs[self.dot] if not self.done else None

    def key(self):
        return (id(self.rule), self.dot, self.start, tuple(sorted((k, str(v)) for k, v in self.feats.items())), self.kids)


BY_LHS: dict = {}
for _r in GRAMMAR:
    BY_LHS.setdefault(_r.lhs, []).append(_r)

NAME_RE = re.compile(r"^[A-Z][a-z][\w'-]*$")


def tokenize(sentence: str) -> list:
    return [t for t in re.findall(r"[\w'’-]+|[.,!?;:]", sentence) if t not in ".,!?;:"]


def parse(sentence: str, lex: Lexicon, *, known_names: Iterable[str] = (), max_parses: int = 40) -> list:
    """Earley chart parse; returns ranked Parse objects (possibly empty)."""
    words = tokenize(sentence)
    if not words:
        return []
    names = {n.lower() for n in known_names}
    lexes: list = []
    unknown: list = []
    for w in words:
        got = lex.lookup(w)
        if not got:
            if NAME_RE.match(w) or w.lower() in names:
                got = [("PN", {"num": "sg"}, w.title())]
            else:
                unknown.append(w)
                got = [("N", {"kind": "unknown"}, f"?{w.lower()}"), ("V", {"valency": 1, "pred": f"?{w.lower()}"}, f"?{w.lower()}")]
        lexes.append(got)
    n = len(words)
    chart: list = [dict() for _ in range(n + 1)]

    def add(col: int, item: _Item, agenda: list | None = None) -> bool:
        k = item.key()
        if k in chart[col]:
            return False
        chart[col][k] = item
        if agenda is not None:
            agenda.append(item)
        return True

    for rule in BY_LHS.get("S", ()):
        add(0, _Item(rule, 0, 0, {}, ()))
    for col in range(n + 1):
        agenda = list(chart[col].values())
        while agenda:
            item = agenda.pop()
            if item.done:  # complete: advance every item waiting on this symbol
                sem = item.rule.build(item.kids, item.feats) if item.rule.build else (item.kids[0] if item.kids else None)
                for prev in list(chart[item.start].values()):
                    if prev.done or prev.next_symbol() != item.rule.lhs:
                        continue
                    merged = unify(prev.feats, {k: v for k, v in item.feats.items() if k in PROPAGATE})
                    if merged is None:
                        continue
                    add(col, _Item(prev.rule, prev.dot + 1, prev.start, merged, prev.kids + (sem,)), agenda)
                continue
            sym = item.next_symbol()
            if sym in BY_LHS:  # predict
                for rule in BY_LHS[sym]:
                    add(col, _Item(rule, 0, col, {}, ()), agenda)
            if col < n:  # scan (the new item belongs to the next column, so it is not on this agenda)
                for cat, feats, sem in lexes[col]:
                    if cat != sym:
                        continue
                    merged = unify(item.feats, {k: v for k, v in feats.items() if k in AGREEMENT})
                    if merged is None:
                        continue
                    add(col + 1, _Item(item.rule, item.dot + 1, item.start, merged, item.kids + (sem,)))
    finished = [it for it in chart[n].values() if it.done and it.rule.lhs == "S" and it.start == 0]
    out: list = []
    for it in finished[:max_parses]:
        sem = it.rule.build(it.kids, it.feats) if it.rule.build else None
        if isinstance(sem, Frame):
            if sem.get("subject") is None:
                if sem.mood == "command":
                    sem = _attach_subject(sem, "listener")  # an imperative is addressed to whoever is listening
                elif sem.mood != "ask" and sem.pred != "and":  # a question may be about the unknown subject
                    continue
            out.append(sem)
    # rank: prefer fewer unknown words, a known subject, a filled object, and a shallower frame
    def score(fr: Frame) -> float:
        s = 2.0 - 0.6 * len(unknown)
        subj = fr.get("subject")
        if isinstance(subj, str) and subj.lower() in names:
            s += 0.6
        if fr.get("object") is not None or fr.get("amount") is not None or fr.get("content") is not None:
            s += 0.4
        if str(fr.pred).startswith("?"):
            s -= 1.0
        s -= 0.15 * sum(1 for _, v in fr.args if isinstance(v, str) and v.startswith("?"))
        return s

    ranked = sorted({fr.describe(): fr for fr in out}.values(), key=lambda fr: -score(fr))
    return [Parse(fr, round(score(fr), 3), tuple(unknown), len(ranked)) for fr in ranked]


# --------------------------------------------------------------- generation


def generate(frame: Frame, lex: Lexicon, *, subject_name: str | None = None, nested: bool = False) -> str:
    """Realize a frame as a sentence in this dialect, using the same categories the parser reads."""
    subj = subject_name or frame.get("subject") or "someone"
    pred = frame.pred
    words: list = []

    OBJECT_PRONOUN = {"speaker": "me", "listener": "you", "group": "us", "anaphor": "them", "people": "everyone"}

    def np(value, subject_position: bool = False) -> str:
        if value is None:
            return ""
        v = str(value)
        if "@" in v:
            head, owner = v.split("@", 1)
            return f"{owner}'s {lex.say(head)}"
        if not subject_position and v in OBJECT_PRONOUN:
            return OBJECT_PRONOUN[v]
        if subject_position and v in ("speaker", "listener", "group", "people"):
            return {"speaker": "I", "listener": "you", "group": "we", "people": "everyone"}[v]
        return lex.say(v, v)

    if frame.mood == "ask":
        if frame.get("wh"):
            words.append({"why": "why", "where": "where", "who": "who", "what": "what", "how": "how"}[str(frame.get("wh"))])
        words.append("does" if frame.tense != "past" else "did")
        words.append(np(subj, True))
        words.append(lex.say(pred + ":verb", _verb_word(pred, "base")))
    elif frame.mood == "command":
        if frame.modality == "should":
            words += [np(subj, True), "should", _verb_word(pred, "base")]
        else:
            words += [_verb_word(pred, "base")]
    else:
        words.append(np(subj, True))
        if frame.modality:
            words.append(str(frame.modality))
            if frame.negated:
                words.append("not")
            words.append(_verb_word(pred, "base"))
        else:
            if frame.negated:
                words += ["does" if frame.tense != "past" else "did", "not", _verb_word(pred, "base")]
            else:
                words.append(_verb_word(pred, frame.tense))
    amount = frame.get("amount")
    if amount is not None:
        word = lex.say(str(amount))
        words.append(word + (" of" if word in ("heaps", "plenty", "scarcely", "hardly") else ""))
    # complements last: "I heard from Miol that ..." reads correctly only in this order
    for role in ("recipient", "object", "at", "in", "from", "to", "with", "than", "content"):
        val = frame.get(role)
        if val is None:
            continue
        if isinstance(val, Frame):
            words += ["that", generate(val, lex, nested=True)]
        elif role in ("at", "in", "from", "to", "with", "than"):
            words += [role, np(val)]
        else:
            words.append(np(val))
    sentence = " ".join(w for w in words if w)
    sentence = re.sub(r"\s+([,.!?])", r"\1", sentence).strip()
    if nested:
        return sentence
    return sentence[:1].upper() + sentence[1:] + ("?" if frame.mood == "ask" else ".")


PRED_WORDS = {
    "has_amount": ("hold", "holds", "held"), "keeps_back": ("keep", "keeps", "kept"), "died": ("die", "dies", "died"),
    "failed": ("fail", "fails", "failed"), "coming": ("come", "comes", "came"), "raided": ("raid", "raids", "raided"),
    "trades": ("trade", "trades", "traded"), "owes": ("owe", "owes", "owed"), "say": ("say", "says", "said"),
    "hear": ("hear", "hears", "heard"), "know": ("know", "knows", "knew"), "showed": ("show", "shows", "showed"),
    "gives": ("give", "gives", "gave"), "bring": ("bring", "brings", "brought"), "go": ("go", "goes", "went"),
    "work": ("work", "works", "worked"), "help": ("help", "helps", "helped"), "is": ("be", "is", "was"),
    "hungry": ("be hungry", "is hungry", "was hungry"), "cold": ("be cold", "is cold", "was cold"),
    "ill": ("be ill", "is ill", "was ill"), "dead": ("be dead", "is dead", "was dead"),
    "trustworthy": ("be trustworthy", "is trustworthy", "was trustworthy"),
    "expensive": ("be dear", "is dear", "was dear"), "cheap": ("be cheap", "is cheap", "was cheap"),
}


def _verb_word(pred: str, tense: str) -> str:
    base, third, past = PRED_WORDS.get(pred, (pred, pred, pred))
    return {"base": base, "present": third, "past": past, "future": base}.get(tense, third)


# ------------------------------------------------- claims, and the other way


def frame_to_claims(frame: Frame, *, speaker: str, listener_context: dict | None = None) -> list:
    """A frame becomes (subject, predicate, object) claims, with reported speech kept as provenance."""
    ctx = listener_context or {}
    out: list = []

    def resolve(value):
        if value is None:
            return None
        v = str(value)
        if v == "speaker":
            return f"person:{speaker}"
        if v == "listener":
            return f"person:{ctx.get('listener', 'you')}"
        if v == "anaphor":
            return ctx.get("last_person") or ctx.get("last_thing") or "person:someone"
        if v == "group":
            return f"settlement:{ctx.get('settlement', 'here')}"
        if "@" in v:
            head, owner = v.split("@", 1)
            return f"settlement:{owner}" if head in ("store", "granary", "market") else f"place:{owner}:{head}"
        if v in ("food", "wood", "stone", "tool"):
            return f"good:{v}"
        if v in ("rain", "snow", "frost"):
            return f"weather:{v}"
        if v in ("sky", "moon"):
            return f"sky:{v}"
        if v.startswith("?"):
            return None
        return f"person:{v}" if v[:1].isupper() else f"thing:{v}"

    subj = resolve(frame.get("subject"))
    if frame.pred == "say" or frame.pred == "hear":
        inner = frame.get("content")
        if isinstance(inner, Frame):
            said_by = frame.get("subject")
            for claim in frame_to_claims(inner, speaker=str(said_by), listener_context=ctx):
                out.append({**claim, "via": str(said_by), "hearsay": True})
        return out
    obj = frame.get("object")
    amount = frame.get("amount")
    predicate = frame.pred
    value = None
    if predicate == "has_amount":
        goods = resolve(obj) or "good:food"
        predicate = "has_amount"
        value = f"{goods.split(':')[-1]}:{amount or 'some'}"
    elif predicate in ("died", "failed", "coming", "hungry", "cold", "ill", "dead", "trustworthy", "expensive", "cheap"):
        value = str(frame.get("of") or frame.get("with") or obj or True)
    elif predicate in ("raided", "trades", "owes", "gives", "help", "showed", "keeps_back"):
        value = str(resolve(obj) or obj or True)
    else:
        value = str(resolve(obj) or obj or True)
    if subj is None:
        return out
    out.append({"subject": subj, "predicate": predicate, "object": value, "negated": frame.negated,
                "modality": frame.modality, "tense": frame.tense, "mood": frame.mood,
                "quantifier": frame.quantifier, "via": None, "hearsay": False})
    return out


def claim_to_frame(subject: str, predicate: str, obj, *, tense: str = "present", modality: str | None = None,
                   negated: bool = False, mood: str = "declare") -> Frame:
    """The inverse: a stored claim becomes a frame the grammar can speak."""
    args: list = [("subject", _short(subject))]
    if predicate == "has_amount":
        good, _, amount = str(obj).partition(":")
        args.append(("amount", amount or "some"))
        args.append(("object", good or "food"))
    elif obj not in (True, "True", None):
        args.append(("object", _short(str(obj))))
    return Frame(predicate, tuple(args), tense, mood, modality, negated)


def _short(ref: str) -> str:
    s = str(ref)
    if ":" in s:
        kind, name = s.split(":", 1)
        if kind in ("person", "settlement", "village"):
            return name
        if kind in ("good", "weather", "sky", "thing", "place"):
            return name.split(":")[-1]
    return s


# ------------------------------------------------------- dialects and drift

SOUND_CHANGES = (("th", "d"), ("ee", "i"), ("oo", "u"), ("k", "c"), ("gh", ""), ("ai", "e"), ("ou", "o"), ("v", "f"), ("z", "s"))
SYNONYMS = {"food": ("food", "grain", "bread", "forage"), "store": ("granary", "store", "loft", "hoard"),
            "much": ("much", "plenty", "heaps", "full"), "little": ("little", "thin", "scarcely", "hardly")}


def dialect_from(base: Lexicon, seed: int, rng) -> Lexicon:
    """A settlement's dialect: pick synonyms, then apply a couple of sound changes to some words."""
    lex = base.copy()
    for concept, options in SYNONYMS.items():
        keep = options[seed % len(options)]
        lex.forms[concept] = keep
        for rival in options:
            if rival != keep and rival in lex.entries:
                lex.entries[rival] = [e for e in lex.entries[rival] if e[2] != concept] or []
                if not lex.entries[rival]:
                    del lex.entries[rival]
    for _ in range(2):
        a, b = SOUND_CHANGES[int(rng.integers(0, len(SOUND_CHANGES)))]
        for concept, surface in list(lex.forms.items()):
            if a in surface and rng.random() < 0.4:
                new = surface.replace(a, b, 1)
                if new and new != surface:
                    lex.forms[concept] = new
                    for cat, feats, sem in list(lex.entries.get(surface, [])):
                        lex.add(new, cat, feats, sem)  # speakers still understand the old form
    return lex


def drift(lex: Lexicon, rng, *, borrow_from: Lexicon | None = None, rate: float = 0.08) -> int:
    """One generation of change: a sound change, a semantic shift, or a borrowing. Returns changes made."""
    changed = 0
    concepts = list(lex.forms)
    for concept in concepts:
        if rng.random() > rate:
            continue
        surface = lex.forms[concept]
        if borrow_from is not None and rng.random() < 0.5:
            loan = borrow_from.forms.get(concept)
            if loan and loan != surface:
                lex.forms[concept] = loan
                for cat, feats, sem in list(borrow_from.entries.get(loan, [])) or [("N", {"kind": "good"}, concept)]:
                    if not any(c == cat for c, _f, _s in lex.entries.get(loan, [])):
                        lex.add(loan, cat, feats, sem)
                changed += 1
                continue
        for _ in range(3):  # try a few changes; not every rule applies to every word
            a, b = SOUND_CHANGES[int(rng.integers(0, len(SOUND_CHANGES)))]
            if a not in surface:
                continue
            new = surface.replace(a, b, 1)
            if not new or new == surface:
                continue
            lex.forms[concept] = new
            for cat, feats, sem in list(lex.entries.get(surface, [])):
                lex.add(new, cat, feats, sem)
            if rng.random() < 0.5 and surface in lex.entries:  # the old form is forgotten by some
                lex.entries[surface] = [e for e in lex.entries[surface] if e[2] != concept]
                if not lex.entries[surface]:
                    del lex.entries[surface]
            changed += 1
            break
    return changed


def intelligibility(speaker: Lexicon, listener: Lexicon, probes: tuple = ()) -> float:
    """Share of probe claims that survive being spoken by one dialect and parsed by the other."""
    probes = probes or (("settlement:Aldmere", "has_amount", "food:much"), ("person:Kalo", "died", "True"),
                        ("settlement:Brenholt", "raided", "settlement:Coralin"), ("weather:snow", "coming", "True"),
                        ("person:Miol", "keeps_back", "good:food"))
    ok = 0
    for subject, predicate, obj in probes:
        sentence = generate(claim_to_frame(subject, predicate, obj), speaker)
        got = parse(sentence, listener, known_names=[_short(subject)])
        if got and got[0].frame.pred == predicate and not got[0].unknown and not any(
                isinstance(v, str) and v.startswith("?") for _r, v in got[0].frame.args):
            ok += 1
    return round(ok / len(probes), 3)
