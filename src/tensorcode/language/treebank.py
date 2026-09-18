"""Grammar and vocabulary as *data*: read from a dependency treebank, not written by hand.

The hand-written English grammar (89 productions, 157 function words, hand-set weights)
is knowledge living in code. A treebank is the same knowledge as data, curated by
linguists and with a held-out split to measure against — the leapfrog seed for syntax,
as WordNet is for words and VerbNet for what verbs do.

This module reads CoNLL-U (Universal Dependencies) and turns it into:

* :func:`closed_class_entries` — the function words (determiners, pronouns, auxiliaries,
  prepositions, conjunctions, particles) with their morphological features and counts,
  so *which* reading a word prefers is estimated from usage instead of hand-weighted;
* the training material for the learned tagger and parser (:mod:`.learned_parser`).

The only judgement here is the alignment between UD's feature names and this grammar's
(``FEATURE_OF``, ``CATEGORY_OF``): two vocabularies for the same distinctions.

Data is not shipped: point ``$TENSORCODE_TREEBANK`` at a UD treebank directory, or put
one at ``~/.cache/tensorcode/seeds/UD_English-EWT`` (CC BY-SA 4.0).
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .grammar import Entry

#: UD part of speech -> this grammar's category.
CATEGORY_OF = {"DET": "Det", "PRON": "Pron", "AUX": "Aux", "ADP": "P", "CCONJ": "Conj", "SCONJ": "Comp",
               "PART": "Part", "NOUN": "N", "PROPN": "Name", "VERB": "V", "ADJ": "Adj", "ADV": "Adv",
               "NUM": "Num", "INTJ": "Intj", "PUNCT": "Punct", "SYM": "Sym", "X": "X"}

#: Closed classes: the words a grammar must know, as opposed to the open vocabulary WordNet covers.
CLOSED = ("DET", "PRON", "AUX", "ADP", "CCONJ", "SCONJ", "PART")

#: UD morphological features -> this grammar's features.
FEATURE_OF: Mapping[tuple[str, str], tuple[str, object]] = {
    ("Number", "Sing"): ("number", "singular"), ("Number", "Plur"): ("number", "plural"),
    ("Person", "1"): ("person", 1), ("Person", "2"): ("person", 2), ("Person", "3"): ("person", 3),
    ("Tense", "Past"): ("tense", "past"), ("Tense", "Pres"): ("tense", "present"),
    ("VerbForm", "Inf"): ("form", "infinitive"), ("VerbForm", "Part"): ("aspect", "participle"),
    ("VerbForm", "Ger"): ("aspect", "progressive"),
    ("Definite", "Def"): ("definite", True), ("Definite", "Ind"): ("definite", False),
    ("Poss", "Yes"): ("possessive", True),
    ("PronType", "Dem"): ("demonstrative", True), ("PronType", "Int"): ("interrogative", True),
    ("Degree", "Cmp"): ("degree", "comparative"), ("Degree", "Sup"): ("degree", "superlative"),
    ("Mood", "Imp"): ("mood", "imperative"),
    ("Polarity", "Neg"): ("polarity", "negative"),
}


#: VerbNet thematic roles a preposition can mark, as this grammar names them.
PREPOSITION_ROLE = {"Destination": "destination", "Goal": "destination", "Location": "location",
                    "Source": "source", "Initial_Location": "source", "Recipient": "recipient",
                    "Beneficiary": "recipient", "Instrument": "instrument"}


@dataclass(frozen=True)
class Token:
    id: int
    form: str
    lemma: str
    upos: str
    feats: tuple[tuple[str, str], ...]
    head: int
    deprel: str

    def features(self) -> dict:
        out: dict = {}
        for k, v in self.feats:
            got = FEATURE_OF.get((k, v))
            if got:
                out[got[0]] = got[1]
        if out.get("possessive") and "person" in out:
            out["possessor"] = out.pop("person")  # "my": the possessor's person, not the phrase's
        return out


Sentence = list[Token]


def find_treebank() -> Path | None:
    for c in (os.environ.get("TENSORCODE_TREEBANK"), "~/.cache/tensorcode/seeds/UD_English-EWT"):
        if c and Path(c).expanduser().is_dir():
            return Path(c).expanduser()
    return None


def read_conllu(path: Path) -> Iterator[Sentence]:
    sentence: Sentence = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            if sentence:
                yield sentence
            sentence = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if "-" in parts[0] or "." in parts[0]:  # multiword ranges and empty nodes
            continue
        feats = tuple(tuple(kv.split("=", 1)) for kv in parts[5].split("|") if "=" in kv)  # type: ignore[misc]
        sentence.append(Token(int(parts[0]), parts[1], parts[2], parts[3], feats, int(parts[6]), parts[7]))
    if sentence:
        yield sentence


def load(split: str = "train", root: Path | None = None) -> list[Sentence]:
    root = root or find_treebank()
    if root is None:
        return []
    files = sorted(root.glob(f"*-ud-{split}.conllu"))
    return [s for f in files for s in read_conllu(f)]


def closed_class_entries(sentences: Sequence[Sentence], *, min_count: int = 3) -> list[Entry]:
    """Function words with their features, weighted by log P(reading | word) from counts.

    One entry per (word, category, feature set) seen at least ``min_count`` times: the
    treebank's own frequencies decide which reading of "that" or "to" a parse prefers.
    """
    seen: Counter = Counter()
    per_word: Counter = Counter()
    for s in sentences:
        for t in s:
            if t.upos not in CLOSED:
                continue
            key = (t.form.lower(), t.upos, tuple(sorted(t.features().items(), key=repr)))
            seen[key] += 1
            per_word[t.form.lower()] += 1
    out = []
    for (word, upos, feats), n in seen.items():
        if n < min_count:
            continue
        weight = math.log(n / per_word[word])
        out.append(Entry(word, CATEGORY_OF[upos], {**dict(feats), "source": "treebank"}, None, round(weight, 3)))
    return out


def preposition_roles(verbnet: Mapping[str, Sequence] | None = None) -> dict[str, str]:
    """Which role each preposition marks, counted over VerbNet's frames.

    The hand-written table said "to" marks a destination; VerbNet's syntax frames say so
    thousands of times, with their own thematic roles, so the mapping is read from them.
    """
    from . import verbnet as vn

    lexicon = verbnet if verbnet is not None else vn.load()
    counts: dict[str, Counter] = defaultdict(Counter)
    for classes in lexicon.values():
        for vc in classes:
            for frame in vc.frames:
                prep = None
                for cat, value in frame.syntax:
                    if cat == "PREP":
                        prep = value
                    elif cat == "NP" and prep and value:
                        # only the roles a preposition itself marks: where something ends up,
                        # where it is, where it came from, who receives it, what it is done
                        # with. A verb decides the rest ("listen to" makes its object a
                        # Theme; "to" does not). Location and Destination stay apart here,
                        # because "in" is where a thing is and "into" is where it goes.
                        role = PREPOSITION_ROLE.get(value.strip())
                        if role:
                            for word in prep.replace("|", " ").split():
                                counts[word.lower()][role] += 1
                        prep = None
    return {word: c.most_common(1)[0][0] for word, c in counts.items() if sum(c.values()) >= 5}


def lemma_table(sentences: Sequence[Sentence]) -> dict[tuple[str, str], str]:
    """(word, part of speech) -> its most common lemma in the treebank."""
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for s in sentences:
        for t in s:
            counts[(t.form.lower(), t.upos)][t.lemma.lower()] += 1
    return {key: c.most_common(1)[0][0] for key, c in counts.items()}


def lemmatize(word: str, upos: str, table: Mapping[tuple[str, str], str]) -> str:
    """The treebank's lemma for a known word; otherwise WordNet's, and failing that, the word.

    WordNet knows the irregulars ("went" -> "go") and its suffix rules cover the regular
    ones, so nothing here needs a list of endings.
    """
    low = word.lower()
    got = table.get((low, upos))
    if got:
        return got
    from .english import ENGLISH_LEXICON
    from .wordnet import seed_lexicon

    lex = seed_lexicon(ENGLISH_LEXICON)
    want = {"NOUN": "N", "VERB": "V", "ADJ": "Adj", "ADV": "Adv", "PROPN": "Name"}.get(upos)
    for entry in lex.lookup(low):
        if entry.cat == want and isinstance(entry.sem, str):
            return entry.sem
    return low
