"""A lexicon seeded from Princeton WordNet 3.0, read straight from its data files.

The core grammar knows about 155 word forms; every content word it meets is otherwise
guessed, and each guess multiplies the ambiguity the chart has to carry ("available
processes include milling" came out as *someone called Available processes things*).
WordNet lists about 150,000 English lemmas with their parts of speech, and its sense
counts (``cntlist.rev``, from hand-tagged text) say how often each lemma is used as a
noun, a verb, an adjective or an adverb. That turns into entry weights directly, so
which category a word prefers is measured, not hand-set.

This is *coverage*, not meaning: the entry's sense is the lemma. Whether WordNet's
categories help beyond coverage is a separate claim that has to beat a random
partition of matched balance (docs/revival/28 §28.6).

The data is not shipped. It is read from ``$TENSORCODE_WORDNET`` or an NLTK download
(``~/nltk_data/corpora/wordnet.zip``); without it, :func:`seed_lexicon` returns the
grammar's lexicon unchanged. No third-party package is imported.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .grammar import Entry, Lexicon

#: WordNet's part-of-speech letters, and the grammar category each becomes.
CATEGORY = {"n": "N", "v": "V", "a": "Adj", "s": "Adj", "r": "Adv"}
#: ``cntlist.rev`` sense keys give the part of speech as a number (``ss_type``).
SS_TYPE = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "s"}
FILES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}

#: A word the core grammar already has as a function word (``can``, ``will``, ``that``)
#: keeps it as its first reading; WordNet's content readings of it are pushed down by
#: this much, so "can you" stays a modal question rather than a tin being addressed.
CLOSED_CLASS_PENALTY = -1.5
#: Add-one smoothing over parts of speech: a lemma WordNet lists but no tagged text used
#: still gets a reading, just a less likely one.
SMOOTHING = 1.0


def find_wordnet() -> Path | None:
    """Where WordNet's data files are, if they are anywhere this machine knows about."""
    candidates = [os.environ.get("TENSORCODE_WORDNET"), "~/nltk_data/corpora/wordnet.zip",
                  "~/nltk_data/corpora/wordnet", "/usr/share/nltk_data/corpora/wordnet.zip",
                  "/usr/local/share/nltk_data/corpora/wordnet.zip"]
    for c in candidates:
        if c and Path(c).expanduser().exists():
            return Path(c).expanduser()
    return None


class _Files:
    """WordNet's files from a zip or a directory, by their bare names."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.zip = zipfile.ZipFile(root) if root.suffix == ".zip" else None
        if self.zip is not None:
            names = self.zip.namelist()
            self.prefix = next((n[: -len("index.noun")] for n in names if n.endswith("index.noun")), "")

    def text(self, name: str) -> str:
        if self.zip is not None:
            return self.zip.read(self.prefix + name).decode("utf-8", "replace")
        return (self.root / name).read_text("utf-8", "replace")

    def digest(self) -> str:
        h = hashlib.sha256()
        for name in ("index.noun", "index.verb", "index.adj", "index.adv", "cntlist.rev"):
            h.update(self.text(name).encode())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class Lemma:
    """What WordNet says about one spelling: its parts of speech and how often each is used."""

    word: str
    counts: tuple[tuple[str, float], ...]  # (pos letter, tagged-sense count), pos letters n/v/a/r
    senses: tuple[tuple[str, int], ...]    # (pos letter, number of senses)


def read_lemmas(root: Path) -> dict[str, Lemma]:
    """Every single-word lemma, with per-part-of-speech usage counts.

    Multiword lemmas (``laser_cutting``) are left out here: the tokenizer splits them,
    and reading them back needs a multiword rule in the grammar, not a lexicon entry.
    """
    files = _Files(root)
    senses: dict[str, dict[str, int]] = defaultdict(dict)
    for pos, name in FILES.items():
        for line in files.text(f"index.{name}").splitlines():
            if not line or line.startswith(" "):
                continue
            parts = line.split()
            lemma = parts[0]
            if "_" in lemma or not any(ch.isalpha() for ch in lemma):
                continue
            senses[lemma][pos] = int(parts[2])
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for line in files.text("cntlist.rev").splitlines():
        parts = line.split()
        if len(parts) < 3 or "%" not in parts[0]:
            continue
        lemma, rest = parts[0].split("%", 1)
        pos = SS_TYPE.get(rest[:1])
        if pos is None:
            continue
        counts[lemma][{"s": "a"}.get(pos, pos)] += float(parts[2])
    out = {}
    for lemma, by_pos in senses.items():
        out[lemma] = Lemma(lemma, tuple(sorted((p, counts[lemma].get(p, 0.0)) for p in by_pos)), tuple(sorted(by_pos.items())))
    return out


def read_exceptions(root: Path) -> dict[str, list[tuple[str, str]]]:
    """Irregular forms (``went -> go``, ``mice -> mouse``), by part of speech."""
    files = _Files(root)
    out: dict[str, list[tuple[str, str]]] = {}
    for pos, name in FILES.items():
        pairs = []
        for line in files.text(f"{name}.exc").splitlines():
            parts = line.split()
            if len(parts) >= 2 and "_" not in parts[0] and "_" not in parts[1]:
                pairs.append((parts[0], parts[1]))
        out[pos] = pairs
    return out


def entries_from(lemmas: dict[str, Lemma], exceptions: dict[str, list[tuple[str, str]]], *,
                 closed_class: frozenset[str] = frozenset()) -> list[Entry]:
    """Lexicon entries: one per (lemma, part of speech), weighted by log P(pos | word)."""
    out: list[Entry] = []
    for lemma in lemmas.values():
        total = sum(c for _, c in lemma.counts) + SMOOTHING * len(lemma.counts)
        for pos, count in lemma.counts:
            weight = math.log((count + SMOOTHING) / total)
            if lemma.word in closed_class:
                weight += CLOSED_CLASS_PENALTY
            # no number on nouns: the grammar's suffix rules add it, and they skip an entry
            # that already says (a "singular" tag would have hidden "processes")
            out.append(Entry(lemma.word, CATEGORY[pos], {"source": "wordnet"}, lemma.word, round(weight, 3)))
    irregular = {"n": {"number": "plural"}, "v": {"tense": "past"}}
    for pos, pairs in exceptions.items():
        feats = irregular.get(pos)
        if feats is None:
            continue
        for form, lemma in pairs:
            if lemma not in lemmas or form in closed_class:
                continue
            out.append(Entry(form, CATEGORY[pos], {**feats, "source": "wordnet"}, lemma, -0.3))
    return out


def seed_lexicon(base: Lexicon, *, root: Path | None = None, cache: Path | None = None) -> Lexicon:
    """``base`` extended with WordNet's content words, or ``base`` itself if WordNet is absent.

    Words ``base`` already knows keep their entries first; WordNet adds its readings
    after them, and a function word's content readings are penalised (see
    :data:`CLOSED_CLASS_PENALTY`). The built entries are cached per WordNet digest.
    """
    root = root or find_wordnet()
    if root is None:
        return base
    closed = frozenset(w for w, entries in base.entries.items() if any(e.cat not in ("N", "V", "Adj", "Adv", "Name") for e in entries))
    files = _Files(root)
    key = hashlib.sha256((files.digest() + "|" + ",".join(sorted(closed))).encode()).hexdigest()[:16]
    cache = cache or Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode"))) / "lexicon"
    path = cache / f"wordnet-{key}.pickle"
    entries = None
    if path.exists():
        try:
            entries = pickle.loads(path.read_bytes())
        except Exception:  # noqa: BLE001 - a stale or partial cache is rebuilt, not trusted
            entries = None
    if entries is None:
        entries = entries_from(read_lemmas(root), read_exceptions(root), closed_class=closed)
        try:
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(entries, protocol=pickle.HIGHEST_PROTOCOL))
        except OSError:
            pass
    known = {(w, e.cat) for w, es in base.entries.items() for e in es}
    return base.extend(*(e for e in entries if (e.word.lower(), e.cat) not in known))
