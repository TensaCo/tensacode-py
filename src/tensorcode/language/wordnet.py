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
from typing import Mapping

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
#: Bumped whenever the entries built from WordNet change, so a cache from older code is not reused.
BUILD_VERSION = 2
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
    key = hashlib.sha256((f"v{BUILD_VERSION}|" + files.digest() + "|" + ",".join(sorted(closed))).encode()).hexdigest()[:16]
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


# ------------------------------------------------------------------ the is-a hierarchy


@dataclass(frozen=True)
class Taxonomy:
    """WordNet's noun hierarchy by lemma: what each word can be a kind of.

    ``kinds("folder")`` is every lemma on a hypernym path from any noun sense of
    "folder" (instance-of links count). It is used as a *type check* — may this
    description fill a slot that wants a "container"? — never to pick an action.
    """

    senses: Mapping[str, tuple[int, ...]]       # lemma -> synset offsets, most used first
    parents: Mapping[int, tuple[int, ...]]      # synset -> hypernym synsets
    lemmas: Mapping[int, tuple[str, ...]]       # synset -> its lemmas

    def kinds(self, lemma: str, *, senses: int | None = None) -> frozenset[str]:
        """Lemmas ``lemma`` is a kind of (including itself), over its first ``senses`` senses."""
        roots = self.senses.get(lemma.lower().replace(" ", "_"), ())
        if senses is not None:
            roots = roots[:senses]
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            stack.extend(self.parents.get(s, ()))
        out = {lemma.lower()}
        for s in seen:
            out.update(w.replace("_", " ") for w in self.lemmas.get(s, ()))
        return frozenset(out)

    def is_a(self, lemma: str, kind: str) -> bool:
        return kind.lower() in self.kinds(lemma)


def read_taxonomy(root: Path) -> Taxonomy:
    files = _Files(root)
    senses: dict[str, tuple[int, ...]] = {}
    for line in files.text("index.noun").splitlines():
        if not line or line.startswith(" "):
            continue
        parts = line.split()
        p_cnt = int(parts[3])
        offsets = parts[4 + p_cnt + 2:]  # after pointer symbols, sense_cnt and tagsense_cnt
        senses[parts[0]] = tuple(int(o) for o in offsets)
    parents: dict[int, tuple[int, ...]] = {}
    lemmas: dict[int, tuple[str, ...]] = {}
    for line in files.text("data.noun").splitlines():
        if not line or line.startswith(" "):
            continue
        parts = line.split()
        offset, w_cnt = int(parts[0]), int(parts[3], 16)
        words = tuple(parts[4 + 2 * i].lower() for i in range(w_cnt))
        i = 4 + 2 * w_cnt
        p_cnt = int(parts[i])
        ups = []
        for j in range(p_cnt):
            sym, target, pos = parts[i + 1 + 4 * j], parts[i + 2 + 4 * j], parts[i + 3 + 4 * j]
            if sym in ("@", "@i") and pos == "n":
                ups.append(int(target))
        lemmas[offset] = words
        parents[offset] = tuple(ups)
    return Taxonomy(senses, parents, lemmas)


_TAXONOMY: Taxonomy | None = None


def taxonomy(root: Path | None = None) -> Taxonomy | None:
    """The noun hierarchy, built once per process and cached on disk; ``None`` without WordNet."""
    global _TAXONOMY
    if _TAXONOMY is not None:
        return _TAXONOMY
    root = root or find_wordnet()
    if root is None:
        return None
    cache = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode"))) / "lexicon"
    path = cache / f"taxonomy-{_Files(root).digest()}.pickle"
    if path.exists():
        try:
            _TAXONOMY = pickle.loads(path.read_bytes())
            return _TAXONOMY
        except Exception:  # noqa: BLE001 - rebuilt if unreadable
            pass
    _TAXONOMY = read_taxonomy(root)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(_TAXONOMY, protocol=pickle.HIGHEST_PROTOCOL))
    except OSError:
        pass
    return _TAXONOMY


# ------------------------------------------------------------------ verb domains


_VERB_DOMAINS: Mapping[str, tuple[str, ...]] | None = None


def read_verb_domains(root: Path) -> dict[str, tuple[str, ...]]:
    """Each verb lemma to the lexicographer categories of its senses, commonest sense first.

    WordNet groups verbs into twenty-odd files by what they are *about*:
    ``verb.communication`` (say, claim, ask), ``verb.cognition`` (think, believe, know),
    ``verb.motion``, ``verb.possession``, and so on. The number is the second field of a
    synset's line in ``data.verb``; ``lexnames`` gives the names.

    This is the closest thing WordNet has to saying which verbs take a *reported* argument,
    and it is curated data rather than a list written here.

    The order matters and the union does not: every sense of "buy" includes
    ``verb.cognition`` ("I don't buy it") and every sense of "delete" includes
    ``verb.communication``, so a test over all of a lemma's domains suppresses nearly
    everything. WordNet's index files list senses commonest first, so ``domains[0]`` is the
    reading a word most likely has.
    """
    files = _Files(root)
    names = {}
    for line in files.text("lexnames").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            names[int(parts[0])] = parts[1]
    domain_of_offset: dict[str, str] = {}
    for line in files.text("data.verb").splitlines():
        if not line or line.startswith(" "):
            continue
        parts = line.split(" ", 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        domain_of_offset[parts[0]] = names.get(int(parts[1]), "")
    out: dict[str, tuple[str, ...]] = {}
    for line in files.text("index.verb").splitlines():
        if not line or line.startswith(" "):
            continue
        parts = line.split()
        lemma, n_synsets = parts[0], int(parts[2])
        seen: list[str] = []
        for offset in parts[-n_synsets:]:
            domain = domain_of_offset.get(offset)
            if domain and domain not in seen:
                seen.append(domain)
        if seen:
            out[lemma] = tuple(seen)
    return out


def verb_domains(root: Path | None = None) -> Mapping[str, tuple[str, ...]]:
    """WordNet's verb categories, read once per process; empty without WordNet."""
    global _VERB_DOMAINS
    if _VERB_DOMAINS is not None:
        return _VERB_DOMAINS
    root = root or find_wordnet()
    _VERB_DOMAINS = read_verb_domains(root) if root is not None else {}
    return _VERB_DOMAINS
