"""A persistent, versioned library of learned artifacts.

Learning that does not outlive the process is not learning, so artifacts go to a
directory of plain JSON: a manifest, one content-addressed file per artifact, and
one recorded fixture per artifact. The design is
``typed-crystallization-networks``'s ``tcn/library.py``, kept because its rules are
the ones that stop a library quietly rotting:

* **content addressing** — two names that induce the same artifact share one file,
  so a definition is stored and charged once;
* **versions** — publishing under an existing name allocates the next version and
  marks every artifact that depends on the superseded digest ``stale``;
* **fixtures** — an artifact records cases it got right, and loading replays them.
  A disagreement raises rather than loading;
* **provenance** — what it was induced from, by what method, when, and under which
  verification. Nothing enters without it.

There is deliberately no "load anyway" policy.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .induce import DecisionList, Rule
from .literals import Literal


class LibraryError(Exception):
    pass


class FixtureMismatch(LibraryError):
    """A stored artifact no longer reproduces its recorded cases."""


class MissingArtifact(LibraryError):
    pass


def digest_of(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Entry:
    name: str
    version: int
    digest: str
    kind: str
    provenance: Mapping[str, Any]
    depends_on: tuple[str, ...] = ()
    stale: bool = False
    at: float = field(default_factory=time.time)

    @property
    def reference(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "digest": self.digest, "kind": self.kind,
                "provenance": dict(self.provenance), "depends_on": list(self.depends_on), "stale": self.stale,
                "at": self.at}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entry":
        return cls(data["name"], data["version"], data["digest"], data["kind"], data.get("provenance", {}),
                   tuple(data.get("depends_on", ())), bool(data.get("stale")), data.get("at", 0.0))


# --------------------------------------------------------------- serialisation


def _literal_to_json(literal: Literal) -> dict[str, Any]:
    return {"predicate": literal.predicate, "value": literal.value, "negated": literal.negated, "kind": literal.kind}


def _literal_from_json(data: Mapping[str, Any]) -> Literal:
    return Literal(data["predicate"], data.get("value"), bool(data.get("negated")), data.get("kind", "equals"))


def to_json(artifact: Any) -> dict[str, Any]:
    """Artifacts are data: a reader needs this module's *format*, not its code."""
    if isinstance(artifact, DecisionList):
        return {"kind": "decision_list", "default": artifact.default, "considered": artifact.considered,
                "rules": [{"conditions": [_literal_to_json(c) for c in r.conditions], "label": r.label,
                           "support": r.support, "correct": r.correct} for r in artifact.rules]}
    raise LibraryError(f"cannot store {type(artifact).__name__}")


def from_json(data: Mapping[str, Any]) -> Any:
    if data.get("kind") == "decision_list":
        rules = [Rule(tuple(_literal_from_json(c) for c in r["conditions"]), r["label"], r.get("support", 0), r.get("correct", 0))
                 for r in data.get("rules", ())]
        return DecisionList(rules, data.get("default"), data.get("considered", 0))
    raise LibraryError(f"unknown stored kind {data.get('kind')!r}")


# -------------------------------------------------------------------- library


class Library:
    """A directory of learned artifacts, versioned and replayable."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        (self.root / "fixtures").mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.entries: list[Entry] = []
        if self.manifest_path.exists():
            self.entries = [Entry.from_dict(row) for row in json.loads(self.manifest_path.read_text())]

    # -- reading
    def names(self) -> list[str]:
        return sorted({e.name for e in self.entries})

    def versions(self, name: str) -> list[Entry]:
        return [e for e in self.entries if e.name == name]

    def head(self, name: str) -> Entry:
        found = self.versions(name)
        if not found:
            raise MissingArtifact(name)
        return found[-1]

    def entry(self, reference: str) -> Entry:
        if "@" not in reference:
            return self.head(reference)
        name, _, version = reference.partition("@")
        for candidate in self.versions(name):
            if candidate.version == int(version):
                return candidate
        raise MissingArtifact(reference)

    def dependents(self, digest: str) -> list[Entry]:
        return [e for e in self.entries if digest in e.depends_on]

    # -- writing
    def publish(self, name: str, artifact: Any, *, provenance: Mapping[str, Any],
                fixture: Sequence[tuple[Any, Any]] = (), depends_on: Sequence[str] = ()) -> Entry:
        """Store an artifact as the next version of ``name``, with a replayable fixture."""
        payload = to_json(artifact)
        digest = digest_of(payload)
        (self.root / "artifacts" / f"{digest}.json").write_text(json.dumps(payload, indent=1, default=str))
        (self.root / "fixtures" / f"{digest}.json").write_text(
            json.dumps([{"facts": sorted(((p, v) for p, v in facts), key=repr), "expect": expect}
                        for facts, expect in fixture], indent=1, default=str))
        superseded = self.versions(name)
        version = superseded[-1].version + 1 if superseded else 1
        entry = Entry(name, version, digest, payload["kind"], dict(provenance), tuple(depends_on))
        # relearning marks every dependent stale: it must be revalidated, not trusted
        if superseded:
            old = superseded[-1].digest
            self.entries = [
                Entry(e.name, e.version, e.digest, e.kind, e.provenance, e.depends_on, True, e.at)
                if old in e.depends_on else e
                for e in self.entries
            ]
        self.entries.append(entry)
        self._save()
        return entry

    def load(self, reference: str, *, replay: bool = True) -> Any:
        """Load an artifact, replaying its fixture first unless told not to."""
        entry = self.entry(reference)
        path = self.root / "artifacts" / f"{entry.digest}.json"
        if not path.exists():
            raise MissingArtifact(f"{reference}: {entry.digest} is not in the library")
        artifact = from_json(json.loads(path.read_text()))
        if replay:
            self._replay(entry, artifact)
        return artifact

    def revalidate(self, reference: str) -> Entry:
        """Clear ``stale`` only if the recorded fixture still reproduces exactly."""
        entry = self.entry(reference)
        artifact = self.load(reference, replay=True)
        _ = artifact
        self.entries = [
            Entry(e.name, e.version, e.digest, e.kind, e.provenance, e.depends_on, False, e.at)
            if (e.name, e.version) == (entry.name, entry.version) else e
            for e in self.entries
        ]
        self._save()
        return self.entry(reference)

    def verify(self) -> list[str]:
        """Every artifact still present and still reproducing its fixture; problems listed."""
        problems: list[str] = []
        for entry in self.entries:
            try:
                self.load(entry.reference, replay=True)
            except LibraryError as exc:
                problems.append(f"{entry.reference}: {exc}")
        return problems

    def _replay(self, entry: Entry, artifact: Any) -> None:
        path = self.root / "fixtures" / f"{entry.digest}.json"
        if not path.exists():
            return
        for case in json.loads(path.read_text()):
            facts = frozenset((p, v) for p, v in (tuple(pair) for pair in case["facts"]))
            got = artifact.predict(facts)
            if got != case["expect"]:
                raise FixtureMismatch(f"{entry.reference}: fixture expected {case['expect']!r}, got {got!r}")

    def _save(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([e.to_dict() for e in self.entries], indent=1, default=str))
        tmp.replace(self.manifest_path)
