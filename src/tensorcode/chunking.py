"""Automatization: a sequence done the same way often enough stops being deliberated.

The deliberate-to-automatic transition is the most reliable finding in skill learning, and a
procedure interpreter has none of it: a learned skill walks its steps, evaluates every guard and
records every decision, on the thousandth run exactly as on the first. That is not carefulness,
it is an inability to learn *how* it does something as opposed to *that* it works.

A chunk here is the compiled form of a sequence that has run identically :attr:`Chunks.repeats`
times: the steps actually taken, the guards whose outcome was the same every time, and nothing
else. Running a chunk skips the deliberation, not the acts — a click is still a click. What it
skips is evaluating guards whose answer has never varied and recording a decision per step.

Automatization trades adaptability for speed, so the trade is made explicit:

* the expanded form is never discarded; a chunk is an *index* into it;
* a chunk is retired the moment a step's outcome diverges from what the recorded runs saw
  (:func:`divergent`), and the walk continues expanded from that point — the fallback is the
  whole reason this is safe;
* every chunk carries how often it was used and how often it fell back, so a chunk that keeps
  breaking is visible rather than quietly wrong.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

#: bindings that mean a body step did not do what the recorded runs saw it do.
#: Deliberately *not* ``error_summary``: the assistant's own ``output_facts`` always fills that
#: in, falling back to the first line of a perfectly good output, so reading it as trouble would
#: retire every chunk on its first use. ``ok`` and ``errors`` are the marks that mean trouble.
TROUBLE = ("error", "problem", "timed_out", "unavailable")


@dataclass(frozen=True)
class Trace:
    """One run of a procedure: which steps ran, which were skipped, and how it ended."""

    procedure: str
    taken: tuple[int, ...]  # program counters whose step executed, in order
    skipped: tuple[int, ...]  # program counters whose guard did not hold
    status: str = "done"
    #: (program counter, binding that showed trouble) for the steps that showed any. Some steps
    #: are *meant* to fail — "stat the folder before creating it" expects "no such file" — so a
    #: chunk has to know what normal looked like rather than treating any error as a surprise.
    marks: tuple[tuple[int, str], ...] = ()

    @property
    def shape(self) -> str:
        """Identity of the *path* through the procedure, which is what can be compiled."""
        return hashlib.sha256(json.dumps([self.procedure, list(self.taken), sorted(self.skipped),
                                          sorted(self.marks)]).encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A compiled path: run these steps in this order, without re-deciding."""

    procedure: str
    taken: tuple[int, ...]
    skipped: frozenset[int]
    shape: str
    from_runs: int
    expected: Mapping[int, str] = field(default_factory=dict)  # pc -> the trouble every run saw there
    uses: int = 0
    fallbacks: int = 0
    retired: bool = False
    retired_because: str = ""

    def assumes_skipped(self, pc: int) -> bool:
        return pc in self.skipped

    def expects(self, pc: int) -> str:
        """The trouble mark every recorded run saw at this step ("" if they all saw none)."""
        return self.expected.get(pc, "")

    def describe(self) -> str:
        state = "retired" if self.retired else "live"
        return (f"chunk {self.procedure}/{self.shape} ({state}): {len(self.taken)} steps, "
                f"{len(self.skipped)} guards assumed, from {self.from_runs} runs, "
                f"used {self.uses}, fell back {self.fallbacks}")


def _marked(bindings: Mapping[str, object], name: str):
    """Values a step bound under ``name``, including under a step's ``as:`` prefix.

    A step that names its results (``"as": "sib"``) binds ``sib_ok`` and ``sib_errors``, so a
    check that only looked for ``ok`` would never see trouble in exactly the procedures that
    keep several commands' results apart.
    """
    for key, value in bindings.items():
        if key == name or key.endswith(f"_{name}"):
            yield key, value


def mark(bindings: Mapping[str, object]) -> str:
    """The name of the binding that says this step had trouble, or ``""`` if none did.

    Deliberately shallow: it reads only the marks the procedures already set. A chunk that skips
    guards cannot notice a subtle difference, which is exactly the known cost of automatization —
    so what is watched is trouble, and the *name* is what is compared, because the text of an
    error varies while its kind does not.
    """
    for name in TROUBLE:
        for key, value in _marked(bindings, name):
            if isinstance(value, bool) and value:
                return key
            if isinstance(value, str) and value.strip():
                return key
    for key, value in _marked(bindings, "errors"):
        if isinstance(value, (list, tuple)) and value:
            return key
    for key, value in _marked(bindings, "ok"):
        if value is False:
            return key
    return ""


def divergent(bindings: Mapping[str, object], expected: str = "") -> str:
    """Why this step differs from the runs a chunk was compiled from ("" when it does not).

    Both directions count. Trouble where there was none is the obvious case; *no* trouble where
    every recorded run had some is equally a divergence, because a guard that branched on it —
    and is now being skipped — would have gone the other way.
    """
    found = mark(bindings)
    if found == expected:
        return ""
    if found and expected:
        return f"{found} where the recorded runs had {expected}"
    if found:
        return f"{found}, which the recorded runs never had"
    return f"no {expected}, which every recorded run had"


@dataclass
class Chunks:
    """What this mind has automatized, and how well each chunk is holding up."""

    repeats: int = 3  # identical paths before a chunk is compiled
    enabled: bool = True
    runs: dict[str, list[str]] = field(default_factory=dict)  # procedure -> recent path shapes
    traces: dict[str, Trace] = field(default_factory=dict)  # shape -> the path it stands for
    compiled: dict[str, Chunk] = field(default_factory=dict)  # procedure -> live chunk
    history: list[str] = field(default_factory=list)  # compile/retire events, in order

    def record(self, trace: Trace) -> Chunk | None:
        """Note how a run went; compile a chunk once the same path has repeated enough."""
        if trace.status != "done":
            self.runs.setdefault(trace.procedure, []).clear()
            return None
        shapes = self.runs.setdefault(trace.procedure, [])
        shapes.append(trace.shape)
        self.traces[trace.shape] = trace
        recent = shapes[-self.repeats:]
        if len(recent) < self.repeats or len(set(recent)) != 1:
            return None
        live = self.compiled.get(trace.procedure)
        if live is not None and live.shape == trace.shape and not live.retired:
            return live
        chunk = Chunk(trace.procedure, trace.taken, frozenset(trace.skipped), trace.shape, self.repeats,
                      dict(trace.marks))
        self.compiled[trace.procedure] = chunk
        self.history.append(f"compiled {trace.procedure}/{trace.shape} after {self.repeats} identical runs")
        return chunk

    def chunk_for(self, procedure: str | None) -> Chunk | None:
        if not self.enabled or procedure is None:
            return None
        chunk = self.compiled.get(procedure)
        return None if chunk is None or chunk.retired else chunk

    def used(self, chunk: Chunk) -> None:
        chunk.uses += 1

    def retire(self, chunk: Chunk, why: str) -> None:
        """Abandon a chunk and fall back to deliberating. The expanded form was never lost."""
        chunk.retired, chunk.retired_because = True, why
        chunk.fallbacks += 1
        self.runs.setdefault(chunk.procedure, []).clear()
        self.history.append(f"retired {chunk.procedure}/{chunk.shape}: {why}")

    def stats(self) -> dict:
        return {
            "compiled": len([c for c in self.compiled.values() if not c.retired]),
            "retired": len([c for c in self.compiled.values() if c.retired]),
            "uses": sum(c.uses for c in self.compiled.values()),
            "fallbacks": sum(c.fallbacks for c in self.compiled.values()),
            "procedures": {p: c.describe() for p, c in self.compiled.items()},
        }


__all__ = ["Chunk", "Chunks", "Trace", "divergent", "mark", "TROUBLE"]
