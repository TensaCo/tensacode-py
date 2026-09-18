"""The knowledge program. Runs against ``tensorcode`` (the prototype).

Sources assert; the store records who asserted what, about when. Contradictions
are detected structurally and preserved. Nothing here asks a model which source
is right: a parser may *extract* a claim from text, but adjudicating unsupported
facts is not an inference any backend is entitled to make.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import tensorcode as tc

from .domain import MonitorPoll, Status, StatusReport, TechNote


def observe_poll(world: tc.Store, poll: MonitorPoll) -> tc.Ref:
    source = world.put(tc.Ref(f"obs:poll-{poll.target}-{poll.at:%H%M}"), poll)
    status = Status.online if poll.reachable else Status.offline
    world.tell(tc.Claim(tc.Ref(f"printer:{poll.target}"), "status", status, tc.Interval.at(poll.at)), tc.Evidence(source, poll.at, method=poll.poller))
    return source


def observe_note(world: tc.Store, note_id: str, note: TechNote) -> tc.Ref | tc.Unknown:
    source = world.put(tc.Ref(f"obs:note-{note_id}"), note)
    report = tc.parse(note, StatusReport)  # extraction, not adjudication
    if isinstance(report, tc.Unknown):
        return report  # the note stays on record as an observation with no claim
    claim = tc.Claim(tc.Ref(f"printer:{report.asset_id}"), "status", report.status, tc.Interval(report.since, report.until))
    world.tell(claim, tc.Evidence(source, note.written_at, locator=report.span, method="parse"))
    return source


def status_at(world: tc.Store, printer: tc.Ref, t: datetime) -> Status | tc.Unknown:
    live = world.claims(printer, "status", at=t)
    sources = Counter(rec.claim.object for rec in live for _ in rec.evidence)
    if len(sources) == 1:
        return next(iter(sources))
    if not sources:
        return tc.Unknown("no_evidence", f"no status claim covers {t:%H:%M}")
    total = sum(sources.values())
    return tc.Unknown("contested", f"{len(live)} incompatible claims", tuple((s, tc.Score(n / total, "vote_share")) for s, n in sources.items()))
