"""Domain types and a rule-based note parser for the asset-status knowledge example. Supporting code."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from datetime import datetime, time

import tensacode as tc
from tensacode.backends.builtin import IN_PROCESS


class Status(enum.Enum):
    online = "online"
    offline = "offline"


@dataclass(frozen=True)
class Site:
    name: str
    timezone: str


@dataclass(frozen=True)
class Printer:
    asset_id: str
    model: str
    site: tc.Ref


# observation payloads: what a source actually produced
@dataclass(frozen=True)
class MonitorPoll:
    poller: str
    target: str
    reachable: bool
    at: datetime


@dataclass(frozen=True)
class TechNote:
    author: str
    text: str
    written_at: datetime


# parse target: what a note asserts, with where it says so
@dataclass(frozen=True)
class StatusReport:
    asset_id: str
    status: Status
    since: datetime | None
    until: datetime | None
    span: str


REGISTRY = tc.TypeRegistry()
for cls in (Status, Site, Printer, MonitorPoll, TechNote, StatusReport):
    REGISTRY.register(cls)

_ASSET = re.compile(r"\bPRN-\d+\b")
_ONLINE = re.compile(r"\b(online|working|up|printed fine|printing fine)\b", re.I)
_OFFLINE = re.compile(r"\b(offline|down|unreachable|not printing|dead)\b", re.I)
_ALL_MORNING = re.compile(r"\ball morning\b", re.I)


@tc.implementation(
    "parse",
    name="tech-note-rules",
    version="1",
    accepts=lambda r: isinstance(r.subject, TechNote) and r.target is StatusReport,
    profile=IN_PROCESS,
)
def parse_note(request: tc.Request) -> StatusReport | tc.Unknown:
    note: TechNote = request.subject
    assets = set(_ASSET.findall(note.text))
    if len(assets) != 1:
        return tc.Unknown("asset_ambiguous" if assets else "no_asset_mentioned")
    on, off = _ONLINE.search(note.text), _OFFLINE.search(note.text)
    if bool(on) == bool(off):
        return tc.Unknown("status_ambiguous" if on else "no_status_mentioned")
    match = on or off
    since = datetime.combine(note.written_at.date(), time(8, 0), note.written_at.tzinfo) if _ALL_MORNING.search(note.text) else note.written_at
    return StatusReport(assets.pop(), Status.online if on else Status.offline, since, note.written_at, f"text[{match.start()}:{match.end()}]")
