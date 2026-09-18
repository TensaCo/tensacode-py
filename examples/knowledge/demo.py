"""Two conflicting observations about one printer, handled structurally.

    python -m examples.knowledge.demo
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import tensorcode as tc
from tensorcode.backends.builtin import StoreFactCheck

from .domain import REGISTRY, MonitorPoll, Printer, Site, Status, TechNote, parse_note
from .program import observe_note, observe_poll, status_at


def at(h: int, m: int) -> datetime:
    return datetime(2026, 9, 16, h, m, tzinfo=timezone.utc)


def show(label: str, value: object) -> None:
    print(f"{label:<44} {value}")


def main() -> None:
    world = tc.Store(REGISTRY)
    world.declare("status", functional=True)
    world.declare("located_on", functional=True)
    runtime = tc.Runtime([parse_note, StoreFactCheck(world)])

    hq, prn3, prn4, floor2 = tc.Ref("site:hq"), tc.Ref("printer:PRN-3"), tc.Ref("printer:PRN-4"), tc.Ref("floor:hq-2")
    world.put(hq, Site("HQ", "America/Chicago"))
    inventory = world.put(tc.Ref("obs:asset-db-export-0901"), {"exported": "2026-09-01"})
    for ref, model in ((prn3, "LaserJet M507"), (prn4, "LaserJet M507")):
        world.put(ref, Printer(ref.id.split(":")[1], model, hq))
        world.tell(tc.Claim(ref, "located_on", floor2, tc.Interval(at(0, 0).replace(day=1), None)), tc.Evidence(inventory, at(0, 0).replace(day=1), method="asset-db"))

    with tc.use(runtime):
        print("== ingest")
        observe_poll(world, MonitorPoll("snmp-poller-2", "PRN-3", reachable=False, at=at(10, 2)))
        observe_poll(world, MonitorPoll("snmp-poller-2", "PRN-4", reachable=True, at=at(10, 2)))
        note = observe_note(world, "n-17", TechNote("sam", "PRN-3 has been online all morning, printed fine for the 9am batch.", at(10, 4)))
        vague = observe_note(world, "n-18", TechNote("sam", "One of the printers upstairs seems slow.", at(10, 5)))
        show("note n-17 ->", note)
        show("note n-18 (no asset) ->", vague)

        print("\n== contradictions (functional predicate, overlapping validity, same scope)")
        for c in world.conflicts():
            for rec in (c.a, c.b):
                ev = rec.evidence[0]
                show(f"  {rec.claim.object.value} during {rec.claim.valid.start:%H:%M}-{rec.claim.valid.end:%H:%M}", f"source={ev.source} method={ev.method} locator={ev.locator}")
            show("  overlap", f"{c.during.start:%H:%M}-{c.during.end:%H:%M}")

        print("\n== queries")
        show("status_at(PRN-3, 09:00)", status_at(world, prn3, at(9, 0)))
        show("status_at(PRN-3, 10:02)", status_at(world, prn3, at(10, 2)))
        show("status_at(PRN-3, 10:30)", status_at(world, prn3, at(10, 30)))
        offline_on_2 = world.match((tc.Var("p"), "located_on", floor2), (tc.Var("p"), "status", Status.offline), at=at(10, 2))
        show("printers on floor 2 with an offline claim @10:02", [b["p"].id for b in offline_on_2])
        sub = world.neighborhood(prn3, depth=1)
        show("neighborhood(PRN-3): entities", [r.id for r in sub.entities])
        show("neighborhood(PRN-3): claims", [(c.claim.predicate, getattr(c.claim.object, "value", str(c.claim.object))) for c in sub.claims])

        print("\n== checks (unknown is not false)")
        for status in (Status.online, Status.offline):
            show(f"check PRN-3 {status.value} @10:02", tc.check(tc.Claim(prn3, "status", status, tc.Interval.at(at(10, 2)))).status)
        show("check PRN-3 online @09:00", tc.check(tc.Claim(prn3, "status", Status.online, tc.Interval.at(at(9, 0)))).status)
        show("check PRN-3 offline @09:00", tc.check(tc.Claim(prn3, "status", Status.offline, tc.Interval.at(at(9, 0)))).status)
        show("check PRN-4 offline @10:02", tc.check(tc.Claim(prn4, "status", Status.offline, tc.Interval.at(at(10, 2)))).status)

        print("\n== later evidence about a different time does not resolve the earlier conflict")
        observe_poll(world, MonitorPoll("snmp-poller-2", "PRN-3", reachable=True, at=at(10, 6)))
        show("status_at(PRN-3, 10:06)", status_at(world, prn3, at(10, 6)))
        show("status_at(PRN-3, 10:02)", status_at(world, prn3, at(10, 2)).reason)

        print("\n== the author retracts; the change is proposed, then committed")
        note_claim = next(r for r in world.claims(prn3, "status") if r.evidence[0].source == tc.Ref("obs:note-n-17"))
        correction = world.put(tc.Ref("obs:note-n-19"), TechNote("sam", "Correction: I was thinking of PRN-4.", at(10, 20)))
        patch = tc.Patch((tc.Retract(note_claim.id, "author retracted", (tc.Evidence(correction, at(10, 20)),)),), world.revision)
        show("proposed patch (not yet applied): conflicts", len(world.conflicts()))
        commit = world.apply(patch)
        show(f"after commit r{commit.revision}: conflicts", len(world.conflicts()))
        show("status_at(PRN-3, 10:02)", status_at(world, prn3, at(10, 2)))
        show("status_at(PRN-3, 09:00) (retraction removed the only claim)", status_at(world, prn3, at(9, 0)).reason)
        show("retracted claims kept for audit", len(world.claims(prn3, "status", include_retracted=True)) - len(world.claims(prn3, "status")))
        try:
            world.apply(patch)
        except tc.records.StaleRevision as exc:
            show("re-applying the same patch", f"StaleRevision: {exc}")

    print("\n== persistence")
    data = json.dumps(world.to_json())
    restored, report = tc.Store.from_json(json.loads(data), REGISTRY)
    show("records", f"{len(world.entities)} entities, {len(world.claims(include_retracted=True))} claims, {len(data)} bytes JSON")
    show("round trip equal (entities, claims, lossless)", (restored.entities == world.entities, [c.id for c in restored.claims(include_retracted=True)] == [c.id for c in world.claims(include_retracted=True)], report.lossless))
    stranger, report = tc.Store.from_json(json.loads(data), tc.TypeRegistry())
    show("load with an empty registry", f"opaque values: {len(report.opaque)}; example: {stranger.get(prn3)}")

    print("\n== trace")
    print(runtime.trace.render())


if __name__ == "__main__":
    main()
