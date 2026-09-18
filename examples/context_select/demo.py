"""Select context for an on-call question about a printer.

    python -m examples.context_select.demo
"""

from __future__ import annotations

from datetime import datetime, timezone

import tensorcode as tc
from tensorcode.backends.builtin import BM25Ranker

from ..knowledge.domain import REGISTRY, MonitorPoll, Printer, TechNote, parse_note
from ..knowledge.program import observe_note, observe_poll
from .program import Snippet, select_context


def at(h: int, m: int) -> datetime:
    return datetime(2026, 9, 16, h, m, tzinfo=timezone.utc)


DOCS = [
    Snippet("tkt-881", "Floor 2 users report PRN-3 print jobs stuck in the queue since about 10am.", tc.Ref("doc:tkt-881")),
    Snippet("tkt-881-fwd", "FW: Floor 2 users report PRN-3 print jobs stuck in the queue since about 10am.", tc.Ref("doc:tkt-881-fwd")),
    Snippet("kb-12", "If a LaserJet M507 drops off the network, power-cycle it and confirm the DHCP lease on VLAN 20.", tc.Ref("doc:kb-12")),
    Snippet("kb-40", "Toner replacement procedure for LaserJet M507: open the front door and pull the cartridge straight out.", tc.Ref("doc:kb-40")),
    Snippet("chg-311", "Change 311: network switch sw-hq-2b firmware upgrade scheduled 09:55-10:10 on floor 2.", tc.Ref("doc:chg-311")),
    Snippet("lunch", "The floor 2 kitchen will be closed for cleaning on Friday.", tc.Ref("doc:memo-7")),
]


def main() -> None:
    world = tc.Store(REGISTRY)
    world.declare("status", functional=True)
    prn3 = tc.Ref("printer:PRN-3")
    world.put(prn3, Printer("PRN-3", "LaserJet M507", tc.Ref("site:hq")))
    runtime = tc.Runtime([parse_note, BM25Ranker(text_of=lambda s: s.text)])
    with tc.use(runtime):
        observe_poll(world, MonitorPoll("snmp-poller-2", "PRN-3", reachable=False, at=at(10, 2)))
        observe_note(world, "n-17", TechNote("sam", "PRN-3 has been online all morning, printed fine for the 9am batch.", at(10, 4)))
        question = "Why are PRN-3 print jobs stuck on floor 2 this morning?"
        for budget in (100, 60, 20):
            print(f"== budget {budget} approx tokens")
            result = select_context(question, prn3, world, DOCS, budget=budget)
            if isinstance(result, tc.Unknown):
                print(f"Unknown: {result.reason} ({result.detail})\n")
                continue
            print(f"used {result.used}/{result.budget}")
            for s in result.items:
                print(f"  + [{tc.approx_tokens(s.text):>2}] {s.text}")
            for s, why in result.dropped:
                print(f"  - {s.id}: {why}")
            print()
    print("== trace")
    print(runtime.trace.render())


if __name__ == "__main__":
    main()
