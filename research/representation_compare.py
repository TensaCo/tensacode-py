"""Legacy TCIR vs proposed records on the same object graph.

    python eval/representation_compare.py --legacy-python PY_WITH_PYDANTIC_2_5 --sandbox DIR

The legacy half runs in a sandbox built by legacy_probe/make_sandbox.py (import-only shims).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "legacy_probe")]

import tensacode as tc  # noqa: E402
from fixtures import CyclicPrinter, CyclicSite, Printer, Site, SiteWithName, Ticket, cyclic_site, tickets  # noqa: E402
from research.object_graph import from_records, to_records  # noqa: E402
from tensacode.records import EncodeError, encode  # noqa: E402


def attempt(out: dict, label: str, fn) -> None:
    try:
        out[label] = {"ok": True, "value": fn()}
    except Exception as e:  # noqa: BLE001
        out[label] = {"ok": False, "error": f"{type(e).__name__}: {e}"}


def proposed() -> dict:
    out: dict = {}
    reg = tc.TypeRegistry()
    reg.register(Ticket, identity="id")
    reg.register(Printer, identity="asset_id")
    reg.register(Site)
    t1, t2 = tickets()

    one = to_records([t1], reg)
    both = to_records([t1, t2], reg)
    attempt(out, "records_one_ticket", lambda: len(one.records))
    attempt(out, "records_two_tickets_shared_printer", lambda: len(both.records))
    attempt(out, "json_bytes_one_ticket", lambda: len(json.dumps({r.id: v for r, v in one.records.items()})))
    attempt(out, "json_two_tickets", lambda: {r.id: v for r, v in both.records.items()})
    attempt(out, "shared_printer_is_one_record", lambda: sum(r.kind == "Printer" for r in both.records) == 1)
    rebuilt, report = from_records(both, reg)
    attempt(out, "reconstruct_python_value_type", lambda: type(rebuilt[0]).__name__)
    attempt(out, "reconstruct_equals_original", lambda: rebuilt[0] == t1 and rebuilt[1] == t2)
    attempt(out, "reconstruct_preserves_shared_identity", lambda: rebuilt[0].printer is rebuilt[1].printer)
    attempt(out, "conversion_lossless", lambda: both.report.lossless and report.lossless)

    reg_name = tc.TypeRegistry()
    reg_name.register(SiteWithName)
    attempt(out, "field_named_name", lambda: encode(SiteWithName("HQ", "America/Chicago"), reg_name).data)

    reg_cycle = tc.TypeRegistry()
    reg_cycle.register(CyclicSite, identity="label")
    reg_cycle.register(CyclicPrinter, identity="asset_id")
    conv = to_records([cyclic_site()], reg_cycle)

    def cycle_roundtrip():
        (site,), rep = from_records(conv, reg_cycle)
        return {"records": len(conv.records), "cycle_restored": site.printers[0].site is site, "lossless": rep.lossless}

    attempt(out, "cycle_through_entities", cycle_roundtrip)
    reg_values = tc.TypeRegistry()
    reg_values.register(CyclicSite)
    reg_values.register(CyclicPrinter)

    def cycle_values():
        try:
            encode(cyclic_site(), reg_values)
        except EncodeError as e:
            return f"refused: {e}"
        return "encoded (unexpected)"

    attempt(out, "cycle_through_values", cycle_values)

    try:
        from pydantic import BaseModel

        class PrinterModel(BaseModel):
            asset_id: str
            floor: int

        reg_p = tc.TypeRegistry()
        reg_p.register(PrinterModel)

        def pyd():
            data = encode(PrinterModel(asset_id="PRN-3", floor=2), reg_p).data
            value, rep = tc.records.decode(json.loads(json.dumps(data)), reg_p)
            return {"data": data, "round_trip_equal": value == PrinterModel(asset_id="PRN-3", floor=2)}

        attempt(out, "pydantic_model", pyd)
    except ImportError:
        pass

    def unknown_type():
        value, rep = tc.records.decode({"$type": "os.system", "fields": {"command": "rm -rf /"}}, tc.TypeRegistry())
        return {"value": repr(value), "opaque": rep.opaque}

    attempt(out, "unknown_type_name", unknown_type)

    def update():
        store = tc.Store(reg)
        ref = tc.Ref("Printer:PRN-3")
        _, t = tickets()
        store.put(ref, t.printer)
        before = store.get(ref)
        patch = tc.Patch((tc.SetField(ref, ("floor",), 3),), store.revision, "moved upstairs")
        commit = store.apply(patch)
        return {"revision": commit.revision, "floor_now": store.get(ref).floor, "previous_value_unchanged": before.floor == 2}

    attempt(out, "update_nested_field", update)

    def aliasing():
        shared = ["hardware"]
        a = dataclasses.replace(tickets()[0], tags=shared)
        b = dataclasses.replace(tickets()[1], tags=shared, printer=a.printer)
        return to_records([a, b], reg).report.losses

    attempt(out, "aliasing_of_value_reported", aliasing)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-python", required=True)
    ap.add_argument("--sandbox", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/representation.json")
    args = ap.parse_args()
    legacy = json.loads(subprocess.run([args.legacy_python, "-W", "ignore", str(ROOT / "research/legacy_probe/probe_legacy.py"), str(args.sandbox)], capture_output=True, text=True, check=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}).stdout)  # legacy tensacode is a namespace package; keep the new one off the path
    new = proposed()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"legacy_tcir": legacy, "proposed_records": new}, indent=1, default=str))
    for key in sorted(set(legacy) | set(new)):
        def show(d):
            v = d.get(key)
            if v is None:
                return "-"
            text = json.dumps(v["value"], default=str) if v["ok"] else "ERROR " + v["error"]
            return text if len(text) < 70 else text[:67] + "..."
        print(f"{key:<40} | {show(legacy):<70} | {show(new)}")


if __name__ == "__main__":
    main()
