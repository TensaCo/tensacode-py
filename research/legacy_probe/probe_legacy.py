"""Run inside a sandbox made by make_sandbox.py:  python probe_legacy.py SANDBOX_DIR  -> JSON on stdout."""

import json
import sys
from pathlib import Path

sys.path[:0] = [sys.argv[1], str(Path(__file__).parent)]
sys.setrecursionlimit(3000)

from fixtures import SiteWithName, cyclic_site, tickets  # noqa: E402
from tensorcode.internal.tcir.graph_merging import merge_identical  # noqa: E402
from tensorcode.internal.tcir.nodes import CompositeValueNode, Node  # noqa: E402
from tensorcode.internal.tcir.parse import parse_node  # noqa: E402

out = {}


def attempt(label, fn):
    try:
        out[label] = {"ok": True, "value": fn()}
    except BaseException as e:  # noqa: BLE001
        out[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}


def count(n, seen=None):
    seen = set() if seen is None else seen
    if id(n) in seen:
        return 0
    seen.add(id(n))
    c, vals = 1, [getattr(n, k) for k in list(type(n).model_fields) + list((n.model_extra or {}).keys())]
    while vals:
        v = vals.pop()
        if isinstance(v, Node):
            c += count(v, seen)
        elif isinstance(v, dict):
            vals.extend(v.keys())
            vals.extend(v.values())
        elif isinstance(v, (list, tuple)):
            vals.extend(v)
    return c


t1, t2 = tickets()
one = parse_node(t1)
both = parse_node([t1, t2])
attempt("records_one_ticket", lambda: count(one))
attempt("records_two_tickets_shared_printer", lambda: count(both))
attempt("json_bytes_one_ticket", lambda: len(one.model_dump_json()))
attempt("json_two_tickets", lambda: both.model_dump_json())
attempt("json_one_ticket", lambda: json.loads(one.model_dump_json()))
attempt("shared_printer_is_one_record", lambda: both.items[0].printer is both.items[1].printer)
attempt("reconstruct_python_value_type", lambda: type(one.python_value).__name__)
attempt("reconstruct_equals_original", lambda: one.python_value == t1)
attempt("json_round_trip_via_Node", lambda: type(Node.model_validate_json(one.model_dump_json())).__name__)
attempt("json_round_trip_nested_type", lambda: type(CompositeValueNode.model_validate_json(one.model_dump_json()).printer).__name__)
attempt("merge_identical", lambda: merge_identical(both))
attempt("field_named_name", lambda: type(parse_node(SiteWithName("HQ", "America/Chicago"))).__name__)
attempt("cycle", lambda: type(parse_node(cyclic_site())).__name__)


def update_floor():
    node = parse_node(t1)
    node.printer.floor.value = 3  # nested pydantic node mutation; no patch, no revision, no validation of meaning
    return node.python_value["printer"]["floor"]


attempt("update_nested_field_python_value", update_floor)
try:
    from pydantic import BaseModel

    class PrinterModel(BaseModel):
        asset_id: str
        floor: int

    attempt("pydantic_model", lambda: type(parse_node(PrinterModel(asset_id="PRN-3", floor=2))).__name__)
except ImportError:
    pass
print(json.dumps(out, indent=1, default=str))
