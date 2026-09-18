import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

import tensorcode as tc
from tensorcode.records import EncodeError, StaleRevision, decode, encode


def at(h, m=0):
    return datetime(2026, 9, 16, h, m, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Device:
    serial: str
    room: tc.Ref


@dataclass
class Node:
    name: str
    children: list = field(default_factory=list)
    parent: object = None


def world():
    w = tc.Store()
    w.declare("status", functional=True)
    return w


def ev(src, t):
    return tc.Evidence(tc.Ref(f"obs:{src}"), t)


def test_ref_requires_kind():
    with pytest.raises(ValueError):
        tc.Ref("no-kind")


def test_same_proposition_from_two_sources_is_one_claim_with_two_evidence():
    w = world()
    c = tc.Claim(tc.Ref("dev:1"), "status", "up", tc.Interval.at(at(9)))
    w.tell(c, ev("a", at(9)))
    rec = w.tell(tc.Claim(tc.Ref("dev:1"), "status", "up", tc.Interval.at(at(9))), ev("b", at(9)))
    assert len(w.claims()) == 1 and len(rec.evidence) == 2


def test_claims_require_evidence():
    with pytest.raises(ValueError):
        world().tell(tc.Claim(tc.Ref("dev:1"), "status", "up"))


def test_conflicts_need_functional_predicate_overlap_and_same_scope():
    w = world()
    d = tc.Ref("dev:1")
    w.tell(tc.Claim(d, "status", "up", tc.Interval(at(8), at(10))), ev("a", at(10)))
    w.tell(tc.Claim(d, "status", "down", tc.Interval.at(at(11))), ev("b", at(11)))  # no overlap
    w.tell(tc.Claim(d, "status", "down", tc.Interval.at(at(9)), scope=tc.Ref("hyp:h1")), ev("c", at(9)))  # other scope
    w.tell(tc.Claim(d, "tag", "x"), ev("d", at(9)))
    w.tell(tc.Claim(d, "tag", "y"), ev("e", at(9)))  # tag not functional
    assert w.conflicts() == []
    w.tell(tc.Claim(d, "status", "down", tc.Interval.at(at(9))), ev("f", at(9)))
    (conflict,) = w.conflicts()
    assert conflict.during == tc.Interval.at(at(9))


def test_temporal_query_and_retraction_keeps_history():
    w = world()
    d = tc.Ref("dev:1")
    rec = w.tell(tc.Claim(d, "status", "up", tc.Interval(at(8), at(10))), ev("a", at(10)))
    assert [r.claim.object for r in w.claims(d, "status", at=at(9))] == ["up"]
    assert w.claims(d, "status", at=at(11)) == []
    w.apply(tc.Patch((tc.Retract(rec.id, "wrong device"),), w.revision))
    assert w.claims(d) == [] and len(w.claims(d, include_retracted=True)) == 1


def test_match_joins_patterns():
    w = world()
    for dev, room, status in (("1", "a", "down"), ("2", "a", "up"), ("3", "b", "down")):
        w.tell(tc.Claim(tc.Ref(f"dev:{dev}"), "in", tc.Ref(f"room:{room}")), ev(dev, at(9)))
        w.tell(tc.Claim(tc.Ref(f"dev:{dev}"), "status", status), ev(dev, at(9)))
    rows = w.match((tc.Var("d"), "in", tc.Ref("room:a")), (tc.Var("d"), "status", "down"))
    assert rows == [{"d": tc.Ref("dev:1")}]


def test_patch_is_inert_atomic_and_revision_checked():
    reg = tc.TypeRegistry()
    reg.register(Device)
    w = tc.Store(reg)
    ref = w.put(tc.Ref("dev:1"), Device("S1", tc.Ref("room:a")))
    patch = tc.Patch((tc.SetField(ref, ("room",), tc.Ref("room:b")), tc.SetField(tc.Ref("dev:missing"), ("room",), None)), w.revision)
    assert w.get(ref).room == tc.Ref("room:a")  # proposing changed nothing
    with pytest.raises(KeyError):
        w.apply(patch)
    assert w.get(ref).room == tc.Ref("room:a")  # all-or-nothing
    good = tc.Patch((tc.SetField(ref, ("room",), tc.Ref("room:b")),), w.revision)
    w.apply(good)
    with pytest.raises(StaleRevision):
        w.apply(good)


def test_unregistered_types_are_refused_on_encode_and_opaque_on_decode():
    with pytest.raises(EncodeError):
        encode(Device("S1", tc.Ref("room:a")), tc.TypeRegistry())
    value, report = decode({"$type": "subprocess.Popen", "fields": {"args": ["rm", "-rf", "/"]}}, tc.TypeRegistry())
    assert isinstance(value, tc.Opaque) and report.opaque


def test_store_json_round_trip():
    reg = tc.TypeRegistry()
    reg.register(Device)
    w = tc.Store(reg)
    w.declare("status", functional=True)
    w.put(tc.Ref("dev:1"), Device("S1", tc.Ref("room:a")))
    w.tell(tc.Claim(tc.Ref("dev:1"), "status", "up", tc.Interval(at(8), None)), tc.Evidence(tc.Ref("obs:x"), at(8), confidence=tc.Score(0.7, "uncalibrated")))
    restored, report = tc.Store.from_json(json.loads(json.dumps(w.to_json())), reg)
    assert report.lossless
    assert restored.entities == w.entities
    assert [(c.claim, c.evidence) for c in restored.claims()] == [(c.claim, c.evidence) for c in w.claims()]


def test_cycles_through_values_are_refused():
    reg = tc.TypeRegistry()
    reg.register(Node)  # a value type: no identity
    root = Node("root")
    root.children.append(Node("child", parent=root))
    with pytest.raises(EncodeError, match="cycle"):
        encode(root, reg)


def test_claims_filters_every_given_field_whichever_index_is_used():
    w = world()
    for i in range(5):
        w.tell(tc.Claim(tc.Ref(f"dev:{i}"), "status", "up"), ev(str(i), at(9)))
    w.tell(tc.Claim(tc.Ref("dev:0"), "room", "a"), ev("r", at(9)))
    w.tell(tc.Claim(tc.Ref("dev:0"), "owner", "b"), ev("o", at(9)))
    assert [r.claim.subject.id for r in w.claims(tc.Ref("dev:3"), "status")] == ["dev:3"]
    assert [r.claim.predicate for r in w.claims(tc.Ref("dev:0"), "room")] == ["room"]
