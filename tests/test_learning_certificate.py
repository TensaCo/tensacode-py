"""Read-set certificates: what an answer rested on, and whether it still holds."""

import pytest

from tensorcode.learning import (
    MISSING, ReadSet, Reader, candidate_literals, certified, decision_list, revalidate, value_digest,
)


def facts(**kw):
    return dict(kw)


CASES = [(frozenset({("urgent", u), ("paid", p)}), "fast" if (u and p) else "queued" if u else "normal")
         for u in (True, False) for p in (True, False)] * 8


def test_a_reader_records_what_was_consulted():
    reader = Reader(facts(urgent=True, paid=False, region="north"))
    reader.get("urgent")
    reader.get("paid")
    certificate = reader.readset()
    assert certificate.keys == ("paid", "urgent")  # "region" was never read
    assert "region" not in certificate.keys


def test_a_miss_is_a_read_so_an_addition_invalidates():
    reader = Reader(facts(urgent=True))
    reader.get("paid")  # absent
    certificate = reader.readset()
    assert certificate.misses == ("paid",)
    assert revalidate(certificate, facts(urgent=True)).holds
    # the key appears: the answer rested on its absence, so it no longer holds
    verdict = revalidate(certificate, facts(urgent=True, paid=True))
    assert verdict.status == "fails" and any("paid" in r for r in verdict.reasons)


def test_revalidation_holds_when_nothing_read_has_changed():
    reader = Reader(facts(a=1, b=2, c=3))
    reader.get("a")
    certificate = reader.readset()
    assert revalidate(certificate, facts(a=1, b=2, c=3)).holds
    assert revalidate(certificate, facts(a=1, b=99, c=3)).holds  # b was never read
    assert revalidate(certificate, facts(a=2, b=2, c=3)).status == "fails"


def test_a_decision_carries_the_certificate_of_its_own_reads():
    rules = decision_list(CASES, candidate_literals(CASES))
    label, rule, certificate = rules.decide({"urgent": True, "paid": True})
    assert label == "fast"
    assert certificate.reads, "the decision read nothing"
    # changing a fact the decision read invalidates it; an unread one does not
    assert revalidate(certificate, {"urgent": True, "paid": True}).holds
    assert revalidate(certificate, {"urgent": True, "paid": True, "unrelated": 1}).holds
    assert revalidate(certificate, {"urgent": False, "paid": True}).status == "fails"
    _ = rule


def test_certified_runs_a_computation_and_returns_its_certificate():
    def count_files(reader):
        return len(reader.get("files", ()))

    answer, certificate = certified(count_files, {"files": ["a", "b"], "other": 1}, note="count")
    assert answer == 2 and certificate.keys == ("files",) and certificate.note == "count"


def test_digests_are_stable_and_order_independent():
    assert value_digest({"a": 1, "b": 2}) == value_digest({"b": 2, "a": 1})
    assert value_digest(frozenset({1, 2})) == value_digest(frozenset({2, 1}))
    assert value_digest(None) != MISSING
    assert value_digest([1, 2]) != value_digest([2, 1])  # a list keeps its order


def test_a_certificate_round_trips_through_json():
    reader = Reader(facts(a=1))
    reader.get("a")
    reader.get("missing")
    before = reader.readset()
    after = ReadSet.from_dict(before.to_dict())
    assert after.reads == before.reads and after.digest() == before.digest()


@pytest.mark.parametrize("before,after", [({}, {"x": None}), ({"x": None}, {})])
def test_absence_and_explicit_none_are_different_dependencies(before, after):
    answer, certificate = certified(lambda reader: "x" in reader, before)
    assert answer == ("x" in before)
    assert certificate.misses == (() if "x" in before else ("x",))
    assert revalidate(certificate, before).holds
    assert revalidate(certificate, after).status == "fails"


@pytest.mark.parametrize("before,after", [
    ({}, {"new": None}),
    ({"a": 1}, {"a": 1, "new": 2}),
    ({"a": 1, "b": 2}, {"a": 1}),
    ({"a": 1}, {"a": 2}),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
])
def test_full_scan_certificate_invalidates_when_scan_changes(before, after):
    answer, certificate = certified(lambda reader: tuple(reader.items()), before)
    assert answer == tuple(before.items())
    assert revalidate(certificate, before).holds
    # Exercise persistence, especially the difference between an empty scan and no scan.
    restored = ReadSet.from_dict(certificate.to_dict())
    assert restored.scanned_keys == tuple(before)
    assert restored.digest() == certificate.digest()
    assert revalidate(restored, after).status == "fails"


def test_point_read_does_not_depend_on_unrelated_additions():
    _, certificate = certified(lambda reader: reader.get("a"), {"a": None})
    assert certificate.scanned_keys is None
    assert revalidate(certificate, {"a": None, "new": 2}).holds


def test_empty_scan_and_no_reads_have_different_certificate_digests():
    _, scan = certified(lambda reader: tuple(reader.items()), {})
    _, unread = certified(lambda reader: 42, {})
    assert scan.digest() != unread.digest()


@pytest.mark.parametrize("version", [None, 0, -1, 2, True, "1"])
def test_legacy_and_unsupported_serialized_certificates_are_rejected(version):
    # An old full scan of {"a": 1} would incorrectly authorize an answer after
    # adding a new key if its missing coverage metadata meant "point reads".
    legacy = {"reads": [["a", value_digest(1)]], "at": 0, "note": "old scan"}
    if version is not None:
        legacy["format_version"] = version
    with pytest.raises(ValueError, match="regenerate"):
        ReadSet.from_dict(legacy)


@pytest.mark.parametrize("missing", ["reads", "scanned_keys"])
def test_current_format_must_include_explicit_dependency_coverage(missing):
    _, certificate = certified(lambda reader: tuple(reader.items()), {"a": 1})
    serialized = certificate.to_dict()
    assert serialized["format_version"] == 1
    del serialized[missing]
    with pytest.raises(ValueError, match="dependency coverage"):
        ReadSet.from_dict(serialized)
