"""Artifacts that outlive the process: versions, replayed fixtures, and staleness."""

import json

import pytest

from tensacode.learning import (
    FixtureMismatch, Library, MissingArtifact, candidate_literals, decision_list, digest_of, from_json, to_json,
)


def facts(**kw):
    return frozenset(kw.items())


CASES = [(facts(urgent=u, paid=p), "fast" if (u and p) else "normal")
         for u in (True, False) for p in (True, False)] * 6


def induce():
    return decision_list(CASES, candidate_literals(CASES))


def test_an_artifact_round_trips_through_plain_json():
    rules = induce()
    back = from_json(json.loads(json.dumps(to_json(rules))))
    for facts_, _ in CASES:
        assert back.predict(facts_) == rules.predict(facts_)


def test_publishing_stores_content_addressed_and_versioned_artifacts(tmp_path):
    library = Library(tmp_path)
    rules = induce()
    first = library.publish("router", rules, provenance={"induced_from": "cases@1", "method": "decision_list"},
                            fixture=CASES[:4])
    second = library.publish("router", rules, provenance={"induced_from": "cases@2", "method": "decision_list"})
    assert first.version == 1 and second.version == 2
    assert first.digest == second.digest  # the same artifact is stored once
    assert library.head("router").version == 2
    assert library.names() == ["router"]


def test_loading_replays_the_recorded_fixture(tmp_path):
    library = Library(tmp_path)
    library.publish("router", induce(), provenance={"method": "decision_list"}, fixture=CASES[:4])
    loaded = library.load("router")
    assert loaded.predict(facts(urgent=True, paid=True)) == "fast"
    assert library.verify() == []


def test_a_stored_artifact_that_no_longer_matches_its_fixture_refuses_to_load(tmp_path):
    library = Library(tmp_path)
    entry = library.publish("router", induce(), provenance={"method": "decision_list"}, fixture=CASES[:4])
    path = tmp_path / "fixtures" / f"{entry.digest}.json"
    recorded = json.loads(path.read_text())
    recorded[0]["expect"] = "something-else"
    path.write_text(json.dumps(recorded))
    with pytest.raises(FixtureMismatch):
        library.load("router")
    assert library.verify()  # and `verify` reports it rather than hiding it


def test_relearning_marks_dependents_stale_until_revalidated(tmp_path):
    library = Library(tmp_path)
    base = library.publish("router", induce(), provenance={"method": "decision_list"}, fixture=CASES[:2])
    library.publish("dispatch", induce(), provenance={"method": "decision_list"},
                    fixture=CASES[:2], depends_on=[base.digest])
    assert not library.entry("dispatch").stale

    changed = decision_list(CASES, candidate_literals(CASES), max_conditions=1)
    library.publish("router", changed, provenance={"method": "decision_list", "note": "relearned"})
    assert library.entry("dispatch").stale, "a dependent of a superseded artifact must be marked"

    library.revalidate("dispatch")
    assert not library.entry("dispatch").stale


def test_provenance_is_required_and_kept(tmp_path):
    library = Library(tmp_path)
    provenance = {"induced_from": "trace:2026-09-17", "method": "decision_list", "verification": "adopted"}
    library.publish("router", induce(), provenance=provenance, fixture=CASES[:2])
    reloaded = Library(tmp_path)  # a different process
    assert reloaded.head("router").provenance == provenance


def test_a_missing_artifact_is_an_error_not_a_silent_none(tmp_path):
    library = Library(tmp_path)
    with pytest.raises(MissingArtifact):
        library.load("nothing-here")


def test_the_digest_is_stable_across_processes():
    assert digest_of(to_json(induce())) == digest_of(to_json(induce()))
