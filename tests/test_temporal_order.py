"""Time: tense as an interval, events as things that can be ordered."""

from datetime import datetime, timedelta, timezone

from tensacode.outcomes import Unknown
from tensacode.records import Claim, Evidence, Ref, Store
from tensacode.temporal import (after, before, changed_since, connective_relation, during, event_time, events,
                                interval_for, order, ordered_by_claims, relate, since, tell_event, tell_order)

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
SOURCE = Ref("obs:session")


def test_tense_becomes_an_interval_a_store_can_use():
    assert interval_for({"tense": "past"}, NOW).end == NOW
    assert interval_for({"tense": "future"}, NOW).start == NOW
    present = interval_for({"tense": "present"}, NOW)
    assert present.contains(NOW)
    assert interval_for({}, NOW) == interval_for({"mood": "declarative"}, NOW)  # no tense commits to nothing


def test_an_event_carries_its_own_clock_not_the_reporters():
    mind = Store()
    happened = NOW - timedelta(hours=5)
    tell_event(mind, Ref("event:commit"), at=happened, kind="commit", source=SOURCE)
    assert event_time(mind, Ref("event:commit")) == happened
    # the evidence's observed_at is when it was recorded, which is later and kept separate
    recorded = min(e.observed_at for e in mind.claims(Ref("event:commit"), "happened_at")[0].evidence)
    assert recorded > happened


def test_events_without_a_time_are_refused_not_guessed():
    mind = Store()
    mind.tell(Claim(Ref("event:vague"), "is_a", "commit"), Evidence(SOURCE, NOW, method="test"))
    got = event_time(mind, Ref("event:vague"))
    assert isinstance(got, Unknown) and got.reason == "no_event_time"
    assert before(mind, Ref("event:vague")) == []


def test_ordering_and_the_queries_built_on_it():
    mind = Store()
    for name, minutes in (("wrote", -30), ("committed", -20), ("pushed", -10)):
        tell_event(mind, Ref(f"event:{name}"), at=NOW + timedelta(minutes=minutes), kind="act", source=SOURCE)
    assert [e.ref.id for e in events(mind)] == ["event:wrote", "event:committed", "event:pushed"]
    assert relate(mind, Ref("event:wrote"), Ref("event:pushed")) == "before"
    assert relate(mind, Ref("event:pushed"), Ref("event:wrote")) == "after"
    assert [e.ref.id for e in after(mind, Ref("event:committed"))] == ["event:pushed"]
    assert [e.ref.id for e in before(mind, Ref("event:committed"))] == ["event:wrote"]
    assert [e.ref.id for e in since(mind, Ref("event:wrote"))] == ["event:committed", "event:pushed"]
    window = during(mind, NOW - timedelta(minutes=25), NOW)
    assert [e.ref.id for e in window] == ["event:committed", "event:pushed"]


def test_simultaneous_is_its_own_answer():
    mind = Store()
    tell_event(mind, Ref("event:a"), at=NOW, source=SOURCE)
    tell_event(mind, Ref("event:b"), at=NOW + timedelta(seconds=1), source=SOURCE)
    assert relate(mind, Ref("event:a"), Ref("event:b")) == "before"
    assert relate(mind, Ref("event:a"), Ref("event:b"), tolerance=timedelta(seconds=2)) == "simultaneous"


def test_order_separates_what_it_cannot_place():
    mind = Store()
    tell_event(mind, Ref("event:timed"), at=NOW, source=SOURCE)
    mind.tell(Claim(Ref("event:undated"), "is_a", "act"), Evidence(SOURCE, NOW, method="test"))
    placed, unplaced = order(mind, [Ref("event:undated"), Ref("event:timed")])
    assert [r.id for r in placed] == ["event:timed"] and [r.id for r in unplaced] == ["event:undated"]


def test_what_changed_since_reads_the_recording_clock():
    mind = Store()
    early, late = NOW - timedelta(minutes=10), NOW
    mind.tell(Claim(Ref("path:/a"), "is_a", "file"), Evidence(SOURCE, early, method="test"))
    mind.tell(Claim(Ref("path:/b"), "is_a", "file"), Evidence(SOURCE, late, method="test"))
    changed = changed_since(mind, NOW - timedelta(minutes=5))
    assert [c.subject.id for c in changed] == ["path:/b"]


def test_language_can_order_two_events_neither_of_which_is_dated():
    """"the grain arrived before the snow came" is worth keeping even with no clock."""
    mind = Store()
    tell_order(mind, Ref("event:grain"), Ref("event:snow"), source=SOURCE)
    assert ordered_by_claims(mind, Ref("event:grain"), Ref("event:snow")) == "before"
    assert ordered_by_claims(mind, Ref("event:snow"), Ref("event:grain")) == "after"
    assert isinstance(ordered_by_claims(mind, Ref("event:grain"), Ref("event:other")), Unknown)


def test_connectives_map_to_the_order_they_assert():
    assert connective_relation("before") == "before"
    assert connective_relation("after") == "after"
    assert connective_relation("while") == "during"
    assert connective_relation("since") == "after"
    assert isinstance(connective_relation("because"), Unknown)  # a reason is not an ordering
