"""Object permanence: surviving absence without asserting stale readings as current."""

from datetime import datetime, timedelta, timezone

import tensacode as tc
from tensacode.change import Item, snapshot
from tensacode.permanence import OBJECTS, Objects, describe_attribute, still_there

T0 = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def item(key, *, label="", value="", window="", where=(0, 0, 10, 10), kind="text"):
    return Item(key, kind, label, value, window, where, False)


def snap(items, seconds=0):
    return snapshot(items, at=T0 + timedelta(seconds=seconds))


def test_a_thing_that_leaves_view_still_exists():
    objects = Objects()
    objects.observe(snap([item("a", label="Files", kind="control")]))
    ref = objects.get("a").ref
    objects.observe(snap([], 1))
    file = objects.get("a")
    assert file.ref == ref and file.exists and not file.present
    assert file.state == "out_of_view"


def test_leaving_view_is_not_being_destroyed_and_says_so():
    objects = Objects()
    objects.observe(snap([item("a", label="Files", kind="control")]))
    objects.observe(snap([], 1))
    said = still_there(objects, "Files")
    assert "can't see it now" in said and "still is" in said


def test_a_window_seen_to_close_takes_its_contents_with_it():
    objects = Objects()
    objects.observe(snap([item("t1", value="one", window="Terminal"), item("t2", value="two", window="Terminal")]))
    assert objects.closed("Terminal", at=T0 + timedelta(seconds=1)) == 2
    assert len(objects.gone()) == 2 and objects.absent() == []
    assert "gone" in still_there(objects, "Terminal") or "gone" in objects.gone()[0].describe()


def test_seeing_something_again_beats_having_written_it_off():
    objects = Objects()
    objects.observe(snap([item("t1", value="one", window="Terminal")]))
    objects.closed("Terminal", at=T0 + timedelta(seconds=1))
    report = objects.observe(snap([item("t1", value="one", window="Terminal")], 2))
    assert report.wrong_about_gone == 1
    assert objects.get("t1").state == "in_view"


def test_an_attribute_of_something_absent_comes_back_dated_and_not_assertable():
    objects = Objects()
    objects.observe(snap([item("a", label="Rate", value="7", kind="control")]))
    assert objects.get("a").assertable("value")
    objects.observe(snap([], 1))
    file = objects.get("a")
    assert not file.assertable("value")
    attribute = file.attribute("value")
    assert attribute.value == "7" and attribute.stale
    assert "not visible now" in attribute.describe()
    assert "as of" in describe_attribute(objects, "Rate", "value")


def test_a_freshness_budget_can_be_asked_for_explicitly():
    objects = Objects()
    objects.observe(snap([item("a", label="Rate", value="7", kind="control")]))
    objects.observe(snap([], 1))
    file = objects.get("a")
    now = T0 + timedelta(seconds=2)
    assert file.assertable("value", within=timedelta(seconds=30), now=now)
    assert not file.assertable("value", within=timedelta(seconds=1), now=now)


def test_an_attribute_of_something_destroyed_is_never_assertable():
    objects = Objects()
    objects.observe(snap([item("t1", label="Rate", value="7", window="Terminal", kind="control")]))
    objects.closed("Terminal", at=T0 + timedelta(seconds=1))
    assert not objects.get("t1").assertable("value", within=timedelta(hours=1))


def test_identity_survives_the_scene_graph_renumbering_its_keys():
    """Scene-graph keys carry an occurrence index, so losing one row renames the rest."""
    objects = Objects()
    objects.observe(snap([item("text:T#1", value="alpha", window="Terminal"),
                          item("text:T#2", value="beta", window="Terminal", where=(0, 20, 10, 10))]))
    ref = [f for f in objects.known() if f.attributes.get("value") == "beta"][0].ref
    report = objects.observe(snap([item("text:T#1", value="beta", window="Terminal")], 1))
    assert report.rematched == 1
    assert [f for f in objects.known() if f.attributes.get("value") == "beta"][0].ref == ref


def test_one_objects_file_is_not_handed_to_the_next_occupant_of_its_slot():
    objects = Objects()
    objects.observe(snap([item("text:T#1", value="keeper.txt", window="Terminal")]))
    objects.observe(snap([item("text:T#1", value="something else", window="Terminal")], 1))
    values = {f.attributes.get("value") for f in objects.known()}
    assert values == {"keeper.txt", "something else"}


def test_existence_is_written_to_its_own_scope_and_does_not_grow_per_look():
    mind = tc.Store()
    objects = Objects()
    for i in range(6):
        objects.observe(snap([item("a", label="Files", kind="control")], i))
        objects.remember(mind, at=T0 + timedelta(seconds=i))
    live = [r for r in mind.claims() if not r.retracted and r.claim.scope == OBJECTS]
    assert {r.claim.predicate for r in live} >= {"exists", "label", "state"}
    assert len(live) <= 6, f"one claim per look would be a leak: {len(live)}"
    assert len([r for r in live if r.claim.predicate == "state"]) == 1
