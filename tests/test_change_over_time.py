"""Change detection: the structures that turn a pile of differences into news."""

from datetime import datetime, timedelta, timezone

from tensorcode.change import (
    APPEARED, DISAPPEARED, FOCUS, MOVED, OCCLUDED, REPLACED, VALUE, WINDOW_OPENED,
    ChangePolicy, Item, Watcher, attribute_windows, diff, snapshot,
)

T0 = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def item(key, *, label="", value="", window="", where=(0, 0, 10, 10), kind="text", focused=False):
    return Item(key, kind, label, value, window, where, focused)


def snap(items, seconds=0, tag=""):
    return snapshot(items, at=T0 + timedelta(seconds=seconds), tag=tag)


def test_nothing_changed_is_reported_as_nothing():
    items = [item("a", value="one"), item("b", value="two", where=(0, 20, 10, 10))]
    assert not diff(snap(items), snap(items, 1)).all


def test_a_value_change_is_typed_and_carries_both_values():
    before, after = snap([item("a", value="one")]), snap([item("a", value="two")], 1)
    (change,) = diff(before, after).all
    assert (change.kind, change.was, change.now) == (VALUE, "one", "two")
    assert "one" in change.describe() and "two" in change.describe()


def test_a_window_opening_is_one_change_not_one_per_control():
    before = snap([item("bg", value="desktop", where=(0, 0, 1000, 700))])
    parts = [item(f"w{i}", value=f"row {i}", window="Text Editor", where=(100, 100 + 10 * i, 50, 10))
             for i in range(12)]
    after = snap([item("bg", value="desktop", where=(0, 0, 1000, 700)), *parts], 1)
    changes = diff(before, after)
    assert [c.kind for c in changes.all] == [WINDOW_OPENED]
    assert changes.all[0].brought == 11  # the window itself is the twelfth


def test_chrome_inside_the_window_is_folded_in_by_geometry():
    """The DOM does not say the title bar belongs to the window; where it sits does."""
    before = snap([item("bg", value="desktop", where=(0, 0, 1000, 700))])
    parts = [item(f"w{i}", value=f"row {i}", window="Files", where=(100, 120 + 10 * i, 50, 10)) for i in range(6)]
    chrome = item("close", label="Close", kind="control", where=(140, 110, 10, 10))  # no window of its own
    after = snap([item("bg", value="desktop", where=(0, 0, 1000, 700)), *parts, chrome], 1)
    kinds = [c.kind for c in diff(before, after).all]
    assert kinds == [WINDOW_OPENED], kinds


def test_something_under_a_new_window_is_occluded_not_destroyed():
    hidden = item("under", value="note", where=(120, 130, 40, 20))
    before = snap([hidden, item("bg", value="desktop", where=(0, 0, 1000, 700))])
    parts = [item(f"t{i}", value=f"line {i}", window="Terminal", where=(100, 100 + 10 * i, 200, 10)) for i in range(8)]
    after = snap([item("bg", value="desktop", where=(0, 0, 1000, 700)), *parts], 1)
    changes = diff(before, after)
    assert changes.of(DISAPPEARED) == ()
    (covered,) = changes.of(OCCLUDED)
    assert covered.key == "under" and "Terminal" in covered.describe()


def test_one_occupant_of_a_slot_replacing_another_is_one_change():
    """A slot is identified by where it is, not how wide its contents happen to be — and a box that
    shifted because its text got longer has not also moved."""
    before = snap([item("title", value="Terminal", where=(40, 8, 80, 16))])
    after = snap([item("title", value="Text Editor", where=(40, 8, 110, 16))], 1)
    (change,) = diff(before, after).all
    assert change.kind in (REPLACED, VALUE)
    assert "Terminal" in change.describe() and "Text Editor" in change.describe()


def test_a_key_that_changes_every_frame_is_learned_to_be_volatile():
    watcher = Watcher(policy=ChangePolicy(volatile_after=2, volatile_fraction=0.5))
    for i in range(6):
        watcher.see([item("clock", value=f"12:0{i}"), item("title", value="Files", where=(0, 20, 10, 10))], at=T0 + timedelta(seconds=i))
    changes = watcher.see([item("clock", value="12:09"), item("title", value="Documents", where=(0, 20, 10, 10))],
                          at=T0 + timedelta(seconds=9))
    assert [c.key for c in changes.steady] == ["title"]
    assert [c.key for c in changes.volatile] == ["clock"]
    assert "on their own" not in changes.summary()  # a steady change is the headline


def test_volatile_changes_are_flagged_and_still_countable():
    """Dropping them would make the noise rate unmeasurable; flagging keeps both countable."""
    watcher = Watcher(policy=ChangePolicy(volatile_after=2, volatile_fraction=0.5))
    for i in range(5):
        watcher.see([item("clock", value=f"12:0{i}")], at=T0 + timedelta(seconds=i))
    changes = watcher.see([item("clock", value="12:30")], at=T0 + timedelta(seconds=30))
    assert changes.all and changes.all[0].volatile
    assert changes.steady == ()


def test_since_a_turn_boundary_spans_every_frame_of_that_turn():
    watcher = Watcher()
    watcher.see([item("a", value="one")], tag="turn:1", at=T0)
    watcher.see([item("a", value="two")], tag="turn:1", at=T0 + timedelta(seconds=1))
    watcher.see([item("a", value="three")], tag="turn:2", at=T0 + timedelta(seconds=2))
    since_first = watcher.since_first("turn:1")
    assert since_first is not None and since_first.all[0].was == "one"


def test_focus_moving_is_its_own_kind_of_change():
    before = snap([item("f", label="Name", kind="control", focused=False)])
    after = snap([item("f", label="Name", kind="control", focused=True)], 1)
    assert [c.kind for c in diff(before, after).all] == [FOCUS]


def test_a_move_under_the_threshold_is_not_news():
    before = snap([item("a", value="x", where=(10, 10, 10, 10))])
    after = snap([item("a", value="x", where=(12, 11, 10, 10))], 1)
    assert not diff(before, after).all
    far = snap([item("a", value="x", where=(60, 10, 10, 10))], 1)
    assert [c.kind for c in diff(before, far).all] == [MOVED]


def test_attribute_windows_gives_chrome_the_window_it_sits_in():
    inside = [item("t1", value="a", window="Files", where=(100, 100, 50, 10)),
              item("t2", value="b", window="Files", where=(100, 200, 50, 10))]
    chrome = item("close", label="Close", kind="control", where=(140, 150, 8, 8))
    outside = item("bar", label="Trash", kind="control", where=(900, 600, 8, 8))
    got = {i.key: i.window for i in attribute_windows([*inside, chrome, outside])}
    assert got["close"] == "Files"
    assert got["bar"] == ""


def test_an_appearance_that_is_not_part_of_a_window_stays_an_appearance():
    before = snap([item("a", value="one")])
    after = snap([item("a", value="one"), item("b", value="two", where=(0, 40, 10, 10))], 1)
    assert [c.kind for c in diff(before, after).all] == [APPEARED]
