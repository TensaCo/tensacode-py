"""Task-stated invariants: corroborate a literal read in two places, or refuse it."""

from examples.browser_agents.perception.invariants import SameIdentifier, check

PROJECT = SameIdentifier("project id", r"[a-z]+-[a-z]+-(\d{4,})", r"task-(\d{4,})\.txt")


def test_the_same_value_read_in_two_places_is_corroborated():
    texts = ["task-72004.txt", "cat ~/Desktop/task-72004.txt", "Set up a project called signal-desk-72004 under ~/Projects"]
    verdict = check(texts, [PROJECT])
    assert verdict.ok and verdict.corroborated == {"project id": "72004"}


def test_a_misread_identifier_is_a_violation_naming_both_readings():
    texts = ["task-72004.txt", "Set up a project called signal-desk-72804 under ~/Projects"]
    verdict = check(texts, [PROJECT])
    assert not verdict.ok and verdict.violations == {"project id": ["72004", "72804"]}
    assert "does not hold" in str(verdict)


def test_an_invariant_whose_places_are_not_all_on_screen_is_reported_as_unseen_not_violated():
    verdict = check(["task-72004.txt"], [PROJECT])
    assert verdict.ok and verdict.unseen == ["project id"] and not verdict.corroborated
