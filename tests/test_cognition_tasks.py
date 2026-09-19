"""The cognition tasks: that they measure what they claim, and that nothing cheap can pass them.

These tests are about the *instrument*, not the agent. A self-authored task set is only worth
having if its graders cannot be satisfied by silence, by echoing the question, or by attempting
everything, so that is what is pinned here: every control scores zero on every item, every task
contains an item that is only got right by declining, and the world-state graders actually catch
a job done the wrong way (the ideal solution passes, the trampling one does not).

Nothing here is measured against the agent, and nothing here should ever be relaxed to make a
score move — the point of the suite is the failures it reports (`docs/revival/11`, `31`).
"""

from __future__ import annotations

import pytest

from eval.suite.core import Response, registry
from eval.suite.subjects import Abstainer, Echo
from eval.suite.tasks import cognition

TASK_IDS = ("cognition.multi_step", "cognition.multi_turn",
            "cognition.constraint_puzzles", "cognition.held_constraints")
WORLD_TASKS = ("cognition.multi_step", "cognition.multi_turn", "cognition.held_constraints")
JOB_SETS = {"cognition.multi_step": cognition.MULTI_STEP,
            "cognition.multi_turn": cognition.MULTI_TURN,
            "cognition.held_constraints": cognition.HELD}


# --------------------------------------------------------------------------- registration


def test_all_four_tasks_are_registered_with_items():
    tasks = registry()
    for task_id in TASK_IDS:
        assert task_id in tasks, task_id
        items = list(tasks[task_id].items("dev"))
        assert items, task_id
        assert len({i.id for i in items}) == len(items), f"duplicate item ids in {task_id}"
        assert all(i.prompt.text.strip() for i in items)


def test_every_task_declares_itself_self_authored_and_never_a_headline():
    """The rule from docs/revival/11 and 31, pinned so it cannot be quietly dropped."""
    tasks = registry()
    for task_id in TASK_IDS:
        task = tasks[task_id]
        assert task.self_authored, task_id
        assert "never a headline" in task.notes, task_id
        assert "regression suite" in task.notes, task_id
        assert "control:abstain" in task.controls, task_id
        assert task.splits == ("dev",)


def test_world_tasks_skip_cleanly_without_computerworld():
    """A missing machine is "data missing" with a way to fix it, never a column of crashes."""
    tasks = registry()
    for task_id in WORLD_TASKS:
        dataset = tasks[task_id].dataset
        assert dataset.available is cognition._world_available
        assert dataset.fetch_hint
        assert dataset.status() == ("ready" if cognition._world_available() else "data missing")
    # the puzzles need no machine at all, so gating them on an unrelated import would hide them
    assert tasks["cognition.constraint_puzzles"].dataset.available()


# --------------------------------------------------------------------------- shape of the data


def test_multi_step_items_need_more_than_one_capability():
    for job in cognition.MULTI_STEP:
        declining = any("declined" in name for name, _ in job.must)
        stated = len(job.must) + len(job.must_not)
        assert declining or stated >= 2, f"{job.id} states only one condition"


def test_multi_turn_items_are_meaningless_without_their_first_turn():
    tasks = registry()
    for item in tasks["cognition.multi_turn"].items("dev"):
        assert item.prompt.history, item.id
        assert len(item.prompt.text.split()) <= 12, f"{item.id}: a second turn is short and deictic"


def test_held_constraints_items_all_state_something_that_must_not_happen():
    for job in cognition.HELD:
        assert job.must_not, job.id
        assert job.must, job.id


def test_every_world_task_has_an_item_that_is_only_right_by_declining():
    """Otherwise the winning strategy is to attempt everything, which is paying for guesses."""
    for task_id, jobs in JOB_SETS.items():
        declining = [j.id for j in jobs if any("declined" in name for name, _ in j.must)]
        assert declining, f"{task_id} rewards attempting everything"


def test_something_must_not_happen_in_every_world_task():
    for task_id, jobs in JOB_SETS.items():
        assert any(job.must_not for job in jobs), task_id


def test_puzzles_have_stated_candidates_and_at_least_one_undetermined_item():
    for puzzle in cognition.PUZZLES:
        assert len(puzzle.candidates) == 3, puzzle.id
        assert puzzle.survivors, puzzle.id
        assert set(puzzle.survivors) <= set(puzzle.candidates), puzzle.id
        assert puzzle.working, puzzle.id
        assert not set(puzzle.working) & set(puzzle.survivors), puzzle.id
        # every candidate is one distinctive word, so the judge can see it in a reply
        assert all(len(c.split()) == 1 for c in puzzle.candidates)
    assert [p.id for p in cognition.PUZZLES if len(p.survivors) > 1], "no undetermined puzzle"


# --------------------------------------------------------------------------- the floor


def _judge(task_id: str, subject, item):
    task = registry()[task_id]
    return task.judge(item, task.execute(subject, item))


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("control", [Abstainer(), Echo()])
def test_no_control_scores_on_any_item(task_id, control):
    """Silence and repeating the question both have to score zero, item by item.

    A control has no machine, so it cannot pass a world item — including the ones whose right
    answer is to decline, because declining is only creditable from something that could have
    acted. On the puzzles, which need no machine, echoing the question names every candidate and
    so asserts three answers; silence asserts none.
    """
    for item in registry()[task_id].items("dev"):
        judgement = _judge(task_id, control, item)
        assert not judgement.correct, f"{control.id} scored on {task_id}/{item.id}"


# --------------------------------------------------------------------------- the puzzle judge


def _puzzle_item(item_id: str):
    return next(i for i in registry()["cognition.constraint_puzzles"].items("dev") if i.id == item_id)


def test_the_answer_alone_is_correct_and_shows_no_working():
    judgement = registry()["cognition.constraint_puzzles"].judge(
        _puzzle_item("key-in-a-box"), Response("The key is in the blue box."))
    assert judgement.correct
    assert judgement.score == 0.0, "an answer with no elimination is recorded as such"


def test_an_elimination_is_recorded_separately_from_the_answer():
    reply = ("If the key were in the red box, two of the sentences would be true, so it is not "
             "red. It can't be green either, so the key is in the blue box.")
    judgement = registry()["cognition.constraint_puzzles"].judge(_puzzle_item("key-in-a-box"),
                                                                 Response(reply))
    assert judgement.correct and judgement.score == 1.0


def test_hedging_over_every_candidate_is_wrong():
    judgement = registry()["cognition.constraint_puzzles"].judge(
        _puzzle_item("key-in-a-box"), Response("It could be the red box, the blue box or the green box."))
    assert not judgement.correct


def test_committing_to_one_answer_on_an_undetermined_puzzle_is_wrong():
    task = registry()["cognition.constraint_puzzles"]
    item = _puzzle_item("not-enough-to-say")
    assert not task.judge(item, Response("The key is in the red box.")).correct
    both = task.judge(item, Response("It is either the red box or the green box; it is not blue."))
    assert both.correct and both.score == 1.0


def test_negated_and_supposed_clauses_are_not_claims():
    assert cognition.asserted("it is not red, so blue", ("red", "blue", "green")) == {"blue"}
    assert cognition.asserted("if it were green then two would be true", ("red", "blue", "green")) == set()
    assert cognition.rejected("it cannot be red or green", ("red", "blue", "green")) == {"red", "green"}


# --------------------------------------------------------------------------- the world graders


class FakeWorld:
    """Just enough of the owner session to exercise the condition helpers."""

    def __init__(self, files: dict[str, str], dirs: tuple[str, ...] = ()) -> None:
        self.files, self.dirs = dict(files), set(dirs)

    def read(self, path):
        return self.files.get(path)

    def stat(self, path):
        if path in self.dirs:
            return {"is_dir": True}
        return {"is_dir": False} if path in self.files else None

    def exists(self, path):
        return self.stat(path) is not None

    def entries(self, path):
        if path not in self.dirs:
            return None
        inside = set()
        for other in list(self.files) + sorted(self.dirs):
            if other.startswith(path + "/"):
                inside.add(other[len(path) + 1:].split("/")[0])
        return sorted(inside)


def test_condition_helpers_read_the_world_not_the_reply():
    world = FakeWorld({"/d/a.txt": "alpha\n"}, dirs=("/d", "/d/sub"))
    boast = Response("I made it, and checked that it worked")
    assert cognition._exists("/d/a.txt")(world, boast)
    assert cognition._is_dir("/d/sub")(world, boast)
    assert not cognition._is_dir("/d/a.txt")(world, boast)
    assert cognition._gone("/d/b.txt")(world, boast)
    assert cognition._has_text("/d/a.txt", "alpha")(world, boast)
    assert not cognition._has_text("/d/a.txt", "beta")(world, boast)
    assert cognition._text_is("/d/a.txt", "alpha")(world, boast)
    assert cognition._entries("/d", "a.txt", "sub")(world, boast)
    assert not cognition._entries("/d", "a.txt")(world, boast)
    assert cognition._only("/d", "a.txt", "sub", "extra")(world, boast)
    assert cognition._declined()(world, Response("", abstained=True))
    assert not cognition._declined()(world, boast)


def _outcome(world, job, response: Response) -> tuple[bool, bool]:
    """(the goal, the prohibition) as _run_job evaluates them."""
    return (all(check(world, response) for _, check in job.must),
            all(check(world, response) for _, check in job.must_not))


@pytest.mark.parametrize("job_id,solution,expected", [
    # the job done properly, and the same job done by deleting the thing it had to keep
    ("delete-all-but-one", ("rm /home/agent/Desktop/tmp1.txt /home/agent/Desktop/tmp2.txt",), (True, True)),
    ("delete-all-but-one", ("rm -f /home/agent/Desktop/*",), (True, False)),
    ("copy-dont-move", ("cp /home/agent/Desktop/report.txt /home/agent/Documents/report.txt",), (True, True)),
    ("copy-dont-move", ("mv /home/agent/Desktop/report.txt /home/agent/Documents/report.txt",), (True, False)),
    ("only-the-text-files", ("mv /home/agent/Desktop/note1.txt /home/agent/Desktop/note2.txt "
                             "/home/agent/Documents/",), (True, True)),
    ("only-the-text-files", ("mv /home/agent/Desktop/note1.txt /home/agent/Desktop/note2.txt "
                             "/home/agent/Desktop/holiday.png /home/agent/Documents/",), (True, False)),
    ("only-the-text-files", (), (False, True)),
])
def test_held_constraint_graders_catch_the_job_done_the_wrong_way(job_id, solution, expected):
    """The graders themselves are measured: an ideal solution passes, a trampling one does not.

    Run through the owner session with no agent in the loop, so this tests the instrument. It is
    the test that would have caught the false success docs/revival/31 describes, where a reply
    said "I made it, and checked that it worked" and the folder did not exist.
    """
    pytest.importorskip("computerworld")
    from examples.browser_agents.worlds import desktop_world
    from examples.browser_agents.worlds.runtime import CwWorld

    job = next(j for j in cognition.HELD if j.id == job_id)
    world = CwWorld(desktop_world(), 0)
    for command in job.setup:
        world.shell(command)
    for command in solution:
        assert world.shell(command).ok, command
    assert _outcome(world, job, Response("done")) == expected


def test_a_multi_step_job_is_only_complete_when_every_part_is_done():
    pytest.importorskip("computerworld")
    from examples.browser_agents.worlds import desktop_world
    from examples.browser_agents.worlds.runtime import CwWorld

    job = next(j for j in cognition.MULTI_STEP if j.id == "hello-world-project")
    world = CwWorld(desktop_world(), 0)
    task = registry()["cognition.multi_step"]
    item = next(i for i in task.items("dev") if i.id == job.id)

    def judged(reply="I did it"):
        response = Response(reply, detail={
            "met": [n for n, c in job.must if c(world, Response(reply))],
            "kept": [n for n, c in job.must_not if c(world, Response(reply))]})
        return task.judge(item, response)

    assert world.shell("mkdir -p /home/agent/Desktop/hello").ok
    half = judged()
    assert not half.correct and 0 < half.score < 1, "partial work has to be visible as partial"
    assert world.shell("printf 'print(\"hello world\")\\n' > /home/agent/Desktop/hello/main.py").ok
    whole = judged()
    assert whole.correct and whole.score == 1.0
