"""Real filesystem effects, with generic planning and independent observations."""
from pathlib import Path

import pytest

from tensorcode.actions import Plan
from tensorcode.agent.filesystem import FileSystemPlugin
from tensorcode.agent.planning import plan_goal
from tensorcode.agent.plugin import Call
from tensorcode.goals import Condition, GoalSpec
from tensorcode.outcomes import Unknown


def execute(plugin, goal):
    plan = plan_goal(goal, [plugin])
    assert isinstance(plan, Plan), plan
    for step in plan.steps:
        receipt = plugin.execute(step.action, key=step.id)
        assert receipt.status == "applied", receipt
    assert all(plugin.observe_condition(condition) is True for condition in goal.conditions)
    return plan


def test_nested_project_content_is_planned_and_observed(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    goal = GoalSpec((Condition("directory_exists", {"path": "demo"}),
                     Condition("content", {"path": "demo/src/main.py", "text": 'print("Hello")\n'})))
    plan = execute(plugin, goal)
    assert len(plan.steps) == 3
    assert (tmp_path / "demo/src/main.py").read_text() == 'print("Hello")\n'
    assert not plan_goal(goal, [plugin]).steps


def test_existing_readme_preserved_and_existing_different_content_rejected(tmp_path):
    (tmp_path / "README.md").write_text("Keep this")
    plugin = FileSystemPlugin(tmp_path)
    execute(plugin, GoalSpec((Condition("content", {"path": "src/main.py", "text": "hello\n"}),)))
    assert (tmp_path / "README.md").read_text() == "Keep this"
    impossible = plan_goal(GoalSpec((Condition("content", {"path": "README.md", "text": "replace"}),)), [plugin])
    assert isinstance(impossible, Unknown)
    receipt = plugin.execute(Call(plugin.name, "write_file", (("path", "README.md"), ("parent", "."), ("text", "replace"))), key="attempt")
    assert receipt.status == "rejected"
    assert (tmp_path / "README.md").read_text() == "Keep this"


def test_plain_file_goal_creates_empty_file(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    execute(plugin, GoalSpec((Condition("file_exists", {"path": "nested/empty"}),)))
    assert (tmp_path / "nested/empty").read_bytes() == b""


@pytest.mark.parametrize("value", ["../escape", "/outside-root"])
def test_escaping_paths_cannot_be_grounded_or_observed(tmp_path, value):
    plugin = FileSystemPlugin(tmp_path)
    condition = Condition("content", {"path": value, "text": "bad"})
    assert isinstance(plugin.observe_condition(condition), Unknown)
    assert list(plugin.enumerate_actions(GoalSpec((condition,)))) == []


def test_symlink_even_inside_root_is_rejected(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    plugin = FileSystemPlugin(tmp_path)
    condition = Condition("content", {"path": "link/file", "text": "bad"})
    assert isinstance(plugin.observe_condition(condition), Unknown)
    assert list(plugin.enumerate_actions(GoalSpec((condition,)))) == []
    receipt = plugin.execute(Call(plugin.name, "write_file", (("path", "link/file"), ("parent", "link"), ("text", "bad"))), key="attempt")
    assert receipt.status == "rejected"
    assert not (tmp_path / "real/file").exists()


def test_wrong_parent_binding_and_changed_preconditions_prevent_execution(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    goal = GoalSpec((Condition("content", {"path": "nested/file", "text": "one"}),))
    plan = plan_goal(goal, [plugin])
    assert isinstance(plan, Plan)
    write = next(step.action for step in plan.steps if step.action.capability == "write_file")
    assert plugin.execute(write, key="missing-parent").status == "rejected"
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/file").write_text("someone else")
    assert plugin.execute(write, key="race").status == "rejected"
    assert (tmp_path / "nested/file").read_text() == "someone else"
    malformed = Call(plugin.name, "write_file", (("path", "nested/other"), ("parent", "."), ("text", "bad")))
    assert plugin.execute(malformed, key="bad-parent").status == "rejected"


def test_content_is_exact_and_all_effects_must_be_observed(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    (tmp_path / "data").write_bytes(b"a\r\nb\n")
    assert plugin.observe_condition(Condition("content", {"path": "data", "text": "a\nb\n"})) is False
    assert plugin.observe_condition(Condition("content", {"path": "data", "text": "a\r\nb\n"})) is True
    assert plugin.observe_condition(Condition("file_exists", {"path": "missing"}, negated=True)) is True
    cap = next(c for c in plugin.capabilities() if c.name == "write_file")
    assert plugin.holds(cap, {"path": "data", "text": "wrong"}) is False
    assert isinstance(plugin.observe_condition(Condition("unmodeled", {"path": "data"})), Unknown)


def test_absolute_path_objects_within_root(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    execute(plugin, GoalSpec((Condition("content", {"path": tmp_path / "nested/file", "text": "data"}),)))


def test_real_language_with_explicit_project_recipe_runs(tmp_path):
    import subprocess
    import sys

    from agent_test_support import selected_agent as Agent, fixture_goal_selector
    from tensorcode.language import verbnet, wordnet

    if wordnet.find_wordnet() is None or verbnet.find_verbnet() is None:
        pytest.skip("requires WordNet and VerbNet data")
    from pathlib import Path
    from tensorcode.agent.refinements import RefinementLibrary
    recipes = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    plugin = FileSystemPlugin(tmp_path, refinements=recipes)
    agent = Agent([plugin], goal_selector=fixture_goal_selector('build-26.1-1', frame_index=0))
    turn = agent.turn("make a python project")
    assert turn.outcomes and turn.outcomes[0].status == "done", turn
    program = tmp_path / "hello-world/main.py"
    assert program.read_text() == 'print("Hello, world!")\n'
    ran = subprocess.run([sys.executable, str(program)], capture_output=True, text=True, check=True)
    assert ran.stdout == "Hello, world!\n"
    assert plugin.planning_enabled


def test_domain_recipes_are_absent_by_default(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    goal = GoalSpec((Condition("file_exists", {"path": "new"}),))
    assert isinstance(plugin.refine_goal(goal), Unknown)
    execute(plugin, goal)


@pytest.mark.parametrize("existing,desired", [("directory", "file_exists"), ("file", "directory_exists")])
def test_occupied_path_with_wrong_kind_has_no_plan(tmp_path, existing, desired):
    target = tmp_path / "occupied"
    if existing == "directory":
        target.mkdir()
    else:
        target.write_text("preserve me")
    plugin = FileSystemPlugin(tmp_path)
    goal = GoalSpec((Condition(desired, {"path": "occupied"}),))
    planned = plan_goal(goal, [plugin])
    assert isinstance(planned, Unknown)
    assert planned.reason == "no_plan"
    assert target.is_dir() if existing == "directory" else target.read_text() == "preserve me"


def test_same_path_cannot_be_both_created_file_and_directory(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    goal = GoalSpec((Condition("file_exists", {"path": "both"}),
                     Condition("directory_exists", {"path": "both"})))
    planned = plan_goal(goal, [plugin])
    assert isinstance(planned, Unknown)
    assert planned.reason == "no_plan"
    assert not (tmp_path / "both").exists()


def test_existing_directory_needs_no_creation(tmp_path):
    (tmp_path / "present").mkdir()
    plugin = FileSystemPlugin(tmp_path)
    goal = GoalSpec((Condition("directory_exists", {"path": "present"}),))
    plan = execute(plugin, goal)
    assert not plan.steps
    assert plugin.observe_condition(Condition("path_exists", {"path": "present"})) is True
