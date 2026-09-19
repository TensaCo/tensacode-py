"""The computerworld body and provider: what an agent sees, does, and can reproduce."""

import pytest

pytest.importorskip("computerworld")

from examples.browser_agents.perception.computerworld import CwProvider, scene_box, scene_point, split_prompt, windows_in  # noqa: E402
from examples.browser_agents.perception.cw_body import CwBody  # noqa: E402
from examples.browser_agents.worlds import desktop_world, note_world  # noqa: E402
from examples.browser_agents.worlds.runtime import CwWorld, expand  # noqa: E402

NOTE = ("Set up a project called demo-1 under ~/Projects. Create README.md containing the line 'Demo One' "
        "and a notes folder with todo.txt listing: first thing; second thing. Make it a git repository and "
        "commit everything with the message 'initial commit'.")


def body(seed: int = 1, note: str = NOTE):
    world = CwWorld(note_world(note, seed), seed)
    return CwBody(world.actor(), CwProvider(), episode=f"test-{seed}"), world


def open_terminal(ui):
    screen = ui.observe()
    dock = next(c for c in screen.controls if c.name == "Terminal")
    ui.click(dock)
    screen = ui.observe()
    return next(c for c in screen.controls if c.name == "Shell input")


def test_the_desktop_offers_a_dock_and_a_terminal_that_opens():
    ui, _ = body()
    screen = ui.observe()
    assert {"Terminal", "Files", "Text Editor"} <= {c.name for c in screen.controls}
    assert all(c.point is not None for c in screen.controls)
    shell = open_terminal(ui)
    assert shell.role == "textbox" and shell.section == "Terminal"
    assert any(r.label == "Terminal" for r in ui.last_scene.regions)


def test_typing_is_verified_and_the_transcript_reads_back_in_order():
    ui, _ = body()
    shell = open_terminal(ui)
    assert ui.fill(shell, "ls ~/Desktop", submit=True).status == "applied"
    ui.fill(shell, "cat ~/Desktop/task-1.txt", submit=True)
    lines = [t.text for t in ui.observe().texts if t.section == "Terminal"]
    assert lines[0] == f"{ui.surface.prompt()} ls ~/Desktop"
    assert "task-1.txt" in lines[1:3]
    said = lines.index(f"{ui.surface.prompt()} cat ~/Desktop/task-1.txt")
    assert "demo-1" in " ".join(lines[said + 1:]), lines


def test_output_is_unwrapped_so_long_text_survives_the_window_width():
    """The scene wraps mid-word; a reader gets the engine's logical lines instead."""
    ui, _ = body()
    shell = open_terminal(ui)
    ui.fill(shell, "cat ~/Desktop/task-1.txt", submit=True)
    printed = [t.text for t in ui.observe().texts if t.section == "Terminal" and not t.text.startswith("agent@")]
    assert NOTE in " ".join(printed)
    wrapped = [t.text for t in ui.last_scene.texts if t.section == "Terminal"]
    assert len(wrapped) > len(printed)  # the screen really does wrap it


def test_a_command_owns_only_the_lines_printed_while_it_ran():
    ui, _ = body()
    shell = open_terminal(ui)
    for command in ("echo one", "cat ~/Desktop/task-1.txt", "echo two"):
        ui.fill(shell, command, submit=True)
    lines = [t.text for t in ui.observe().texts if t.section == "Terminal"]
    prompt = ui.surface.prompt()
    assert lines.index("one") == lines.index(f"{prompt} echo one") + 1
    assert lines.index("two") == lines.index(f"{prompt} echo two") + 1


def test_the_transcript_is_the_engine_s_own_text_including_the_exit_status():
    """The body used to echo what it had typed, because the engine did not.

    It does now: one entry per command, with the line it printed, the output, and the exit
    status. Nothing in the transcript is this body's reconstruction any more, and whether a
    command worked is the machine's answer rather than a search for "not found".
    """
    ui, _ = body()
    shell = open_terminal(ui)
    ui.fill(shell, "echo hello", submit=True)
    ui.observe()
    texts = {t.text: t for t in ui.last_read.texts if t.section == "Terminal"}
    assert texts[f"{ui.surface.prompt()} echo hello"].provenance[0].method == "semantic.v1"
    assert texts["hello"].provenance[0].method == "semantic.v1"
    [entry] = ui.surface.terminal_entries()
    assert entry.exit_code == 0 and entry.ok is True

    ui.fill(shell, "nosuchcommand", submit=True)
    failed = ui.surface.terminal_entries()[-1]
    assert failed.ok is False and failed.exit_code == 127
    assert "[exit 127]" in [t.text for t in ui.observe().texts if t.section == "Terminal"]


def test_the_same_seed_replays_to_the_same_world():
    def run(seed):
        ui, world = body(seed)
        shell = open_terminal(ui)
        ui.fill(shell, "mkdir -p ~/Projects/p && cd ~/Projects/p && echo x > a.txt", submit=True)
        return world.state_hash()

    assert run(3) == run(3)
    assert run(3) != run(4)


def test_scoring_reads_the_world_the_agent_cannot():
    ui, world = body()
    shell = open_terminal(ui)
    ui.fill(shell, "mkdir -p ~/Projects/demo-1/notes && cd ~/Projects/demo-1", submit=True)
    ui.fill(shell, "echo 'Demo One' > README.md && git init && git add . && git commit -m 'initial commit'", submit=True)
    assert world.read("/home/agent/Projects/demo-1/README.md").strip() == "Demo One"
    assert world.commits("/home/agent/Projects/demo-1")[0][0] == "initial commit"
    assert world.pending("/home/agent/Projects/demo-1") == []
    assert world.stat("/home/agent/Projects/demo-1")["is_dir"] is True
    assert world.exists("/home/agent/nope") is False


def test_geometry_goes_through_the_engines_transform():
    node = {"bounds": {"x": 10, "y": 20, "width": 40, "height": 8}, "transform": {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 100, "ty": 200}}
    assert scene_box(node) == (110, 220, 40, 8)
    assert scene_point(node) == (130, 224)


def test_prompt_split_and_window_titles():
    assert split_prompt(["out", "$ ls"]) == (["out"], "$ ls")
    assert split_prompt(["out"]) == (["out"], None)
    raw = {"nodes": [
        {"interaction": "window:4:drag", "semantic": {"label": "Move terminal"}, "bounds": {"x": 0, "y": 0, "width": 100, "height": 10}},
        {"interaction": "window:4:focus", "semantic": {"label": "terminal"}, "bounds": {"x": 0, "y": 0, "width": 500, "height": 400}},
    ]}
    assert windows_in(raw)["4"][0] == "Terminal"


def test_world_definitions_are_data():
    world = desktop_world({"Desktop/x.txt": "hi\n"})
    assert world["computers"][0]["initial_files"]["Desktop/x.txt"] == "hi\n"
    assert "task-9.txt" in str(note_world("n", 9))
    assert expand("~/Projects/x") == "/home/agent/Projects/x"
