"""Reading a terminal out of a rendered frame: wraps, chrome, the caret, and refusals."""

from __future__ import annotations

import numpy as np
import pytest

from examples.browser_agents.perception.protocol import TextBlock
from examples.browser_agents.vision.cw_e2e import (
    INVARIANTS,
    PROJECT_ID,
    PROJECT_NAME,
    caret_in,
    pane_box,
    pane_lines,
    wrapped_together,
)
from examples.browser_agents.vision.perceive import PROMPT

WINDOW = (130, 70, 960, 610)  # a terminal window; its inside right edge is 1082


def line(text: str, y: int, x: int = 134, pitch: float = 7.9) -> TextBlock:
    return TextBlock(text=text, box=(x, y, int(len(text) * pitch), 16))


def test_a_full_width_line_continues_on_the_next_one():
    first = line("a" * 119, 200)  # runs to the right margin
    second = line("tail of it", 219)
    assert wrapped_together([first, second], WINDOW[0] + WINDOW[2] - 8) == ["a" * 119 + "tail of it"]


def test_a_wrap_that_ate_a_space_shows_it_as_an_indent():
    """The engine's own measurements: a continuation at the margin glues, an indented one does not."""
    mid_word = [line("x" * 110 + " in notes/ w", 200), line("ith these items", 219)]
    at_space = [line("y" * 110 + " all as 'initial", 200), line("commit'.", 219, x=142)]
    right = WINDOW[0] + WINDOW[2] - 8
    assert wrapped_together(mid_word, right) == ["x" * 110 + " in notes/ with these items"]
    assert wrapped_together(at_space, right) == ["y" * 110 + " all as 'initial commit'."]


def test_a_short_line_is_its_own_line():
    assert wrapped_together([line("readme-first.txt", 160), line("task-41001.txt", 180)], WINDOW[0] + WINDOW[2] - 8) == [
        "readme-first.txt", "task-41001.txt",
    ]


def test_window_chrome_above_the_margin_is_not_output():
    title = TextBlock(text="terminal", box=(561, 84, 64, 18))
    tab = TextBlock(text="Terminal", box=(161, 125, 60, 16))
    out = pane_lines((title, tab, line("readme-first.txt", 160), line("indented", 180, x=166)), WINDOW)
    assert [t.text for t in out] == ["readme-first.txt", "indented"]


def test_the_pane_runs_to_the_window_edges():
    box = pane_box([line("readme-first.txt", 160)], WINDOW, None)
    assert box[0] < 134 and box[1] < 160
    assert box[0] + box[2] == WINDOW[0] + WINDOW[2] - 8
    assert box[1] + box[3] == WINDOW[1] + WINDOW[3] - 8


def test_a_bare_dollar_is_a_prompt_but_a_price_is_not():
    assert PROMPT.match("$ ls ~/Desktop")
    assert PROMPT.match("$")
    assert not PROMPT.match("$51.25")
    assert not PROMPT.match("Total $8")


def test_the_caret_is_found_by_shape_not_by_reading():
    pytest.importorskip("cv2")
    frame = np.zeros((300, 400, 3), np.uint8)
    frame[:, :] = (40, 10, 35)  # the pane's background
    frame[100:116, 20:60] = (200, 200, 200)  # a word: wide, so not a caret
    frame[200:216, 20:23] = (200, 200, 200)  # a thin stroke: narrow, so not a caret
    frame[240:256, 24:33] = (220, 220, 220)  # the caret: a solid block
    assert caret_in(frame, (0, 0, 400, 300)) == (24, 240, 9, 16)


def test_a_pane_with_no_caret_and_no_block_has_none():
    pytest.importorskip("cv2")
    frame = np.zeros((300, 400, 3), np.uint8)
    frame[:, :] = (40, 10, 35)
    assert caret_in(frame, (0, 0, 400, 300)) is None


def test_the_invariant_refuses_two_readings_of_one_identifier():
    agree = ["task-41001.txt", "New project please: orbit-notes-41001 (in ~/Projects)."]
    disagree = ["task-41001.txt", "New project please: orbit-notes-41007 (in ~/Projects)."]
    from examples.browser_agents.perception.invariants import check

    assert check(agree, [PROJECT_ID]).corroborated == {"project id": "41001"}
    assert check(disagree, [PROJECT_ID]).violations == {"project id": ["41001", "41007"]}


def test_a_lost_hyphen_is_caught_by_shape_not_by_agreement():
    """The measured false success: "atlas-sync-31288" read as "atlas sync-31288"."""
    from examples.browser_agents.perception.invariants import check, mask

    misread = ["task-31288.txt", "New project please: atlas sync-31288 (in ~/Projects). README.md should say 'Atlas Sync'."]
    assert check(misread, [PROJECT_ID]).unseen == ["project id"]  # agreement cannot see it
    verdict = check(misread, [PROJECT_NAME])
    assert verdict.violations == {"project name": ["atlas"]}
    left = mask(misread, INVARIANTS, verdict)[1]
    assert "project please: <refused> sync-31288" in left  # the grammar now abstains: no name to parse
    assert "'Atlas Sync'" in left  # and the rest of the line is untouched


def test_a_well_formed_name_passes():
    from examples.browser_agents.perception.invariants import check

    good = ["New project please: atlas-sync-31288 (in ~/Projects)."]
    assert check(good, [PROJECT_NAME]).corroborated == {"project name": "atlas-sync-31288"}
