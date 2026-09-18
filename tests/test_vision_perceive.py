"""Pixel perception rules on synthetic screenshots, with fake OCR and detector outputs (no models)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import tensorcode as tc  # noqa: E402
from tensorcode.cognition import integrate  # noqa: E402

from examples.browser_agents.vision.desktop_e2e import expect_typed, shell_reading  # noqa: E402
from examples.browser_agents.vision.icon_memory import IconMemory  # noqa: E402
from examples.browser_agents.vision.models import Word  # noqa: E402
from examples.browser_agents.vision.perceive import inside, perceive, phrases_from_words, scene_fragment  # noqa: E402

WHITE, INK, LINE = (255, 255, 255), (30, 30, 30), (160, 160, 160)


def canvas(w=800, h=500, color=WHITE):
    return np.full((h, w, 3), color, np.uint8)


def ocr(*words):
    return lambda rgb: list(words)


def controls(scene, role=None):
    return [i.control for i in scene.controls if role is None or i.control.role == role]


def test_phrases_split_at_wide_gaps_and_keep_close_words():
    words = [Word("Save", (10, 10, 40, 14), 0.9, 0), Word("draft", (56, 10, 40, 14), 0.9, 0), Word("Send", (300, 10, 40, 14), 0.9, 0)]
    assert [p.text for p in phrases_from_words(words)] == ["Save draft", "Send"]


def test_outlined_field_is_a_textbox_named_by_the_label_above():
    img = canvas()
    cv2.rectangle(img, (100, 120), (500, 152), LINE, 1)
    scene = perceive(img, ocr=ocr(Word("Badge", (100, 96, 50, 14), 0.95, 0), Word("ID", (154, 96, 18, 14), 0.95, 0)))
    (box,) = controls(scene, "textbox")
    assert box.name == "Badge ID" and inside(box.point, (100, 120, 400, 32))


def test_filled_row_without_border_is_a_button_not_a_field():
    img = canvas()
    cv2.rectangle(img, (40, 100), (260, 126), (225, 235, 250), -1)
    scene = perceive(img, ocr=ocr(Word("Desktop", (60, 106, 60, 14), 0.95, 0)))
    assert not controls(scene, "textbox")
    assert [c.name for c in controls(scene, "button")] == ["Desktop"]


def test_square_box_left_of_label_is_a_checkbox_but_a_dot_is_not():
    img = canvas()
    cv2.rectangle(img, (100, 200), (115, 215), (110, 110, 110), 1)
    cv2.circle(img, (108, 300), 6, (230, 120, 20), -1)
    scene = perceive(img, ocr=ocr(Word("VPN", (122, 201, 34, 14), 0.95, 0), Word("online", (122, 293, 50, 14), 0.95, 1)), text_candidates=False)
    (box,) = controls(scene, "checkbox")
    assert box.name == "VPN" and box.checked is False and inside(box.point, (100, 200, 16, 16))


def test_window_frame_gives_sections_and_prompt_line_is_where_typing_goes():
    img = canvas(1000, 700, (60, 40, 90))
    cv2.rectangle(img, (150, 100), (850, 600), (20, 20, 24), -1)
    cv2.rectangle(img, (150, 100), (850, 140), (48, 48, 52), -1)
    words = [Word("Terminal", (470, 112, 64, 16), 0.99, 0), Word("agent@box:~$", (160, 160, 120, 16), 0.9, 1)]
    scene = perceive(img, ocr=ocr(*words))
    assert [w.title for w in scene.windows] == ["Terminal"]
    (prompt,) = [c for c in controls(scene, "textbox") if c.hint == "command prompt"]
    assert prompt.section == "Terminal" and prompt.box[0] > 280  # the input starts after the prompt text
    assert {t.section for t, _ in scene.texts} == {"Terminal"}


def test_scene_becomes_scored_snapshot_claims():
    img = canvas()
    cv2.rectangle(img, (100, 120), (500, 152), LINE, 1)
    scene = perceive(img, ocr=ocr(Word("Email", (100, 96, 44, 14), 0.9, 0)))
    frag = scene_fragment(scene, frame=3)
    assert frag.snapshot_of == tc.Ref("scope:screen") and frag.method == "pixel-scene-graph@1"
    label = next(c for c, _ in frag.claims if c.predicate == "label" and c.object == "Email")
    assert frag.confidence[label.id].kind == "uncalibrated" and 0 < frag.confidence[label.id].value <= 1
    mind = tc.Store()
    thought = integrate(mind, frag)
    assert any(r.claim.predicate == "is_a" and r.claim.object == "textbox" for r in thought.added)
    assert mind.claims(predicate="label")[0].evidence[0].confidence is not None


def test_icon_memory_recognizes_the_same_icon_with_different_padding_only():
    img = canvas(400, 200, (40, 40, 48))
    cv2.rectangle(img, (40, 40), (76, 76), (30, 30, 30), -1)
    cv2.putText(img, ">_", (44, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    cv2.circle(img, (200, 58), 18, (230, 120, 20), -1)
    memory = IconMemory()
    assert memory.remember(img, (34, 34, 48, 48), "Terminal")
    assert memory(img, (38, 38, 42, 42))[0] == "Terminal"
    assert memory(img, (180, 38, 42, 42))[0] == ""


def test_shell_reading_uses_what_was_typed_and_fixes_known_ocr_confusions():
    typed = ["ls ~/Desktop # 4d50c5"]
    assert expect_typed("agent@box:-S ls -/Desktop # 4d5Oc5", typed) == "agent@box:~$ ls ~/Desktop # 4d50c5"
    assert expect_typed("hello world", typed) == "hello world"
    assert shell_reading("(in -/Projects) it’s") == "(in ~/Projects) it's"
    assert shell_reading("a-/b stays") == "a-/b stays"


def test_wrapped_command_lines_are_read_together():
    from examples.browser_agents.browser import Text
    from examples.browser_agents.vision.desktop_e2e import rejoin_wrapped

    typed = ["echo 'write smoke tests' > notes/todo.txt && echo 'draft the schema' >> notes/todo.txt"]
    rows = (
        Text("text", "agent@box:~/p$ echo 'write smoke tests' > notes/todo.txt && echo 'draft the sch", "Terminal", (10, 100, 700, 16), 0),
        Text("text", "ema' >> notes/todo.txt", "Terminal", (10, 119, 200, 16), 0),
        Text("text", "agent@box:~/p$", "Terminal", (10, 138, 120, 16), 0),
    )
    out = rejoin_wrapped(rows, typed)
    assert [t.text for t in out] == [f"agent@box:~/p$ {typed[0]}", "agent@box:~/p$"]


def test_prompt_and_command_split_into_two_phrases_are_read_together():
    from examples.browser_agents.browser import Text
    from examples.browser_agents.vision.desktop_e2e import rejoin_wrapped

    typed = ["mkdir -p ~/Projects/orbit-desk-71005/notes"]
    rows = (Text("text", "agent@box:~/p$", "Terminal", (10, 100, 300, 16), 0), Text("text", "mkdir -P -/Projects/orbit-desk-71005/notes", "Terminal", (330, 100, 380, 16), 0))
    assert [t.text for t in rejoin_wrapped(rows, typed)] == [f"agent@box:~/p$ {typed[0]}"]


def test_command_phrase_a_pixel_above_its_prompt_is_still_read_after_it():
    from examples.browser_agents.browser import Text
    from examples.browser_agents.vision.desktop_e2e import rejoin_wrapped

    typed = ["echo 'collect sample data' > notes/todo.txt && echo 'draft the schema' >> notes/todo.txt"]
    rows = (
        Text("text", "echo collect sample data' > notes/todo.txt", "Terminal", (560, 494, 346, 18), 0),
        Text("text", "agent@vision-e2e:~/Projects/ledger-lab-72009$", "Terminal", (185, 495, 362, 17), 0),
        Text("text", "&& echo draft the schema' ' >> notes/todo.txt", "Terminal", (185, 514, 356, 16), 0),
        Text("text", "agent@vision-e2e:~/Projects/ledger-lab-72009$", "Terminal", (185, 534, 364, 17), 0),
    )
    assert [t.text for t in rejoin_wrapped(rows, typed)] == [f"agent@vision-e2e:~/Projects/ledger-lab-72009$ {typed[0]}", "agent@vision-e2e:~/Projects/ledger-lab-72009$"]


def test_prompt_with_colon_misread_is_still_a_prompt():
    from examples.browser_agents.vision.desktop_e2e import expect_typed

    assert expect_typed("agent@vision-e2ei-/Projects/orbit-notes-72008 cd ~/Projects/ledger-lab-72009", ["cd ~/Projects/ledger-lab-72009"]) == "agent@vision-e2e:~/Projects/orbit-notes-72008$ cd ~/Projects/ledger-lab-72009"


def test_prompt_with_a_space_misread_into_the_path():
    from examples.browser_agents.vision.desktop_e2e import expect_typed
    from examples.browser_agents.vision.perceive import PROMPT

    line = "agent@vision-e2e-2:-/Projects/ledger-lab 0-72009$ mkdir -P -/Projects/ledger-lab-72009/notes"
    assert PROMPT.match("agent@vision-e2e-2:-/Projects/ledger-lab 0-72009$")
    assert expect_typed(line, ["mkdir -p ~/Projects/ledger-lab-72009/notes"]) == "agent@vision-e2e-2:~/Projects/ledger-lab0-72009$ mkdir -p ~/Projects/ledger-lab-72009/notes"


def test_contradicted_digit_runs_are_refused_not_guessed():
    from examples.browser_agents.browser import Text
    from examples.browser_agents.vision.desktop_e2e import refuse_uncertain, uncertain_digit_runs

    rows = (
        Text("text", "task-72004.txt", "Terminal", (10, 10, 100, 14), 0),
        Text("text", "cat ~/Desktop/task-72004.txt", "Terminal", (10, 30, 300, 14), 0),
        Text("text", "Set up a project called signal-desk-72804 under ~/Projects", "Terminal", (10, 50, 600, 14), 0),
    )
    assert uncertain_digit_runs(rows) == {"72804"}  # read once, contradicted by a name read twice
    out = refuse_uncertain(rows)
    assert out[2].text.endswith("under ~/Projects") and "signal-desk-?????" in out[2].text
    assert out[0].text == "task-72004.txt"  # the corroborated reading is untouched


def test_agreeing_digit_runs_are_left_alone():
    from examples.browser_agents.browser import Text
    from examples.browser_agents.vision.desktop_e2e import uncertain_digit_runs

    rows = (Text("text", "task-72004.txt", "T", (0, 0, 10, 10), 0), Text("text", "signal-desk-72004", "T", (0, 20, 10, 10), 0))
    assert uncertain_digit_runs(rows) == set()


def test_unsure_digit_words_are_refused_instead_of_guessed():
    from examples.browser_agents.vision.desktop_e2e import mask_low_confidence_digits

    img = canvas()
    words = [Word("task-71008.txt", (10, 10, 90, 14), 0.62, 0), Word("Signal", (10, 40, 44, 14), 0.61, 1), Word("2026", (60, 40, 30, 14), 0.99, 1)]
    scene = perceive(img, ocr=ocr(*words), text_candidates=False)
    texts = {t.text for t in mask_low_confidence_digits(scene, 0.95)}
    assert "??????????????" in texts  # the unsure identifier is refused
    assert "Signal 2026" in texts  # a confident digit word, and non-digit words, are kept
