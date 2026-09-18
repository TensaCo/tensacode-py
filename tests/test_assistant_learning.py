"""Compiling a teacher-guided trace into a skill: accept only generalizations that round-trip."""

from examples.browser_agents.assistant import learning as L

ADA_TRACE = [
    {"do": "open_app", "app": "Slack", "effect": False, "ok": True, "screen": []},
    {"do": "fill", "label": "Message agent-runs on slack", "text": "@Ada Kernel I'm happy", "effect": False, "ok": True, "screen": ["Ada Kernel 7:45 AM · mac-studio"]},
    {"do": "click", "label": "Send message", "effect": True, "ok": True, "screen": ["Ada Kernel 7:45 AM · mac-studio"]},
]


def test_trace_generalizes_over_slots_with_name_lookup_and_verification():
    skill, why = L.compile_skill("tell ada im happy", ADA_TRACE, "Posted it.", "tell {person} {message}", {"person": "ada", "message": "im happy"})
    assert skill is not None, why
    assert [s["do"] for s in skill.steps] == ["open_app", "find_name", "fill", "click", "expect"]
    assert skill.steps[2]["text"] == "@{person_name} {message|sentence}"
    assert skill.steps[3]["effect"] is True
    assert L.fill(skill.steps[2]["text"], {"person_name": "Maya Chen", "message": "dont be late"}) == "@Maya Chen Don't be late"


def test_pattern_must_reproduce_the_request_slots():
    skill, why = L.compile_skill("tell ada im happy", ADA_TRACE, "", "message {person} {message}", {"person": "ada", "message": "im happy"})
    assert skill is None and "does not reproduce" in why


def test_a_slot_the_steps_never_use_is_rejected():
    trace = [{"do": "open_app", "app": "Slack", "effect": False, "ok": True, "screen": []}, {"do": "click", "label": "Send message", "effect": True, "ok": True, "screen": []}]
    skill, why = L.compile_skill("tell ada im happy", trace, "", "tell {person} {message}", {"person": "ada", "message": "im happy"})
    assert skill is None and "never use" in why


def test_skill_matching_prefers_specific_patterns():
    lib = L.Library(path=__import__("pathlib").Path("/nonexistent/skills.json"))
    general = L.Skill("a", "tell {person} {message}", ["person", "message"], [], "", {})
    specific = L.Skill("b", "tell {person} on slack {message}", ["person", "message"], [], "", {})
    lib.skills = [general, specific]
    hit, slots = lib.match("tell ada on slack the build is green")
    assert hit.id == "b" and slots == {"person": "ada", "message": "the build is green"}
    assert lib.match("tell ada") is None


def test_sentence_tidies_casual_text_only():
    assert L.sentence("im happy") == "I'm happy"
    assert L.sentence("Build is GREEN") == "Build is GREEN"
