"""Wants, frames and priming — and the four chat failures that motivated all of it."""

from datetime import datetime, timezone

import tensacode as tc
from tensacode.awareness import AwarenessPolicy, nucleate
from tensacode.frames import Frames, Place, modality_of
from tensacode.memory import Memory
from tensacode.outcomes import Unknown
from tensacode.priming import Cue, Priming, prime, primeable_from
from tensacode.wants import Answer, Satisfier, Want, Wants, want_from

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def seen(mind, subject, predicate, obj, *, source="obs:frame-1", method="dom-scene-graph"):
    return mind.tell(tc.Claim(tc.Ref(subject), predicate, obj), tc.Evidence(tc.Ref(source), NOW, method=method))


# ------------------------------------------------------------------- wants


def test_a_want_ranks_its_satisfiers_by_worth_and_names_the_next_move():
    wants = Wants()
    look = Satisfier("perceive", "look at the dock", cost=1.0, odds=tc.Score(0.9, "uncalibrated"))
    ask = Satisfier("ask", "ask the user", cost=10.0, odds=tc.Score(1.0, "uncalibrated"))
    wants.add(Want("how many icons are in the sidebar?", satisfiers=(ask, look)))

    want, satisfier = wants.next_to_pursue()

    assert satisfier is look, "looking is cheap and likely; asking the user is a last resort"
    assert "look at the dock" in wants.as_unknown(want).detail


def test_a_want_is_answered_from_memory_and_cites_the_claims():
    mind = tc.Store()
    memory = Memory(mind)
    memory.told(tc.Ref("person:user"), "name", "Jacob")
    wants = Wants()
    want = Want("what is my name?", subject=tc.Ref("person:user"), predicate="name")

    answer = wants.look_up(want, mind)

    assert isinstance(answer, Answer) and answer.value == "Jacob"
    assert answer.claims and answer.modality == ("hearsay",)


def test_memory_that_disagrees_yields_no_answer_and_offers_both():
    mind = tc.Store()
    seen(mind, "display:main", "color_depth", "24-bit")
    seen(mind, "display:main", "color_depth", "32-bit", source="obs:frame-2")
    wants = Wants()

    found = wants.look_up(Want("what colour depth?", subject=tc.Ref("display:main"), predicate="color_depth"), mind)

    assert isinstance(found, Unknown) and found.reason == "disagreement"
    assert len(found.candidates) == 2, "both readings are offered as candidates, neither as the answer"


def test_a_want_that_needs_an_observation_refuses_hearsay():
    mind = tc.Store()
    memory = Memory(mind)
    memory.told(tc.Ref("window:Files"), "item_count", 7)
    wants = Wants()

    found = wants.look_up(Want("how many items?", subject=tc.Ref("window:Files"), predicate="item_count",
                               requires="observed"), mind)

    assert isinstance(found, Unknown) and found.reason == "provenance_unmet"
    assert "hearsay" in found.detail

    seen(mind, "window:Files", "item_count", 7)
    assert isinstance(wants.look_up(Want("how many items?", subject=tc.Ref("window:Files"),
                                         predicate="item_count", requires="observed"), mind), Answer)


def test_an_unknown_becomes_a_want_carrying_its_candidates():
    unknown = Unknown("no_implementation", "nothing can parse that", candidates=(("app:code", tc.Score(0.4, "uncalibrated")),))

    want = want_from(unknown, "which app writes code?", satisfiers=(Satisfier("ask", "ask the user", cost=5.0),))

    assert want.question == "which app writes code?"
    assert [s.kind for s in want.satisfiers] == ["ask", "derive"], "its candidates survive as things to reconsider"


def test_open_wants_are_settled_by_memory_and_the_rest_stay_open():
    mind = tc.Store()
    Memory(mind).told(tc.Ref("person:user"), "name", "Jacob")
    wants = Wants()
    wants.add(Want("what is my name?", subject=tc.Ref("person:user"), predicate="name"))
    unanswerable = wants.add(Want("what colour is the display?", subject=tc.Ref("display:main"), predicate="color"))

    answered = wants.pursue_from_memory(mind)

    assert [a.value for a in answered] == ["Jacob"]
    assert list(wants.open) == [unanswerable], "what memory cannot settle is still wanted, not guessed"


# ------------------------------------------------------------------ frames


def test_the_same_claim_is_reachable_semantically_spatially_and_by_modality():
    class Control:
        def __init__(self, box, section):
            self.box, self.section = box, section

    mind = tc.Store()
    mind.put(tc.Ref("ui:dock/button/Code"), Control((8, 420, 40, 40), "Dock"))
    rec = seen(mind, "ui:dock/button/Code", "label", "Visual Studio Code", method="atspi")
    frames = Frames(mind)

    assert rec.id in {r.id for r in frames.semantic(predicate="label")}
    assert rec.id in {r.id for r in frames.spatial(window="Dock")}
    assert rec.id in {r.id for r in frames.spatial(near=(20, 430), slack=40)}
    assert rec.id in {r.id for r in frames.modality("structure")}
    binding = frames.binding(rec.id)
    assert binding.place.window == "Dock" and binding.modality == ("structure",)


def test_modality_is_read_off_evidence_not_declared():
    mind = tc.Store()
    ocr = seen(mind, "text:Terminal#1", "reads", "atlas-sync-31288", source="obs:pixels", method="ocr@crnn")
    tree = seen(mind, "ui:field#1", "value", "atlas-sync-31288", source="obs:scene", method="dom-scene-graph")
    told = mind.tell(tc.Claim(tc.Ref("person:ada"), "mood", "happy"), tc.Evidence(tc.Ref("said:user"), NOW, method="told"))

    assert modality_of(ocr) == ("pixels",)
    assert modality_of(tree) == ("structure",)
    assert modality_of(told) == ("hearsay",)


def test_two_modalities_disagreeing_stays_visible():
    mind = tc.Store()
    seen(mind, "project:current", "name", "atlas-sync-31288", source="obs:scene", method="dom-scene-graph")
    seen(mind, "project:current", "name", "atlas-sync-31788", source="obs:pixels", method="ocr@crnn")
    frames = Frames(mind)

    clashes = frames.disagreements()

    assert len(clashes) == 1
    described = clashes[0].describe()
    assert "structure" in described and "pixels" in described
    assert "31288" in described and "31788" in described


def test_spatial_adjacency_carries_awareness_across_modalities():
    """Nucleation crossing frames: a pixel reading beside a structural one comes along."""
    class Box:
        def __init__(self, box, section):
            self.box, self.section = box, section

    mind = tc.Store()
    mind.put(tc.Ref("ui:field#1"), Box((100, 100, 200, 20), "Terminal"))
    mind.put(tc.Ref("text:ocr#1"), Box((104, 130, 200, 20), "Terminal"))
    field = seen(mind, "ui:field#1", "value", "", source="obs:scene", method="dom-scene-graph")
    ocr = seen(mind, "text:ocr#1", "reads", "agent@assistant:~$", source="obs:pixels", method="ocr@crnn")
    frames = Frames(mind)

    without = nucleate(mind, [field.id], AwarenessPolicy(budget=8, floor=0.2, max_hops=1))
    with_frames = nucleate(mind, [field.id], AwarenessPolicy(budget=8, floor=0.2, max_hops=1),
                           extra_links=frames.links)

    assert ocr.id not in without.aware_ids(), "nothing in the claim graph links these two"
    assert ocr.id in with_frames.aware_ids(), "being 30 pixels apart does"
    assert any("--spatial-->" in line for line in with_frames.why(ocr.id))


def test_place_distance_is_none_when_either_side_is_unplaced():
    assert Place(window="Dock").distance(Place()) is None
    assert Place(box=(0, 0, 10, 10)).near((5, 5), slack=1.0)


# ----------------------------------------------------------------- priming


def test_a_procedure_with_an_act_is_primed_by_the_act_claim_reproducing_dispatch():
    class Proc:
        def __init__(self, id, act):
            self.id, self.act = id, act

    mind = tc.Store()
    seen(mind, "request:1.0", "act", "list", source="utterance:1", method="parse")
    aware = nucleate(mind, [tc.Ref("request:1.0")], AwarenessPolicy(budget=8))

    priming = prime([Proc("list_dir", "list"), Proc("delete_path", "delete")], aware, threshold=0.5)
    ready = priming.ready()

    assert [p.id for p, _ in ready] == ["list_dir"]
    assert "matched" in "\n".join(priming.why("list_dir"))
    assert priming.partial()["list_dir"] > 0


def test_several_procedures_are_partly_warm_at_once():
    mind = tc.Store()
    seen(mind, "path:/home/agent/Desktop/notes.txt", "is_a", "regular file")
    seen(mind, "path:/home/agent/Desktop/notes.txt", "size", 120)
    aware = nucleate(mind, [tc.Ref("path:/home/agent/Desktop/notes.txt")], AwarenessPolicy(budget=8))

    reading = primeable_from(type("P", (), {"id": "read_file", "cues": (Cue(predicate="is_a", object="regular file"),)})())
    sizing = primeable_from(type("P", (), {"id": "size_of", "cues": (Cue(predicate="size"),)})())
    priming = Priming([reading, sizing], decay=0.5)
    priming.observe(aware)

    warm = priming.partial()
    assert set(warm) == {"read_file", "size_of"}, "both are candidates, neither was selected"


def test_a_chain_needs_its_stages_in_order_and_cools_if_they_stall():
    mind = tc.Store()
    terminal = seen(mind, "ui:Shell input", "is_a", "textbox")
    chain = primeable_from(type("P", (), {
        "id": "read_command_output",
        "chain": ((Cue(predicate="is_a", object="textbox"),), (Cue(predicate="command"),), (Cue(predicate="prompt_returned"),)),
        "prime_threshold": 0.2,
        "window": 1,
    })())
    priming = Priming([chain], decay=0.9)

    priming.observe(nucleate(mind, [terminal.id], AwarenessPolicy(budget=6)))
    assert priming.ready() == [], "one stage is not the chain"
    assert priming.stage["read_command_output"] == 1

    typed = seen(mind, "command:1", "command", "ls -1pA ~")
    priming.observe(nucleate(mind, [typed.id], AwarenessPolicy(budget=6)))
    assert priming.stage["read_command_output"] == 2 and priming.ready() == []

    done = seen(mind, "command:1", "prompt_returned", True)
    priming.observe(nucleate(mind, [done.id], AwarenessPolicy(budget=6)))
    assert [p.id for p, _ in priming.ready()] == ["read_command_output"], "in order, it fires"
    assert "chain stage 3 of 3" in "\n".join(priming.why("read_command_output"))


def test_a_chain_that_stalls_falls_back_to_the_beginning():
    mind = tc.Store()
    terminal = seen(mind, "ui:Shell input", "is_a", "textbox", source="obs:frame-1")
    elsewhere = seen(mind, "ui:toast#1", "announces", "Saved", source="obs:frame-2")
    chain = primeable_from(type("P", (), {
        "id": "typing_sequence",
        "chain": ((Cue(predicate="is_a", object="textbox"),), (Cue(predicate="command"),)),
        "window": 1,
    })())
    priming = Priming([chain])

    priming.observe(nucleate(mind, [terminal.id], AwarenessPolicy(budget=4)))
    assert priming.stage["typing_sequence"] == 1

    mind.apply(tc.Patch((tc.Retract(terminal.id, "window closed"),), mind.revision))
    for _ in range(3):  # the terminal is gone and no command was ever typed
        priming.observe(nucleate(mind, [elsewhere.id], AwarenessPolicy(budget=4)))

    assert priming.stage["typing_sequence"] == 0, "the chain cooled and starts over"
    assert priming.ready() == [], "a half-finished chain never fires"


def test_a_stalled_chain_rearms_while_its_first_stage_is_still_true():
    """Cooling is not forgetting: if the shell input is still there, stage one holds again."""
    mind = tc.Store()
    terminal = seen(mind, "ui:Shell input", "is_a", "textbox")
    chain = primeable_from(type("P", (), {
        "id": "typing_sequence",
        "chain": ((Cue(predicate="is_a", object="textbox"),), (Cue(predicate="command"),)),
        "window": 1,
    })())
    priming = Priming([chain])

    for _ in range(4):
        priming.observe(nucleate(mind, [terminal.id], AwarenessPolicy(budget=4)))

    assert priming.stage["typing_sequence"] == 1, "it waits at the stage it can satisfy"
    assert priming.ready() == []


# -------------------------------- the failures that motivated the design


def test_the_name_exchange_that_failed_now_works_end_to_end():
    """'my name is Jacob. what is my name?' — tell lands as a claim, ask reads it back."""
    mind = tc.Store()
    memory = Memory(mind)
    wants = Wants()

    memory.told(tc.Ref("person:user"), "name", "Jacob")  # "my name is Jacob"
    want = Want("what is my name?", subject=tc.Ref("person:user"), predicate="name",
                satisfiers=(Satisfier("memory", "look in what I was told", cost=0.1, odds=tc.Score(0.9, "uncalibrated")),))
    wants.add(want)
    answer = wants.look_up(want, mind)

    assert isinstance(answer, Answer) and answer.value == "Jacob"
    assert answer.modality == ("hearsay",), "and it knows it knows this only because you said so"


def test_how_many_icons_is_answerable_from_what_was_perceived():
    """The answer was in the graph the whole time; nothing had to act on the machine."""
    class Control:
        def __init__(self, box, section):
            self.box, self.section = box, section

    mind = tc.Store()
    for i, name in enumerate(["Files", "Firefox", "Terminal", "Text Editor", "Slack"]):
        mind.put(tc.Ref(f"ui:dock/button/{name}"), Control((8, 100 + i * 48, 40, 40), "Dock"))
        seen(mind, f"ui:dock/button/{name}", "label", name, source="obs:scene", method="atspi")
        seen(mind, f"ui:dock/button/{name}", "is_a", "button", source="obs:scene", method="atspi")
    frames = Frames(mind)

    in_dock = [r for r in frames.spatial(window="Dock") if r.claim.predicate == "is_a"]

    assert len(in_dock) == 5
    assert {r.claim.subject.id.rsplit("/", 1)[-1] for r in in_dock} == {"Files", "Firefox", "Terminal", "Text Editor", "Slack"}


def test_an_unanswerable_question_stays_a_want_rather_than_becoming_an_action():
    """'what color is the display' has no answer here — and must not turn into a listing."""
    mind = tc.Store()
    seen(mind, "path:/home/agent", "is_a", "directory")  # the home folder the old code listed
    wants = Wants()
    want = Want("what colour is the display?", subject=tc.Ref("display:main"), predicate="color",
                satisfiers=(Satisfier("perceive", "look at the screen in pixels", cost=2.0,
                                      odds=tc.Score(0.3, "uncalibrated")),))
    wants.add(want)

    found = wants.look_up(want, mind)

    assert isinstance(found, Unknown) and found.reason == "not_in_memory"
    assert want.id in wants.open, "it stays wanted"
    assert wants.next_to_pursue()[1].kind == "perceive", "and the next move is to look, not to list a folder"
