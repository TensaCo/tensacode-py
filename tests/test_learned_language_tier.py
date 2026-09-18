"""The learned parser and answerer as runtime implementations.

Span decoding and the label space are tested without torch. The rest is skipped unless the
trained artifacts are present, so the suite stays green on a machine that has neither.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eval.training import schema as S

SP = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))
PARSER = SP / "artifacts" / "request-parser"
ANSWERER = SP / "artifacts" / "span-answerer"
torch = pytest.importorskip("torch", reason="the learned tier needs the neural extra")


def test_label_space_covers_every_act_a_procedure_exists_for():
    """If the assistant grows an act, training must know about it rather than silently mislabel."""
    procedures = (Path(__file__).parents[1] / "examples/browser_agents/assistant/procedures.py").read_text()
    declared = {line.split('act="')[1].split('"')[0] for line in procedures.splitlines() if 'act="' in line}
    assert declared - set(S.ACTS) == set(), f"acts with no place in the label space: {sorted(declared - set(S.ACTS))}"


def test_every_span_slot_has_both_bio_tags():
    for slot in S.SPAN_SLOTS:
        assert f"B-{slot}" in S.TAG_INDEX and f"I-{slot}" in S.TAG_INDEX


def test_span_decoding_recovers_characters_and_stops_at_o():
    from tensorcode.backends.neural import _decode_spans

    text = "make a folder called recipes on my desktop"
    #          0    5 7      14     21      29 32 35
    offsets = [[0, 0], [0, 4], [5, 6], [7, 13], [14, 20], [21, 28], [29, 31], [32, 34], [35, 42]]
    tags = ["O", "B-name", "I-name"]
    one = _decode_spans(tags, [0, 0, 0, 0, 0, 1, 0, 0, 0], offsets, text, [1] * 9)
    assert one == {"name": "recipes"}
    # B then I is one span; the following O closes it
    two = _decode_spans(tags, [0, 0, 0, 0, 0, 1, 2, 0, 0], offsets, text, [1] * 9)
    assert two == {"name": "recipes on"}
    # a second B after an O starts nothing new for a slot already seen: the first run wins
    again = _decode_spans(tags, [0, 1, 0, 0, 0, 1, 0, 0, 0], offsets, text, [1] * 9)
    assert again == {"name": "make"}


def test_a_threshold_fit_that_cannot_reach_its_target_says_so():
    """The first version silently returned 1.0, which refuses every item and looks deliberate."""
    import numpy as np

    from eval.training.calibration import TargetUnreachable, fit_threshold

    # a model that is right a quarter of the time can never reach 90% selective accuracy
    confidence = np.linspace(0.9, 0.1, 40)
    correct = np.array([1.0, 0.0, 0.0, 0.0] * 10)
    fit = fit_threshold(confidence, correct, target=0.9)
    assert fit.reachable is False
    assert fit.threshold == 0.0 and not fit.refuses_everything, "it must fall back to answering, not to silence"
    assert "no threshold reached" in fit.note and f"{fit.best_selective_accuracy:.3f}" in fit.note
    with pytest.raises(TargetUnreachable):
        fit_threshold(confidence, correct, target=0.9, strict=True)

    # a reachable target gives a real operating point, and one lucky item is not one
    ranked = np.array([1.0] * 20 + [0.0] * 20)
    good = fit_threshold(np.linspace(1.0, 0.0, 40), ranked, target=0.9)
    # the lowest threshold still meeting the target, so it answers a little past the correct run
    assert good.reachable and good.coverage_at_threshold == pytest.approx(0.55)
    one_lucky = fit_threshold(np.linspace(1.0, 0.0, 40), np.array([1.0] + [0.0] * 39), target=0.9)
    assert one_lucky.reachable is False, "2.5% coverage is not a calibrated operating point"


def test_a_statement_about_the_world_is_read_as_a_speech_act_not_an_act():
    from eval.training.schema import speech_act_of

    for statement in ("Nise is hungry.", "Resource:wood is cheap.", "the north field failed",
                      "I heard the mill is broken.", "Mara owes me two sacks.",
                      "Anem said wood is dear.", "they say the ferry is late"):
        assert speech_act_of(statement, "unknown") == "world_statement", statement
    assert speech_act_of("my name is Jacob", "tell") == "self_disclosure"
    assert speech_act_of("delete the recipes folder", "delete") == "command"
    assert speech_act_of("how many icons are in the sidebar", "ask_screen") == "question"
    # a request that mentions telling someone is still a request
    assert speech_act_of("tell ada im happy", "unknown") == "command"


@pytest.mark.skipif(not (PARSER / "weights.pt").exists(), reason="no trained parser artifact")
def test_the_parser_refuses_a_statement_it_could_have_read_as_a_request():
    from tensorcode.backends.neural import NeuralRequestParser

    parser = NeuralRequestParser(PARSER, device="cpu")
    for statement in ("Nise is hungry.", "wood is cheap.", "the north field failed"):
        got = parser.parse([statement])[0]
        assert got.speech_act == "world_statement" and got.act == "unknown", f"{statement} -> {got!r}"
    asked = parser.parse(["delete notes.txt"])[0]
    assert asked.speech_act == "command" and asked.act == "delete"


@pytest.mark.skipif(not (PARSER / "weights.pt").exists(), reason="no trained parser artifact")
def test_the_parser_answers_through_the_runtime_and_abstains_off_domain():
    import tensorcode as tc
    from tensorcode.backends.neural import NeuralRequestParser, ParsedRequest

    parser = NeuralRequestParser(PARSER, device="cpu")
    runtime = tc.Runtime([parser], policy=tc.Policy(localities=frozenset({"in_process"}), cache=False))
    with tc.use(runtime):
        got = tc.parse("make a folder called recipes on my desktop", ParsedRequest)
        assert isinstance(got, ParsedRequest) and got.act == "create_folder"
        assert got.slots["name"] == "recipes" and got.slots["place"] == "~/Desktop"
        refused = tc.parse("who won the world cup in 1998", ParsedRequest)
        assert isinstance(refused, tc.Unknown)
    answered = [s for s in runtime.trace.of("parse") if s.answered_by]
    assert answered and answered[0].answered_by.startswith("neural-request-parser")


@pytest.mark.skipif(not (PARSER / "weights.pt").exists(), reason="no trained parser artifact")
def test_the_artifact_carries_its_own_label_space():
    import json

    config = json.loads((PARSER / "config.json").read_text())
    assert config["acts"] == list(S.ACTS) and config["tags"] == list(S.TAGS)
    assert config["parameters"] > 0 and config["train_size"] > 0


@pytest.mark.skipif(not (ANSWERER / "weights.pt").exists(), reason="no trained answerer artifact")
def test_the_answerer_extracts_a_span_and_refuses_what_is_not_there():
    import tensorcode as tc
    from tensorcode.backends.neural import Answer, NeuralAnswerer, QuestionOverPassages

    answerer = NeuralAnswerer(ANSWERER, device="cpu")
    runtime = tc.Runtime([answerer], policy=tc.Policy(localities=frozenset({"in_process"}), cache=False))
    passages = (("Ada", "Ada Lovelace worked with Charles Babbage on the Analytical Engine."),
                ("Ada", "She is often called the first computer programmer."))
    with tc.use(runtime):
        got = tc.parse(QuestionOverPassages("Who did Ada Lovelace work with?", passages), Answer)
        assert isinstance(got, Answer) and "Babbage" in got.text
        missing = tc.parse(QuestionOverPassages("What is the population of Denmark?", passages), Answer)
        assert isinstance(missing, (tc.Unknown, Answer))


# --------------------------------------------------------- the generator's own labels

def test_the_act_label_is_the_verb_the_utterance_uses():
    """A label drawn independently of the surface verb mislabels half the rows it touches.

    One template used to pick "move "/"copy " for the text and move/copy for the label with two
    separate draws, so about half of those rows said one thing and were labelled the other. The
    noise sat in training and evaluation alike, so it showed up as neither a training loss nor an
    evaluation error -- only as a capability that silently was not there.
    """
    import re

    from eval.training.parser_data import generate_templates

    move = re.compile(r"\b(move|mv)\b")
    copy = re.compile(r"\b(copy|cp|duplicate)\b")
    wrong = []
    for ex in generate_templates(8000, seed=0):
        if ex.act not in ("move", "copy"):
            continue
        bare = re.sub(r"(['\"“][^'\"”]*['\"”])", " ", ex.text)  # a verb inside a quote is content
        said_move, said_copy = bool(move.search(bare)), bool(copy.search(bare))
        if said_move ^ said_copy:  # exactly one verb present: the label is then determined
            if ("move" if said_move else "copy") != ex.act:
                wrong.append((ex.text, ex.act))
    assert not wrong, f"{len(wrong)} rows are labelled with a verb they do not use, e.g. {wrong[:3]}"


def test_a_place_phrase_does_not_run_into_the_span_before_it():
    """"...called desktopon ~/Documents" taught the parser to split a word no user will type."""
    import re

    from eval.training.parser_data import generate_templates

    # A standalone preposition is recognised by what follows it: a path, or "the"/"my" plus a folder.
    # (A bare "<word>on " also matches real words like "mention", so the tail is what makes it a place.)
    patterns = (re.compile(r"""[A-Za-z0-9'"”](on|in|into|under|inside) (~|/)"""),
                re.compile(r"""[A-Za-z0-9'"”](in|on) (the|my) """))
    glued = [ex.text for ex in generate_templates(8000, seed=0)
             if any(pat.search(ex.text) for pat in patterns)]
    assert not glued, f"{len(glued)} utterances glue a place phrase to what precedes it, e.g. {glued[:3]}"


def test_recorded_spans_still_land_on_their_values():
    """Whatever the assembler does about separators, the offsets must stay exact."""
    from eval.training.parser_data import generate_templates

    for ex in generate_templates(4000, seed=1):
        for slot, (start, end) in ex.spans.items():
            assert ex.text[start:end] == ex.value(slot), (ex.text, slot)


def test_measured_utterances_are_kept_out_of_the_training_data():
    """Both corpora come from one small vocabulary, so collisions are the default, not the exception.

    Thirty percent of the 152-case benchmark appeared verbatim in the training data before this
    filter existed, which made that row part memorisation score.
    """
    from eval.training.parser_data import (drop_contaminated, generate_templates, held_out_utterances)

    root = Path(__file__).parents[1]
    held_out = held_out_utterances(root)
    assert len(held_out) > 50, "the held-out sets should contribute many candidate utterances"
    rows = generate_templates(20000, seed=0)
    kept, report = drop_contaminated(rows, held_out)
    assert report["rows_dropped"] > 0, "the collision this guards against does happen"
    assert len(kept) == len(rows) - report["rows_dropped"]
    assert not [r for r in kept if r.text.strip().lower().rstrip("?.!").strip() in held_out]


def test_every_act_in_the_label_space_can_name_its_slots():
    """Every slot an act declares must be something the model actually has a head for."""
    expressible = (set(S.SPAN_SLOTS) | set(S.FLAGS) | {name for name, _ in S.CLOSED_HEADS}
                   | set(S.WHOLE_INPUT_SLOTS) | {"place"})
    for act in S.ACTS:
        assert S.slots_of(act) is not None
        assert set(S.slots_of(act)) <= expressible, f"{act} declares a slot with no head"
