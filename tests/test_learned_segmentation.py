"""Learned boundaries preserve source evidence even when search is uncertain."""
import hashlib
import json

import pytest

from tensorcode.language.segmentation import Segmenter, load_model, save_model


@pytest.fixture
def segmenter():
    return Segmenter.train([
        ("can't", ((0, 2), (2, 5))),
        ("won't", ((0, 2), (2, 5))),
        ("hello!", ((0, 5), (5, 6))),
        ("hello", ((0, 5),)),
    ] * 4, epochs=8, seed=0)


def test_boundaries_are_learned_from_supplied_spans(segmenter):
    assert segmenter.segment("can't").candidates[0].spans == ((0, 2), (2, 5))
    assert segmenter.segment("hello!").candidates[0].spans == ((0, 5), (5, 6))
    other = Segmenter.train([("can't", ((0, 5),))] * 8, epochs=8)
    assert other.segment("can't").candidates[0].spans == ((0, 5),)


def test_all_alternatives_cover_original_unicode_source(segmenter):
    text = "  I’m\t'raw'…\n🙂x  "
    result = segmenter.segment(text, beam_width=16, max_candidates=16)
    assert result.candidates
    assert len({candidate.spans for candidate in result.candidates}) == len(result.candidates)
    for candidate in result.candidates:
        covered = [i for start, end in candidate.spans for i in range(start, end)]
        assert covered == [i for i, char in enumerate(text) if not char.isspace()]
        assert all(not any(c.isspace() for c in text[start:end]) for start, end in candidate.spans)
        assert candidate.score_kind == "uncalibrated"
        assert candidate.provenance == ("learned:character-boundaries",)


def test_binary_search_retains_both_boundaries_without_claiming_certainty():
    model = Segmenter.train([("x", ((0, 1),))])
    result = model.segment("ab", beam_width=2, max_candidates=2)
    assert {candidate.spans for candidate in result.candidates} == {((0, 2),), ((0, 1), (1, 2))}
    assert result.expansions == 2
    assert result.complete and not result.truncated
    assert result == model.segment("ab", beam_width=2, max_candidates=2)


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_exhausted_budget_never_completes_an_unseen_suffix(segmenter, budget):
    result = segmenter.segment("abcdef", max_expansions=budget)
    assert result.expansions <= budget
    assert result.candidates == ()
    assert result.truncated and not result.complete
    assert result.reason == "budget_exhausted"


def test_final_boundary_budget_can_retain_complete_partial_search(segmenter):
    result = segmenter.segment("ab", max_expansions=1)
    assert len(result.candidates) == 1
    assert result.candidates[0].spans[-1][1] == 2
    assert result.truncated and result.reason == "budget_exhausted"


@pytest.mark.parametrize("text,spans", [("", ()), (" \t\n", ()), (" x y ", ((1, 2), (3, 4)))])
def test_structural_gaps_need_no_model_expansion(segmenter, text, spans):
    result = segmenter.segment(text, max_expansions=0)
    assert result.candidates[0].spans == spans
    assert result.complete and result.expansions == 0


def test_pruning_and_output_caps_are_explicit(segmenter):
    assert segmenter.segment("abc", beam_width=1).reason == "beam_pruned"
    result = segmenter.segment("ab", beam_width=2, max_candidates=1)
    assert result.reason == "candidate_limit" and not result.complete


@pytest.mark.parametrize("spans", [((0, 1),), ((0, 4),), ((0, 2), (1, 3)), ((True, 3),), ((2, 3), (0, 2))])
def test_invalid_gold_spans_do_not_train(spans):
    with pytest.raises(ValueError):
        Segmenter.train([("abc", spans)])


def test_training_rejects_token_whitespace_and_empty_corpus():
    with pytest.raises(ValueError, match="whitespace"):
        Segmenter.train([("a b", ((0, 3),))])
    with pytest.raises(ValueError, match="requires"):
        Segmenter.train([])


def test_untrained_missing_and_corrupt_models_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="no trained model"):
        Segmenter().segment("input")
    path = tmp_path / "model.json"
    with pytest.raises(FileNotFoundError):
        load_model(path)
    path.write_text("not json")
    with pytest.raises(ValueError, match="invalid segmentation artifact"):
        load_model(path)


def test_json_artifact_roundtrip_retains_predictions_and_provenance(segmenter, tmp_path):
    path = tmp_path / "model.json"
    save_model(path, segmenter, {"training_source": "authored-test-fixture"})
    restored = load_model(path)
    assert restored.segment("can't") == segmenter.segment("can't")
    assert restored.metadata["training_source"] == "authored-test-fixture"
    assert restored.metadata["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", [
    lambda doc: doc.update(version=True),
    lambda doc: doc.update(version=2),
    lambda doc: doc.update(labels=["word", "space"]),
    lambda doc: doc.update(weights={"feature": {"join": float("inf")}}),
    lambda doc: doc.update(weights={"feature": {"guess": 1}}),
    lambda doc: doc.update(weights={"feature": {"start": True}}),
])
def test_invalid_artifacts_are_rejected(segmenter, tmp_path, mutation):
    path = tmp_path / "model.json"
    save_model(path, segmenter)
    document = json.loads(path.read_text())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_model(path)


def test_invalid_save_preserves_existing_artifact(segmenter, tmp_path):
    path = tmp_path / "model.json"
    save_model(path, segmenter)
    original = path.read_bytes()
    segmenter.model.weights["corrupt"] = {"join": float("nan")}
    with pytest.raises(ValueError):
        save_model(path, segmenter)
    assert path.read_bytes() == original
