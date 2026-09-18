"""The learned perception/action implementations: contracts, not accuracy.

Accuracy lives in eval/results (it depends on trained artifacts that are not in the repo).
What is checked here is what the runtime relies on: features are deterministic, a
checkpoint round-trips, the ops satisfy the implementation protocol, ranking is ordered,
and a tie abstains instead of guessing.
"""

from __future__ import annotations

import numpy as np
import pytest

import tensacode as tc
from tensacode.runtime import Implementation
from examples.browser_agents.browser import Control
from examples.browser_agents.learned.chooser import LearnedChooser
from examples.browser_agents.learned.text_embed import LabelEncoder, LearnedLabelMatcher, features, normalize


def tiny_encoder(dim: int = 8, buckets: int = 64, seed: int = 0) -> LabelEncoder:
    rng = np.random.default_rng(seed)
    return LabelEncoder(buckets, dim, rng.standard_normal((buckets, dim)).astype(np.float32),
                        rng.standard_normal((dim, dim)).astype(np.float32), np.zeros(dim, dtype=np.float32))


def control(name: str, role: str = "button") -> Control:
    return Control(role=role, name=name, value="", checked=None, disabled=False, hint="", group="", section="",
                   input_type="", box=(0, 0, 10, 10), point=(5, 5), current=False, shown="")


def test_features_are_deterministic_and_normalisation_is_stable():
    assert features("Staff number", buckets=128) == features("Staff number", buckets=128)
    assert normalize("  Personnel  no. ") == normalize("PERSONNEL NO.")
    assert features("a", buckets=8) != []


def test_encoder_round_trips_and_gives_unit_vectors(tmp_path):
    encoder = tiny_encoder()
    vecs = encoder.encode(["Full name", "Staff number"])
    assert vecs.shape == (2, 8)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    path = tmp_path / "enc.npz"
    encoder.save(path)
    again = LabelEncoder.load(path)
    assert np.allclose(again.encode(["Full name"]), encoder.encode(["Full name"]))
    assert -1.0 <= encoder.similarity("Full name", "Legal name") <= 1.0


def test_label_matcher_is_an_implementation_and_ranks_in_order():
    matcher = LearnedLabelMatcher(tiny_encoder(), source="test")
    assert isinstance(matcher, Implementation)
    candidates = (control("Submit request"), control("Cancel"), control("Send back"))
    request = tc.Request("rank", ("Submit request",), None, {"candidates": candidates})
    assert matcher.accepts(request)
    ranked = matcher.run([request])[0].value
    scores = [s.value for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(s.kind == "learned-similarity" for _, s in ranked)


def test_label_matcher_declines_requests_that_are_not_control_ranking():
    matcher = LearnedLabelMatcher(tiny_encoder(), source="test")
    assert not matcher.accepts(tc.Request("rank", ("x",), None, {"candidates": ("not a control",)}))
    assert not matcher.accepts(tc.Request("classify", "x", None, {}))


class Option:
    """Stand-in for an intention: the chooser only reads the fields it features."""

    def __init__(self, kind: str, why: str, records: int = 0) -> None:
        self.__class__.__name__ = kind
        self.why = why
        self.records = ()
        self.priority = 0.0
        self._records = records

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.why!r})"


def test_chooser_picks_its_argmax_and_reports_a_score():
    dim = 20  # 10 intention kinds + 10 numeric features; the text features need an encoder
    w = np.zeros(dim, dtype=np.float32)
    w[0] = 1.0  # the first feature is the one-hot for "Press"
    chooser = LearnedChooser(w, None, source="test")
    press, wait = type("Press", (Option,), {})("Press", "click submit"), type("Wait", (Option,), {})("Wait", "waiting")
    request = tc.Request("choose", (wait, press), None, {})
    assert chooser.accepts(request)
    out = chooser.run([request])[0]
    assert out.value is press
    assert out.score is not None and out.score.kind == "learned-utility"


def test_chooser_abstains_on_a_tie_rather_than_guessing():
    chooser = LearnedChooser(np.zeros(20, dtype=np.float32), None, source="test", margin=0.0)
    a, b = type("Press", (Option,), {})("Press", "one"), type("Press", (Option,), {})("Press", "two")
    out = chooser.run([tc.Request("choose", (a, b), None, {})])[0]
    assert isinstance(out.value, tc.Unknown) and out.value.reason == "tie_within_margin"


def test_chooser_round_trips_through_a_checkpoint(tmp_path):
    path = tmp_path / "ranker.npz"
    w = np.arange(20, dtype=np.float32)
    np.savez(path, w=w, meta=np.array([20]))
    loaded = LearnedChooser.load(path, source="test")
    assert np.allclose(loaded.w, w)


@pytest.mark.parametrize("kind", ["Press", "Enter", "Wait", "Finish"])
def test_every_intention_kind_featurizes_to_the_same_width(kind):
    from eval.training.rollout_decisions import describe_intention
    from eval.training.train_intention_ranker import featurize

    described = describe_intention(type(kind, (Option,), {})(kind, "why"))
    assert described["kind"] == kind
    assert featurize(described, index=0, n=2, encoder=None).shape == (20,)
