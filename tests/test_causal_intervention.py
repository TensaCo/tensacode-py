"""Causation: earned by intervention, kept apart from co-occurrence."""

from tensorcode.causal import (Causal, Contrast, causes_of, correlations, counterfactual, distinguish,
                              experiment, learn, moved, tell_causal)
from tensorcode.outcomes import Score
from tensorcode.records import Ref, Store


class Bulb:
    """A toy world with a confound: the switch lights the bulb, the clock ticks regardless.

    Ground truth by construction — the switch causes ``lit``; ``tick`` advances whenever the
    world steps, whether or not the switch was touched. A reader that only watches will see
    both move together every time the switch is flipped.
    """

    def __init__(self, tick: int = 0) -> None:
        self.lit, self.tick = False, tick

    def copy(self) -> "Bulb":
        return Bulb(self.tick)

    def step(self, flip: bool) -> None:
        self.tick += 1  # the confound: time passes on every step
        if flip:
            self.lit = True

    def observe(self) -> dict[str, object]:
        return {"lit": self.lit, "tick": self.tick}


def test_intervention_separates_the_cause_from_the_confound():
    world = Bulb()

    def prepare() -> Bulb:
        return world.copy()

    # both branches take a step, so "time passed" happens either way and cancels out;
    # only flipping the switch differs
    contrasts = experiment(
        prepare=prepare,
        act=lambda w: w.step(flip=True),
        observe=lambda w: w.observe(),
        cause="flip:switch",
        trials=3,
        settle=None,
    )
    by_aspect = {c.aspect: c for c in contrasts}
    # against a do-nothing control the clock looks caused too, which is why the control matters
    assert by_aspect["lit"].effect == 1.0
    assert by_aspect["tick"].effect == 1.0

    controlled = experiment(
        prepare=prepare,
        act=lambda w: w.step(flip=True),
        control=lambda w: w.step(flip=False),  # the control acts too, so "time passed" cancels
        observe=lambda w: w.observe(),
        cause="flip:switch",
        trials=3,
    )
    effects = {c.aspect: c.effect for c in controlled}
    assert effects["lit"] == 1.0, effects
    assert effects["tick"] == 0.0, effects  # both branches ticked; the switch did not cause it

    links = learn(controlled, least_effect=0.5)
    assert [(link.aspect, link.support) for link in links] == [("lit", "interventional")]


def test_co_occurrence_credits_the_confound_and_cannot_tell():
    """The baseline intervention has to beat: watching gives the switch credit for the clock."""
    observations = []
    world = Bulb()
    for _ in range(5):
        before = world.observe()
        world.step(flip=True)
        after = world.observe()
        observations.append(({"flip:switch"}, moved(before, after)))
    seen = correlations(observations, cause="flip:switch", least=0.5)
    aspects = {link.aspect for link in seen}
    # the watcher gets it wrong in both directions: it credits the clock, which moves on
    # every step, and misses the bulb, whose effect saturates after the first flip
    assert aspects == {"tick"}, aspects
    assert all(link.support == "observational" for link in seen)

    controlled = experiment(prepare=Bulb, act=lambda w: w.step(flip=True),
                            control=lambda w: w.step(flip=False),
                            observe=lambda w: w.observe(), cause="flip:switch", trials=3)
    verdicts = distinguish(controlled, seen)
    assert verdicts["tick"] == "merely_correlated"
    assert verdicts["lit"] == "caused"  # intervention finds what watching missed


def test_an_aspect_the_act_never_moves_produces_no_link():
    contrasts = experiment(prepare=Bulb, act=lambda w: w.step(flip=False),
                           control=lambda w: w.step(flip=False),
                           observe=lambda w: w.observe(), cause="flip:nothing", trials=2)
    assert learn(contrasts, least_effect=0.5) == []


def test_causal_claims_say_how_they_were_learned():
    mind = Store()
    interventional = Causal("flip:switch", "lit", True, "interventional", Score(1.0, "probability", basis="intervention@n=3"), 3)
    observational = Causal("flip:switch", "tick", 4, "observational", Score(1.0, "probability", basis="co-occurrence@n=5"), 5)
    for link in (observational, interventional):
        tell_causal(mind, link, source=Ref("obs:experiment"))
    assert [r.claim.object for r in mind.claims(interventional.ref, "support")] == ["interventional"]
    # the interventional link is offered first, so a caller that wants the better evidence gets it
    assert causes_of(mind, "lit") == [interventional.ref]
    assert causes_of(mind, "tick", interventional_only=True) == []


def test_a_counterfactual_is_recorded_without_entering_the_world():
    mind = Store()
    scope = counterfactual(mind, name="no-flip", facts=[(Ref("entity:bulb"), "lit", False)],
                           source=Ref("obs:experiment"), because="the switch was not flipped")
    assert mind.claims(Ref("entity:bulb"), "lit", scope=scope)
    assert mind.claims(Ref("entity:bulb"), "lit", scope=None) == []  # the shared world is untouched


def test_effect_size_needs_paired_runs():
    lonely = Contrast("flip:switch", "lit", with_act=(True,), without_act=())
    assert lonely.trials == 0 and lonely.effect == 0.0
    assert learn([lonely]) == []


def test_variable_treated_outcomes_are_retained_without_a_fixed_effect():
    from tensorcode.outcomes import Unknown

    contrast = Contrast("sample", "color", ("red", "blue", "red"), ("gray",) * 3)
    link, = learn([contrast])
    assert isinstance(link.effect, Unknown)
    assert link.effect.reason == "variable_outcome"
    assert [(value, score.value, score.kind) for value, score in link.effect.candidates] == [
        ("red", 2 / 3, "vote_share"), ("blue", 1 / 3, "vote_share")]
    assert link.contrast == contrast
    assert contrast.outcomes == (("red", 2), ("blue", 1))
    assert link.strength.value == 1.0 and link.strength.kind == "vote_share"
    mind = Store()
    tell_causal(mind, link, source=Ref("observation:sample"))
    assert mind.claims(link.ref, "effect") == []
    assert mind.claims(link.ref, "effect_unresolved")[0].claim.object == "variable_outcome"
    assert mind.claims(link.ref, "treated_observations")[0].claim.object == contrast.with_act
    assert mind.claims(link.ref, "control_observations")[0].claim.object == contrast.without_act
    assert mind.claims(link.ref, "treated_outcome_counts")[0].claim.object == contrast.outcomes


def test_treated_outcome_share_is_distinct_from_change_share():
    contrast = Contrast("sample", "value", (1, 1, 2, 2), (1, 0, 2, 0))
    link, = learn([contrast])
    assert link.strength.value == 0.5
    assert contrast.outcomes == ((1, 2), (2, 2))
    # Outcomes include unchanged trials: they describe treatment observations,
    # while the separate contrast measures disagreement with the control.
    assert sum(count for _, count in contrast.outcomes) == link.trials == 4


def test_constant_observed_outcome_retains_sample_without_claiming_calibration():
    contrast = Contrast("sample", "value", (True, True), (False, False))
    link, = learn([contrast])
    assert link.effect is True
    assert link.contrast is contrast
    assert link.strength.kind == "vote_share"
    assert "changed-pairs@n=2" in link.strength.basis


def test_unpaired_extra_values_do_not_change_the_learned_outcomes():
    contrast = Contrast("sample", "value", (1, 1, 99), (0, 0))
    link, = learn([contrast])
    assert link.effect == 1 and link.trials == 2
    assert contrast.outcomes == ((1, 2),)
    assert link.contrast.with_act == (1, 1, 99)


def test_missing_observations_are_distinct_from_observed_none():
    from tensorcode.outcomes import Unknown

    def observe(world):
        return {"value": None} if world else {}

    contrast, = experiment(prepare=list, act=lambda world: world.append(True),
                           observe=observe, cause="sample")
    assert contrast.with_act == (None,)
    assert isinstance(contrast.without_act[0], Unknown)
    assert contrast.trials == 0
    assert learn([contrast]) == []
    observed_none = Contrast("sample", "value", (None,), (0,))
    link, = learn([observed_none])
    assert link.effect is None and link.trials == 1


def test_incomplete_pairs_are_retained_but_do_not_vote():
    from tensorcode.outcomes import Unknown

    contrast = Contrast("sample", "value", (Unknown("unobserved"), 1, 2),
                        (0, 0, Unknown("unobserved")))
    link, = learn([contrast])
    assert link.effect == 1 and link.trials == 1
    assert link.contrast == contrast
    assert contrast.outcomes == ((1, 1),)
    mind = Store()
    tell_causal(mind, link, source=Ref("obs:sample"))
    assert isinstance(mind.claims(link.ref, "treated_observations")[0].claim.object[0], Unknown)


def test_different_empirical_evidence_has_distinct_causal_record_identity():
    first, = learn([Contrast("sample", "value", (1, 1), (0, 0))])
    second, = learn([Contrast("sample", "value", (1, 1, 1), (0, 0, 0))])
    assert first.ref != second.ref
