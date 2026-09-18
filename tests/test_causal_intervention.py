"""Causation: earned by intervention, kept apart from co-occurrence."""

from tensacode.causal import (Causal, Contrast, causes_of, correlations, counterfactual, distinguish,
                              experiment, learn, moved, tell_causal)
from tensacode.outcomes import Score
from tensacode.records import Ref, Store


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
