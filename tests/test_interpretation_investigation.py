from dataclasses import FrozenInstanceError

import pytest

from tensorcode.agent.investigation import CandidateHypothesis, investigate
from tensorcode.goals import Condition
from tensorcode.outcomes import Unknown


class Observer:
    def __init__(self, answers, name="observer"):
        self.answers = answers
        self.name = name
        self.probes = []

    def observe_condition(self, condition):
        self.probes.append(condition)
        assert not condition.negated
        return self.answers.get(condition.pred, Unknown("missing"))


def hypothesis(name, *predictions):
    return CandidateHypothesis(name, predictions, ("authored test model",))


def atom(name, negated=False):
    return Condition(name, {"item": ["structured", {"value": 2}]}, negated)


def test_balanced_discriminating_probe_precedes_shared_confirmation():
    shared, split = atom("shared"), atom("split")
    models = [hypothesis("one", shared, split), hypothesis("two", shared, atom("split", True))]
    provider = Observer({"shared": True, "split": True})
    result = investigate(models, [provider])
    assert [o.condition.pred for o in result.observations] == ["split", "shared"]
    assert result.selected_id == "one"
    assert result.assessments[1].contradicted == (atom("split", True),)
    assert result.assessments[1].confirmed == (shared,)


def test_all_predictions_required_even_after_eliminating_rival():
    models = [hypothesis("one", atom("split"), atom("additional")),
              hypothesis("two", atom("split", True))]
    result = investigate(models, [Observer({"split": True})])
    assert result.selected_id is None
    assert result.assessments[0].unresolved == (atom("additional"),)


def test_missing_predictions_do_not_contradict_candidate():
    result = investigate([hypothesis("one", atom("split")), hypothesis("two")],
                         [Observer({"split": True})])
    assert result.selected_id is None
    assert all(a.viable for a in result.assessments)


@pytest.mark.parametrize("answers", [{}, {"split": Unknown("unavailable")}, {"split": 1}, {"split": None}])
def test_missing_unknown_invalid_never_count_as_false(answers):
    result = investigate([hypothesis("one", atom("split", True))], [Observer(answers)])
    assert result.selected_id is None
    assert result.assessments[0].viable
    assert isinstance(result.observations[0].value, Unknown)


def test_conflicting_providers_preserved_and_cannot_eliminate():
    result = investigate([hypothesis("one", atom("split")), hypothesis("two", atom("split", True))],
                         [Observer({"split": True}, "a"), Observer({"split": False}, "b")])
    assert result.selected_id is None
    assert result.observations[0].value.reason == "conflicting_observations"
    assert [e.value for e in result.observations[0].providers] == [True, False]
    assert all(a.viable for a in result.assessments)


def test_invalid_provider_blocks_other_positive_evidence():
    result = investigate([hypothesis("one", atom("split"))],
                         [Observer({"split": True}, "a"), Observer({"split": 1}, "b")])
    assert result.selected_id is None
    assert result.observations[0].value.reason == "invalid_observation"


def test_unknown_provider_abstains_without_erasing_valid_observation():
    result = investigate([hypothesis("one", atom("split"))],
                         [Observer({"split": True}, "a"), Observer({}, "b")])
    assert result.selected_id == "one"
    assert result.observations[0].providers[1].status == "unknown"


def test_no_arbitrary_winner_and_each_atom_queried_once():
    provider = Observer({"shared": True})
    result = investigate([hypothesis("a", atom("shared"), atom("shared")),
                          hypothesis("b", atom("shared"))], [provider])
    assert result.selected_id is None
    assert len(provider.probes) == 1
    assert result.reason == "unresolved_alternatives"


def test_budget_retains_partial_evidence():
    result = investigate([hypothesis("a", atom("one"), atom("two"))],
                         [Observer({"one": True, "two": True})], max_probes=1)
    assert result.selected_id is None
    assert result.reason == "probe_budget_exhausted"
    assert len(result.observations) == 1
    assert result.assessments[0].confirmed == (atom("one"),)
    assert result.assessments[0].unresolved == (atom("two"),)


def test_empty_and_no_evidence_singleton_do_not_select():
    assert investigate([], []).reason == "no_hypotheses"
    assert investigate([hypothesis("a")], []).selected_id is None
    assert investigate([hypothesis("a", atom("one"))], []).selected_id is None


def test_all_contradicted():
    result = investigate([hypothesis("a", atom("one"))], [Observer({"one": False})])
    assert result.reason == "all_hypotheses_contradicted"
    assert not result.assessments[0].viable


def test_snapshot_is_detached_from_inputs_provider_and_returned_values():
    condition = atom("one")
    provider = Observer({"one": True})
    result = investigate([hypothesis("a", condition)], [provider])
    condition.args["item"].append("changed input")
    provider.probes[0].args["item"].append("changed provider")
    result.observations[0].condition.args["item"].append("changed returned view")
    result.assessments[0].confirmed[0].args["item"].append("changed assessment")
    assert result.observations[0].condition == atom("one")
    assert result.assessments[0].confirmed == (atom("one"),)
    with pytest.raises(FrozenInstanceError):
        result.observations[0].order = 99


def test_validation():
    with pytest.raises(ValueError, match="basis"):
        CandidateHypothesis("a", (), ())
    with pytest.raises(ValueError, match="contradictory"):
        hypothesis("a", atom("one"), atom("one", True))
    with pytest.raises(ValueError, match="unique"):
        investigate([hypothesis("a"), hypothesis("a")], [])
    with pytest.raises(ValueError, match="unique"):
        investigate([], [Observer({}), Observer({})])
    for budget in (-1, True, 1.5):
        with pytest.raises(ValueError):
            investigate([], [], max_probes=budget)


def test_equal_repr_does_not_conflate_distinct_domain_values():
    class Ref:
        def __init__(self, identity):
            self.identity = identity
        def __eq__(self, other):
            return isinstance(other, Ref) and self.identity == other.identity
        def __repr__(self):
            return "same"
    conditions = [Condition("one", {"ref": Ref(1)}), Condition("one", {"ref": Ref(2)})]
    observer = Observer({"one": True})
    result = investigate([hypothesis("a", *conditions)], [observer])
    assert result.selected_id == "a"
    assert len(observer.probes) == 2


def test_observer_exception_is_retained_and_does_not_become_false():
    class Broken(Observer):
        def observe_condition(self, condition):
            raise RuntimeError("cannot read")
    result = investigate([hypothesis("a", atom("one", True))], [Broken({})])
    assert result.selected_id is None
    assert result.observations[0].providers[0].reason == "observation_error"
