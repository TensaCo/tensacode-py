"""Learning: can it acquire structure, and does that carry to language it was not given."""

from __future__ import annotations

from ..core import Dataset, Item, Judgement, Prompt, Response, Task, register


def _formal_available() -> bool:
    return True  # the languages are definitions, not downloads


def _formal_items(split: str):
    from eval.grammar_induction.formal import LANGUAGES
    from eval.grammar_induction.formal_v4 import FRESH

    source = LANGUAGES if split == "dev" else FRESH
    return [Item(f"{split}-{name}", Prompt(f"learn the language {name}"), None, {"name": name, "spec": spec})
            for name, spec in source.items()]


def _learn_language(subject, item: Item) -> Response:
    """Learn the language from short positive examples; score precision and recall on longer ones."""
    import random

    from eval.grammar_induction.formal import test_set, training_set
    from eval.grammar_induction.formal_v2 import measure, sample_cfg
    from tensorcode.language.induce import learn_congruential

    spec = item.meta["spec"]
    rng = random.Random(0)
    train = training_set(spec)
    positives, _ = test_set(spec, rng)
    lo, hi = spec["train_len"] + 1, 3 * spec["train_len"]
    grammar = learn_congruential(train, complete_up_to=spec["train_len"])
    got = measure(item.meta["name"], spec, grammar.accepts, sample_cfg(grammar, rng, lo, hi, 200), positives)
    return Response(f"P={got['precision']} R={got['recall']}", detail=got)


def _judge_language(item: Item, response: Response) -> Judgement:
    d = response.detail or {}
    precision, recall = d.get("precision") or 0.0, d.get("recall") or 0.0
    exact = precision >= 0.95 and recall >= 0.95
    return Judgement(True, bool(exact), score=round(min(precision, recall), 3),
                     note="correct == learned exactly (precision and recall >= 0.95 on longer strings)")


register(Task(
    id="learning.formal_languages", area="learning",
    what="learning non-regular languages from positive examples",
    dataset=Dataset(name="formal languages (mathematical definitions)", license="n/a",
                    url="eval/grammar_induction/", load=_formal_items, available=_formal_available),
    judge=_judge_language, run=_learn_language, splits=("dev", "test"),
    notes="dev = the five developed against; test = four the fix never saw. Pre-registered"))

register(Task(
    id="learning.chomsky_benchmark", area="learning",
    what="a published formal-language suite, instead of languages we chose",
    dataset=Dataset(name="Neural Networks and the Chomsky Hierarchy (Deletang et al. 2023)",
                    license="Apache-2.0 (check)", url="https://github.com/google-deepmind/neural_networks_chomsky_hierarchy",
                    load=lambda split: [], available=lambda: False,
                    fetch_hint="clone the suite and adapt its generators"),
    judge=lambda item, response: Judgement(False, None)))

register(Task(
    id="learning.in_use", area="learning",
    what="new words and new constructions picked up from conversation",
    dataset=Dataset(name="held-out constructions removed from the seed grammar", license="n/a", url="—",
                    load=lambda split: [], available=lambda: False,
                    fetch_hint="planned: delete N constructions from the seed, expose the learner to text, count recovery"),
    judge=lambda item, response: Judgement(False, None),
    notes="the owner's 'actually learns language' requirement; nothing is learned in use today"))

register(Task(
    id="learning.sample_efficiency", area="learning", what="how many examples a new skill takes",
    dataset=Dataset(name="skills taught in conversation", license="n/a", url="—", load=lambda split: [],
                    available=lambda: False, fetch_hint="planned"),
    judge=lambda item, response: Judgement(False, None)))
