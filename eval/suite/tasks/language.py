"""Language: parsing against gold trees, generation, and the roles prepositions mark."""

from __future__ import annotations

from ..core import Dataset, Item, Judgement, Prompt, Response, Task, register


def _ud_available() -> bool:
    from tensorcode.language.treebank import find_treebank

    return find_treebank() is not None


def _ud_items(split: str):
    from tensorcode.language.treebank import load

    return [Item(f"ud-{split}-{i}", Prompt(" ".join(t.form for t in s)), s, {"sentence": s})
            for i, s in enumerate(load(split))]


UD = Dataset(name="UD English-EWT", license="CC BY-SA 4.0",
             url="https://github.com/UniversalDependencies/UD_English-EWT", load=_ud_items,
             available=_ud_available,
             fetch_hint="git clone UD_English-EWT into ~/.cache/tensorcode/seeds")


_MODEL: object = None


def _model():
    """The trained parser, unpickled once.

    It was being read from disk for every item, which on a 2,001-sentence treebank turned a
    component measurement into an overnight job.
    """
    global _MODEL
    if _MODEL is None:
        from pathlib import Path

        from tensorcode.language.learned_parser import load_model

        _MODEL = load_model(Path.home() / ".cache" / "tensorcode" / "models" / "ud_ewt_parser.pickle") or False
    return _MODEL or None


def _parse_and_score(subject, item: Item) -> Response:
    """Parsing is a component task: it asks the subject for its parser, not for a reply."""
    got = _model()
    if got is None:
        return Response("__no_model__", abstained=True)
    tagger, parser = got
    sentence = item.meta["sentence"]
    words = [t.form for t in sentence]
    tags = tagger.tag(words)
    from eval.parsing.legacy_baseline import parse as legacy_parse

    heads, labels = legacy_parse(parser, words, tags)  # Explicit historical repaired baseline.
    scored = [t for t in sentence if t.upos != "PUNCT"]
    uas = sum(1 for t in scored if heads.get(t.id) == t.head) / max(1, len(scored))
    las = sum(1 for t in scored if heads.get(t.id) == t.head and labels.get(t.id) == t.deprel) / max(1, len(scored))
    tag_acc = sum(1 for t, g in zip(tags, sentence) if t == g.upos) / max(1, len(sentence))
    return Response(f"las={las:.3f}", detail={"las": las, "uas": uas, "tagging": tag_acc, "decoder": "legacy-repaired-evaluation-only"})


def _judge_parse(item: Item, response: Response) -> Judgement:
    if response.text == "__no_model__":
        return Judgement(False, None, note="no trained parser")
    return Judgement(True, None, score=response.detail["las"], note="score is LAS; UAS in the row detail")


register(Task(
    id="language.dependency_parsing", area="language", what="syntax against a treebank's gold trees",
    dataset=UD, judge=_judge_parse, run=_parse_and_score, headline="score",
    splits=("dev", "test"), notes="score = labelled attachment (LAS), punctuation excluded"))


def _snacs_available() -> bool:
    from tensorcode.language.deps_semantics import find_streusle

    return find_streusle() is not None


register(Task(
    id="language.preposition_roles", area="language",
    what="which role a preposition marks, against SNACS annotations",
    dataset=Dataset(name="STREUSLE (SNACS)", license="CC BY-SA 4.0",
                    url="https://github.com/nert-nlp/streusle", load=lambda split: [],
                    available=_snacs_available,
                    fetch_hint="git clone streusle into ~/.cache/tensorcode/seeds; the task's items are not wired yet"),
    judge=lambda item, response: Judgement(False, None), notes="planned: per-token role accuracy"))

register(Task(
    id="language.generation_roundtrip", area="language",
    what="can it say what it means: re-read its own reply and recover the frame",
    dataset=Dataset(name="the agent's own replies on held-out prompts", license="n/a", url="—",
                    load=lambda split: [], available=lambda: False,
                    fetch_hint="planned: parse each reply and compare to the frame it was generated from"),
    judge=lambda item, response: Judgement(False, None)))

register(Task(
    id="language.unsupervised_parsing", area="language",
    what="grammar induced from raw text, scored against gold brackets",
    dataset=Dataset(name="UD English-EWT (text only)", license="CC BY-SA 4.0", url="—",
                    load=lambda split: [], available=lambda: False,
                    fetch_hint="planned: unlabelled bracket F1 vs right-branching and DMV baselines"),
    judge=lambda item, response: Judgement(False, None)))
