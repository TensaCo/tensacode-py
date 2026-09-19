"""Vision beyond questions: recognition, grounding, and the parts of it we have not built.

CIFAR-10 is here as a **diagnostic**, not a capability claim: ten coarse classes at 32x32
say almost nothing about seeing. The real tasks are the ones marked missing, and the
scorecard prints them so the gap is visible.
"""

from __future__ import annotations

from ..core import Dataset, Item, Judgement, Prompt, Response, Task, register


def _cifar_available() -> bool:
    from pathlib import Path

    return (Path.home() / ".cache" / "tensorcode" / "datasets" / "cifar-10-batches-py").exists()


def _cifar_items(split: str):
    from eval.vision_hierarchy.cifar import load

    x, y = load("test_batch")
    start = 4000 if split == "dev" else 6000
    return [Item(f"cifar-{split}-{i}", Prompt("what is in this picture?"), [str(int(y[i]))],
                 {"pixels": x[i]}) for i in range(start, start + 200)]


def _recognise(subject, item: Item) -> Response:
    from tensorcode.agent.vision_plugin import VisionPlugin
    from tensorcode.records import Ref

    plugin = VisionPlugin()
    claims = list(plugin.see(item.meta["pixels"], Ref("image:1")))
    label = next((c.object for c in claims if c.predicate == "is_a"), None)
    return Response(str(label) if label else "I don't know", abstained=label is None)


def _judge_cifar(item: Item, response: Response) -> Judgement:
    from eval.vision_hierarchy.train_concepts import LABELS

    if response.abstained:
        return Judgement(False, None)
    want = LABELS[int(item.gold[0])]
    return Judgement(True, response.text.strip().lower() == want)


register(Task(
    id="vision.cifar10_diagnostic", area="vision", what="ten coarse classes at 32x32: a smoke test, not a capability",
    dataset=Dataset(name="CIFAR-10", license="research use", url="https://www.cs.toronto.edu/~kriz/cifar.html",
                    load=_cifar_items, available=_cifar_available,
                    fetch_hint="curl the CIFAR-10 python tarball into ~/.cache/tensorcode/datasets"),
    judge=_judge_cifar, run=_recognise, splits=("dev", "test"),
    notes="DIAGNOSTIC ONLY. Ten classes, 32x32. Never quote as 'vision works'"))


def _missing(name, license, url, hint):
    return Dataset(name=name, license=license, url=url, load=lambda split: [], available=lambda: False,
                   fetch_hint=hint)


register(Task(
    id="vision.object_naming", area="vision", what="naming what is in a photo, open vocabulary, tied to WordNet",
    dataset=_missing("Visual Genome objects", "CC BY 4.0", "https://homes.cs.washington.edu/~ranjay/visualgenome/",
                     "sample VG region descriptions + objects into the cache"),
    judge=lambda item, response: Judgement(False, None),
    notes="the replacement for CIFAR: thousands of concepts, and they are WordNet synsets"))

register(Task(
    id="vision.gqa", area="vision", what="compositional questions about images (relations, attributes)",
    dataset=_missing("GQA", "CC BY 4.0", "https://cs.stanford.edu/people/dorarad/gqa/",
                     "sample the balanced val split"),
    judge=lambda item, response: Judgement(False, None)))

register(Task(
    id="vision.screenspot", area="vision", what="pointing at the right UI element from a description",
    dataset=_missing("ScreenSpot", "Apache-2.0", "https://github.com/njucckevin/SeeClick",
                     "sample desktop + web splits"),
    judge=lambda item, response: Judgement(False, None),
    notes="the grounding half of screen understanding: where, not just what"))

register(Task(
    id="vision.hierarchy_vs_flat", area="vision", what="does a learned feature hierarchy beat one flat layer",
    dataset=_missing("CIFAR-10 + STL-10", "research use", "—",
                     "planned rerun with receptive-field grouping, on two datasets"),
    judge=lambda item, response: Judgement(False, None),
    notes="last attempt failed by 5.6 points; a second attempt needs new pre-registration"))
