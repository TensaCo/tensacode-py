"""Explicit authored segmentation for tests of downstream mechanisms only."""
import re
from types import SimpleNamespace


def install_segmentation_fixture(reader):
    def segment(text, **bounds):
        spans = tuple(match.span() for match in re.finditer(r"\w+|[^\w\s]", text))
        candidate = SimpleNamespace(spans=spans, score=0.0,
                                    provenance=("authored:test-segmentation",))
        return SimpleNamespace(candidates=(candidate,), expansions=0, complete=True,
                               truncated=False, reason=None)
    reader.segmenter = SimpleNamespace(segment=segment)
    reader.segmentation_artifact = {"path": "fixture", "sha256": "authored-test-segmentation"}
    reader.segmentation_error = None
    reader.segmentation_beam_width = 4
    reader.segmentation_max_candidates = 2
    reader.segmentation_max_expansions = 100000
