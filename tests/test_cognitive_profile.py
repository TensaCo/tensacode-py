"""The cognitive-profile harness: its statistics, and its refusal to report what it cannot ground."""

import json
from pathlib import Path

from eval.profile.profile import build, luck_corrected, wilson

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"


def test_wilson_bounds_a_proportion_and_declines_an_empty_sample():
    low, high = wilson(9, 10)
    assert 0.5 < low < 0.9 < high <= 1.0
    assert wilson(0, 0) is None


def test_luck_correction_removes_the_credit_chance_would_have_got():
    # 16 correct of 274, of which 8.52 were expected by chance
    assert luck_corrected(16, 274, 8.52) == round((16 - 8.52) / 274, 4)
    # it never reports a negative capability
    assert luck_corrected(3, 100, 9.0) == 0.0


def test_an_axis_with_no_probe_data_is_ungrounded_not_zero():
    profile = build(probes={})
    by_key = {a["key"]: a for a in profile["axes"]}
    for key in ("compositionality", "belief_revision"):
        assert by_key[key]["grounded"] is False
        assert by_key[key]["ungrounded_reason"], f"{key} must say why it could not be grounded"
        assert by_key[key]["verdict"] == "ungrounded"


def test_every_measure_declares_its_provenance():
    profile = build(probes={})
    for axis in profile["axes"]:
        for m in axis["measures"]:
            assert m["env_author"] and m["grader"] and m["source"], f"{axis['key']}: {m['name']}"
            assert m["floor"], "a measure without a floor is not interpretable"


def test_the_published_profile_is_reproducible_from_disk():
    published = json.loads((RESULTS / "cognitive_profile.json").read_text())
    assert published["axes_total"] == len(published["axes"]) == 10
    # the harvested (non-probe) axes recompute to the same values
    fresh = {a["key"]: a for a in build(probes={})["axes"]}
    for axis in published["axes"]:
        if axis["key"] in ("compositionality", "belief_revision", "grounding"):
            continue  # live probes; not reproducible without a running assistant
        assert [m["value"] for m in fresh[axis["key"]]["measures"]] == [m["value"] for m in axis["measures"]]
