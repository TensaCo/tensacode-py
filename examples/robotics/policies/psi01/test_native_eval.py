"""Result-critical camera corruption and frozen protocol checks, no simulator."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "native_eval", Path(__file__).with_name("native_eval.py")
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_camera_mask_never_mutates_source_and_has_exact_fixed_area():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    observed = native.transform_rgb(image, "occlusion")
    assert np.count_nonzero(observed[:, :, 0]) == 54 * 96
    assert np.all(observed[153:207, 272:368] == 127)
    assert not np.any(image)
    assert np.array_equal(native.transform_rgb(image, "nominal"), image)
    assert np.array_equal(native.transform_rgb(image, "paraphrase"), image)
    with pytest.raises(ValueError):
        native.transform_rgb(image, "object-targeted-mask")


def test_protocol_arms_share_cells_and_checkpoint_ablation():
    protocol = json.loads(Path(__file__).with_name("native_protocol.json").read_text())
    assert protocol["configs"] == [0, 1, 2, 3]
    assert [c["id"] for c in protocol["conditions"]] == [
        "nominal",
        "paraphrase",
        "occlusion",
    ]
    assert (
        protocol["checkpoint_sha256"]["psi01"]
        == protocol["checkpoint_sha256"]["psi01-no-logits"]
    )
    assert protocol["action_tracker"] == "SIMPLE Psi0Agent / AMO, g1_dex3_wholebody"
