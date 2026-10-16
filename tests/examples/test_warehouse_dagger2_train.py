"""Conditional fit rejects split drift without executing a model fit."""

import copy

import pytest

from examples.robotics.warehouse.learned_policy.dagger2_train import (
    PROTOCOL_SHA,
    validate_sources,
)


def inputs():
    cases = {f"bc-train-{i:02}": "hash" for i in range(6)}
    protocol = {
        "collection": {"case_sha256": cases},
        "fit": {"initial_weights_sha256": "a" * 64},
    }
    original = {"train": [{}] * 70, "validation": [{}] * 18, "test": []}
    corrective = {
        "train": [
            {"physical_run_id": case, "robot_id": f"r{i}"}
            for case in cases
            for i in range(2)
        ],
        "validation": [],
        "test": [],
        "dagger2": {"protocol_sha256": PROTOCOL_SHA, "qualification_gate_passed": True},
    }
    corrective["dagger2"].update(
        qualification_report_sha256="b" * 64,
        collection_source_archive_sha256="c" * 64,
        case_sha256=cases,
        collection_weights_sha256="a" * 64,
    )
    return protocol, original, corrective


def test_six_train_cases_and_original_validation_are_preserved():
    protocol, original, corrective = inputs()
    validate_sources(protocol, original, corrective)
    before = copy.deepcopy(corrective)
    assert corrective == before
    corrective["train"][0]["physical_run_id"] = "teacher-validation-offset"
    with pytest.raises(ValueError, match="membership"):
        validate_sources(protocol, original, corrective)


@pytest.mark.parametrize("change", ["gate", "validation", "duplicates"])
def test_unqualified_or_leaking_manifest_cannot_fit(change):
    protocol, original, corrective = inputs()
    if change == "gate":
        corrective["dagger2"]["qualification_gate_passed"] = False
    elif change == "validation":
        original["validation"].pop()
    else:
        corrective["train"][0] = corrective["train"][1]
    with pytest.raises(ValueError):
        validate_sources(protocol, original, corrective)
