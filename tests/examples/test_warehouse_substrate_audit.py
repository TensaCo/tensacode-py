"""Dispatch audit exercises the actual public coordinator across substrates."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.robotics.warehouse.coordination import Coordinator
from examples.robotics.warehouse.substrate_audit import (
    DEFAULT_OUTPUT,
    evaluate,
    fixtures,
)


@pytest.mark.parametrize("substrate", ["alg0", "alg1", "alg2"])
def test_boundary_dispatch_contract(substrate):
    protocol = json.loads((DEFAULT_OUTPUT / "protocol.json").read_text())
    coordinator = Coordinator(
        substrate=substrate, cost_absolute_tolerance=0.0, cost_relative_tolerance=0.0
    )
    for case in fixtures({**protocol, "random_cases": 0}):
        output = evaluate(case, coordinator)
        chosen = output["assignments"][0]["robot_id"] if output["assignments"] else None
        assert chosen == case["expected_robot"], case["id"]
        assert all(
            g["eligible"] in (0, 1)
            for robots in output["gates"].values()
            for g in robots.values()
        )


def test_fixture_seed_reproducibility_and_provenance():
    protocol = json.loads((DEFAULT_OUTPUT / "protocol.json").read_text())
    a, b = fixtures(protocol), fixtures(protocol)
    assert a == b and len(a) == 112
    assert a != fixtures({**protocol, "seed": protocol["seed"] + 1})
    assert len({case["id"] for case in a}) == len(a)


def test_retained_study_distinguishes_primary_and_exploratory_outcomes():
    import hashlib

    primary = json.loads((DEFAULT_OUTPUT / "results.json").read_text())
    probe = json.loads((DEFAULT_OUTPUT / "tie-counterexamples.json").read_text())
    assert (
        primary["protocol_sha256"]
        == hashlib.sha256((DEFAULT_OUTPUT / "protocol.json").read_bytes()).hexdigest()
    )
    assert (
        probe["protocol_sha256"]
        == hashlib.sha256(
            (DEFAULT_OUTPUT / "tie-probe-protocol.json").read_bytes()
        ).hexdigest()
    )
    assert primary["case_count"] == 112
    assert primary["all_sampled_decisions_equal"] and not primary["decision_mismatches"]
    assert primary["numerical_cost_differences"]
    assert probe["screened_pairs"] == 960
    assert probe["actual_dispatch_mismatches"] == sum(
        c["actual_dispatch_mismatch"] for c in probe["counterexamples"]
    )
    assert probe["actual_dispatch_mismatches"] > 0
    for counterexample in probe["counterexamples"]:
        first, second = counterexample["case"]["robots"]
        assert sorted(first) == sorted(second)
        assert len(set(counterexample["chosen"].values())) > 1


@pytest.mark.parametrize("version", ["", "v2"])
def test_audit_source_overlays_and_receipts_are_bound(version):
    import hashlib

    directory = DEFAULT_OUTPUT / version
    manifest = json.loads((directory / "source-manifest.json").read_text())
    report = json.loads((directory / "results.json").read_text())
    assert len(manifest["base_commit"]) == 40
    assert set(manifest["sources"]) == set(report["source_sha256"])
    for name, binding in manifest["sources"].items():
        assert binding["sha256"] == report["source_sha256"][name]
        if binding["kind"] == "archived_overlay":
            assert (
                hashlib.sha256((directory / binding["path"]).read_bytes()).hexdigest()
                == binding["sha256"]
            )
        else:
            assert binding["commit"] == manifest["base_commit"]
    for name, digest in manifest["receipt_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
