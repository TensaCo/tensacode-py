"""Portable links must preserve frozen artifacts through final publication."""

import gzip
import json

import pytest

from examples.robotics.warehouse import package_final_policy as pack


def test_candidate_binding_requires_exact_frozen_bytes(tmp_path):
    weights = tmp_path / "portable" / "weights.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"frozen weights")
    manifest = {
        "candidates": [
            {
                "id": "candidate",
                "files": {"weights.pt": {"sha256": pack.digest(weights.read_bytes())}},
            }
        ]
    }
    assert (
        pack.bind_candidates(manifest, tmp_path)["candidate"]["weights.pt"] == weights
    )
    weights.write_bytes(b"replacement weights")
    with pytest.raises(ValueError, match="missing or changed"):
        pack.bind_candidates(manifest, tmp_path)


def test_relative_model_and_source_links_survive_publication(tmp_path):
    portable = tmp_path / "portable" / "weights.pt"
    portable.parent.mkdir()
    portable.write_bytes(b"frozen weights")
    staging = tmp_path / ".stage"
    archive = staging / "shared" / "source.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"source snapshot")
    run = staging / "runs" / "policy" / "case"
    pack.relative_link(portable, run / "learned-policy" / "weights.pt")
    pack.relative_link(archive, run / "source.zip")
    records = run / "records.json.gz"
    records.write_bytes(gzip.compress(b'[{"t":0}]', mtime=0))
    before = pack.evidence_manifest(run)
    staging.rename(tmp_path / "final")
    final_run = tmp_path / "final" / "runs" / "policy" / "case"
    assert pack.evidence_manifest(final_run) == before
    assert before["records.json.gz"]["raw_sha256"] == pack.digest(b'[{"t":0}]')
    assert (final_run / "source.zip").read_bytes() == b"source snapshot"


def test_incomplete_matrix_never_publishes(tmp_path, monkeypatch):
    from examples.robotics.warehouse import final_policy_audit

    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "execution-manifest.json").write_text(json.dumps({"candidates": []}))
    monkeypatch.setattr(
        final_policy_audit, "audit", lambda *args: {"complete_and_matched": False}
    )
    output = tmp_path / "final"
    with pytest.raises(ValueError, match="incomplete"):
        pack.package(plan, tmp_path / "raw", output, tmp_path / "models")
    assert not output.exists()
    assert not list(tmp_path.glob(".final-evidence-*"))


def test_partial_keeps_all_forty_statuses_and_corrupt_available_bytes(
    tmp_path, monkeypatch
):
    from examples.robotics.warehouse import final_policy_audit

    plan = tmp_path / "plan"
    plan.mkdir()
    frozen = b'{"candidates": []}'
    (plan / "execution-manifest.json").write_bytes(frozen)
    raw = tmp_path / "raw"
    damaged = raw / "policy" / "case0"
    damaged.mkdir(parents=True)
    (damaged / "probe.json").write_bytes(b'{"incomplete":')
    (damaged / "records.json.gz").write_bytes(b"broken gzip bytes")
    unfinished = raw / "policy" / "case1"
    unfinished.mkdir()
    (unfinished / "records.json").write_bytes(b'[{"t": 0},')
    rows = [
        {
            "policy": "policy",
            "case": f"case{i}",
            "status": "invalid" if i == 0 else "incomplete" if i == 1 else "unrun",
            **({"error": "Malformed evidence"} if i == 0 else {}),
        }
        for i in range(40)
    ]
    monkeypatch.setattr(
        final_policy_audit,
        "audit",
        lambda *args: {
            "complete_and_matched": False,
            "runs": json.loads(json.dumps(rows)),
            "summary": {},
        },
    )
    output = tmp_path / "final"
    pack.package(plan, raw, output, tmp_path / "models", allow_partial=True)
    audit = json.loads((output / "audit.json").read_text())
    assert len(audit["runs"]) == 40
    assert [r["status"] for r in audit["runs"]] == [r["status"] for r in rows]
    assert audit["runs"][0]["error"] == "Malformed evidence"
    assert audit["comparative_claims_qualified"] is False
    assert "INCOMPLETE" in (output / "README.md").read_text()
    assert (
        output / "runs/policy/case0/records.json.gz"
    ).read_bytes() == b"broken gzip bytes"
    assert (
        gzip.decompress((output / "runs/policy/case1/records.json.gz").read_bytes())
        == b'[{"t": 0},'
    )
    assert not (output / "runs/policy/case2").exists()
    files = json.loads((output / "runs/policy/case0/manifest.json").read_text())[
        "files"
    ]
    assert "decompression_error" in files["records.json.gz"]
    assert (plan / "execution-manifest.json").read_bytes() == frozen


def test_partial_does_not_replace_changed_or_missing_model_evidence(tmp_path):
    frozen = tmp_path / "portable/weights.pt"
    frozen.parent.mkdir()
    frozen.write_bytes(b"frozen")
    raw = tmp_path / "raw"
    (raw / "learned-policy").mkdir(parents=True)
    (raw / "learned-policy/weights.pt").write_bytes(b"wrong checkpoint")
    destination = tmp_path / "snapshot"
    pack.stage_partial(raw, destination, {"weights.pt": frozen, "policy.json": frozen})
    assert (
        destination / "learned-policy/weights.pt"
    ).read_bytes() == b"wrong checkpoint"
    assert not (destination / "learned-policy/weights.pt").is_symlink()
    assert not (destination / "learned-policy/policy.json").exists()
