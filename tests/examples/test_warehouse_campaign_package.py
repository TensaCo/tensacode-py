"""Campaign publication must preserve evidence and reject incomplete validation."""

import gzip

import pytest

from examples.robotics.warehouse import package_campaign as pack


def test_size_gate_does_not_publish_partial_campaign(tmp_path, monkeypatch):
    def stage(source, destination, validator):
        destination.mkdir()
        (destination / "probe.json").write_text("original")
        return {"bytes": 8}

    monkeypatch.setattr(pack, "stage_run", stage)
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="above"):
        pack.package_campaign([tmp_path / "a", tmp_path / "b"], output, max_bytes=10)
    assert not output.exists()
    assert not list(tmp_path.glob(".campaign-*"))


def test_failed_validation_does_not_publish_first_run(tmp_path, monkeypatch):
    def stage(source, destination, validator):
        if source.name == "b":
            raise ValueError("invalid evidence")
        destination.mkdir()
        return {"bytes": 1}

    monkeypatch.setattr(pack, "stage_run", stage)
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="invalid evidence"):
        pack.package_campaign([tmp_path / "a", tmp_path / "b"], output)
    assert not output.exists()


def test_compressed_evidence_preserves_original_bytes(tmp_path):
    original = tmp_path / "records.json"
    original.write_bytes(b'[{"t":0,"value":1.0000}]\n')
    output = tmp_path / "records.json.gz"
    metadata = pack.compress(original, output)
    assert gzip.decompress(output.read_bytes()) == original.read_bytes()
    assert metadata["raw_sha256"] == pack.digest(original.read_bytes())


def test_initial_graph_excludes_future_observations():
    first = {"id": "parcel", "observed_at": 0.0, "position": [1, 2, 3]}
    later = {**first, "observed_at": 1.0, "position": [9, 9, 9]}
    graph = {
        "revision": 2,
        "entities": [later],
        "relations": [],
        "history": [
            {"revision": 1, "observation": first},
            {"revision": 2, "observation": later},
        ],
    }
    projected = pack.graph_projections(graph, 0.0)
    assert projected["initial"]["entities"] == [first]
    assert projected["final"]["entities"] == [later]
    assert len(graph["history"]) == 2


def test_explicit_resource_mapping_reaches_run_packager(tmp_path, monkeypatch):
    observed = []

    def stage(source, destination, validator, resources=None):
        observed.append(resources)
        destination.mkdir()
        return {"bytes": 1}

    monkeypatch.setattr(pack, "stage_run", stage)
    logs = tmp_path / "resource-logs"
    pack.package_campaign(
        [tmp_path / "a"], tmp_path / "published", resources={"a": str(logs)}
    )
    assert observed == [str(logs)]
