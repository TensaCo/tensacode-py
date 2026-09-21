import json

import pytest

from tensorcode.tools.agents import JsonMemory, MemoryRecord


def contains_text(request):
    matches = [
        record
        for record in request.candidates
        if str(request.query).casefold() in str(record.value).casefold()
    ]
    return matches[: request.limit]


def test_memory_uses_caller_retrieval_and_survives_restart(tmp_path):
    path = tmp_path / "memory.json"
    memory = JsonMemory(path, retrieve=contains_text)

    first = memory.append(
        "Customer asked about a card fee",
        kind="observation",
        metadata={"turn": 1},
    )
    response = memory.append(
        "The assistant discussed fee policy",
        kind="response",
        metadata={"turn": 1, "observation_source_id": first.source_id},
    )

    restarted = JsonMemory(path, retrieve=contains_text)
    assert restarted.records == (first, response)
    assert restarted.search("card", limit=3) == (first,)
    assert first.source_id == "memory-00000001"
    assert response.source_id == "memory-00000002"
    assert first.kind == "observation"
    assert response.kind == "response"
    assert json.loads(path.read_text())["format"] == "tensorcode-memory"


def test_memory_transaction_rolls_back_all_staged_records(tmp_path):
    path = tmp_path / "memory.json"
    memory = JsonMemory(path, retrieve=contains_text)
    existing = memory.append("kept", kind="observation")

    with pytest.raises(RuntimeError, match="turn failed"):
        with memory.transaction() as transaction:
            transaction.append("not committed", kind="observation")
            transaction.append("also not committed", kind="response")
            raise RuntimeError("turn failed")

    assert memory.records == (existing,)
    assert JsonMemory(path, retrieve=contains_text).records == (existing,)


def test_memory_rejects_retrieval_items_outside_supplied_candidates():
    invented = MemoryRecord("not-stored", "observation", "invented")
    memory = JsonMemory(retrieve=lambda request: (invented,))
    memory.append("stored", kind="observation")

    with pytest.raises(ValueError, match="candidate"):
        memory.search("anything")


def test_memory_rejects_values_its_json_codec_cannot_persist(tmp_path):
    memory = JsonMemory(tmp_path / "memory.json", retrieve=contains_text)

    with pytest.raises(TypeError, match="JSON"):
        memory.append(object(), kind="observation")

    assert memory.records == ()


def test_memory_snapshots_mutable_inputs_and_returned_records():
    memory = JsonMemory(retrieve=contains_text)
    supplied = {"nested": ["original"]}
    memory.append(supplied, kind="observation")

    supplied["nested"].append("caller mutation")
    exposed = memory.records
    exposed[0].value["nested"].append("reader mutation")

    assert memory.records[0].value == {"nested": ["original"]}


def test_failed_transaction_cannot_mutate_preexisting_values():
    memory = JsonMemory(retrieve=contains_text)
    memory.append({"status": "kept"}, kind="observation")

    with pytest.raises(RuntimeError):
        with memory.transaction() as transaction:
            transaction._records[0].value["status"] = "mutated"
            raise RuntimeError("rollback")

    assert memory.records[0].value == {"status": "kept"}


def test_memory_rejects_nested_transactions_instead_of_losing_inner_commit():
    memory = JsonMemory(retrieve=contains_text)

    with memory.transaction() as outer:
        outer.append("outer", kind="observation")
        with pytest.raises(RuntimeError, match="nested"):
            memory.append("inner", kind="observation")

    assert [record.value for record in memory.records] == ["outer"]


def test_automatic_source_id_skips_explicitly_occupied_ids():
    memory = JsonMemory(retrieve=contains_text)
    explicit = memory.append(
        "explicit", kind="observation", source_id="memory-00000001"
    )

    automatic = memory.append("automatic", kind="observation")

    assert explicit.source_id == "memory-00000001"
    assert automatic.source_id == "memory-00000002"


def test_retrieval_cannot_mutate_a_candidate_into_new_evidence():
    def mutate_candidate(request):
        request.candidates[0].value["text"] = "invented"
        return request.candidates

    memory = JsonMemory(retrieve=mutate_candidate)
    memory.append({"text": "original"}, kind="observation")

    with pytest.raises(ValueError, match="candidate"):
        memory.search("anything")

    assert memory.records[0].value == {"text": "original"}
