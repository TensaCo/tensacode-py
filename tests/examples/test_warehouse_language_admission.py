from examples.robotics.warehouse.language_admission import prepare_candidate
from examples.robotics.warehouse.language_eval import fixture_scene


def test_correct_candidate_requires_scheduler_checks_and_fresh_evidence():
    text = "Move parcel_0 to receiver_0"
    raw = '{"instruction":["move","parcel_0","receiver_0"]}'
    result = prepare_candidate(text, raw, fixture_scene(), 10.0)
    assert result["status"] == "ready_for_scheduler_checks"
    assert result["dispatch_authorized"] is False
    assert result["raw_candidate"] == raw
    assert (
        prepare_candidate(text, raw, fixture_scene(), 20.0)["status"]
        != "ready_for_scheduler_checks"
    )


def test_wrong_order_cannot_be_repaired_by_certificate():
    result = prepare_candidate(
        "Move parcel_0 to receiver_0 then move parcel_1 to receiver_1",
        '{"instruction":["parallel",["move","parcel_0","receiver_0"],["move","parcel_1","receiver_1"]]}',
        fixture_scene(),
        10.0,
    )
    assert result["status"] == "needs_review"
    assert result["semantic_certificate"]["status"] == "rejected"
    assert "instruction" not in result


def test_query_never_becomes_action_proposal():
    result = prepare_candidate(
        "Where is parcel_0?", '{"where":"parcel_0"}', fixture_scene(), 10.0
    )
    assert result["status"] == "answered"
    assert result["dispatch_authorized"] is False
    assert "instruction" not in result


def test_unsupported_constraint_and_invalid_output_stop_before_grounding():
    for text, raw in [
        (
            "Move parcel_0 to receiver_0 unless occupied",
            '{"instruction":["move","parcel_0","receiver_0"]}',
        ),
        ("Move parcel_0 to receiver_0", '{"instruction":'),
    ]:
        result = prepare_candidate(text, raw, fixture_scene(), 10.0)
        assert result["status"] in {"needs_review", "invalid"}
        assert "grounding" not in result
