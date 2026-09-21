import importlib.util
import json
from pathlib import Path

import pytest

from tensorcode.ops import llm


def example():
    path = Path(__file__).parents[2] / "examples/support_triage.py"
    spec = importlib.util.spec_from_file_location("support_triage_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchModel:
    def __init__(self, outputs):
        self.outputs = tuple(outputs)
        self.requests = ()

    def complete_batch(self, requests):
        self.requests = tuple(requests)
        return self.outputs


def test_routes_real_jsonl_records_with_supplied_policy_and_no_invented_confidence(tmp_path):
    mod = example()
    path = tmp_path / "tickets.jsonl"
    path.write_text(
        '\n'.join(
            (
                json.dumps({"id": "case-2", "text": "Production login is down."}),
                json.dumps({"id": "case-1", "text": "Can I change my avatar?"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    tickets = mod.load_tickets(path, max_tickets=10, max_ticket_chars=200)
    model = BatchModel(
        (
            llm.ModelOutput(
                structured={
                    "label": "incident",
                    "distribution": {"incident": 0.9, "question": 0.1},
                    "abstained": False,
                }
            ),
            llm.ModelOutput(
                structured={
                    "label": None,
                    "distribution": None,
                    "abstained": True,
                }
            ),
        )
    )

    routed = mod.route_tickets(
        tickets,
        labels=("incident", "question"),
        policy="Route outages to incident. Ambiguous requests must abstain.",
        model=model,
    )

    assert routed == [
        {
            "id": "case-2",
            "route": "incident",
            "abstained": False,
            "distribution": {"incident": 0.9, "question": 0.1},
        },
        {
            "id": "case-1",
            "route": None,
            "abstained": True,
            "distribution": None,
        },
    ]
    assert [request.messages[-1].content for request in model.requests] == [
        "Production login is down.",
        "Can I change my avatar?",
    ]
    assert all("Route outages to incident" in request.instructions for request in model.requests)
    assert "confidence" not in routed[0]


def test_preserves_provider_supplied_confidence_without_deriving_one():
    mod = example()
    model = BatchModel(
        (
            llm.ModelOutput(
                structured={
                    "label": "billing",
                    "distribution": None,
                    "confidence": 0.72,
                    "abstained": False,
                }
            ),
        )
    )
    result = mod.route_tickets(
        (mod.Ticket("t-1", "Duplicate charge"),),
        labels=("billing", "other"),
        policy="Use billing for payment issues.",
        model=model,
    )
    assert result[0]["confidence"] == 0.72
    assert result[0]["distribution"] is None


@pytest.mark.parametrize(
    "records, message",
    [
        ([{"id": "same", "text": "one"}, {"id": "same", "text": "two"}], "duplicate"),
        ([{"id": "x", "text": ""}], "text"),
        ([{"id": "x", "text": "too long"}], "max_ticket_chars"),
    ],
)
def test_rejects_invalid_ticket_files(tmp_path, records, message):
    mod = example()
    path = tmp_path / "tickets.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    limit = 3 if message == "max_ticket_chars" else 100
    with pytest.raises(ValueError, match=message):
        mod.load_tickets(path, max_tickets=10, max_ticket_chars=limit)


def test_rejects_empty_or_duplicate_labels():
    mod = example()
    ticket = mod.Ticket("x", "hello")
    with pytest.raises(ValueError, match="labels"):
        mod.route_tickets((ticket,), labels=("same", "same"), policy="p", model=BatchModel(()))


def test_rejects_model_route_outside_supplied_labels():
    mod = example()
    model = BatchModel(
        (llm.ModelOutput(structured={"label": "invented", "abstained": False}),)
    )
    with pytest.raises(llm.InvalidModelOutput, match="configured labels"):
        mod.route_tickets(
            (mod.Ticket("x", "hello"),),
            labels=("billing", "other"),
            policy="caller policy",
            model=model,
        )
