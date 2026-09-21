"""Route real support tickets with a caller-supplied policy and labels.

Input is UTF-8 JSONL with one ``{"id": "case-1", "text": "..."}`` object per
line. Output is JSONL in the same order. Missing provider confidence remains
absent; missing distributions remain null.

Example::

  python examples/support_triage.py --input tickets.jsonl --policy policy.txt \
    --label billing --label incident --label question \
    --base-url http://localhost:8000/v1 --model local-model
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NamedTuple

from tensorcode.integrations import OpenAICompatibleModel
from tensorcode.ops import text as text_ops


class Ticket(NamedTuple):
    id: str
    text: str


def load_tickets(path: Path, *, max_tickets: int, max_ticket_chars: int) -> tuple[Ticket, ...]:
    if max_tickets < 1 or max_ticket_chars < 1:
        raise ValueError("ticket limits must be positive")
    tickets, seen = [], set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if len(tickets) >= max_tickets:
                raise ValueError(f"input exceeds max_tickets={max_tickets}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict) or set(record) != {"id", "text"}:
                raise ValueError(f"line {line_number} must contain only id and text")
            ticket_id, text = record["id"], record["text"]
            if not isinstance(ticket_id, str) or not ticket_id.strip():
                raise ValueError(f"line {line_number} id must be a nonempty string")
            if ticket_id in seen:
                raise ValueError(f"duplicate ticket id: {ticket_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"line {line_number} text must be a nonempty string")
            if len(text) > max_ticket_chars:
                raise ValueError(f"line {line_number} exceeds max_ticket_chars={max_ticket_chars}")
            seen.add(ticket_id)
            tickets.append(Ticket(ticket_id, text))
    return tuple(tickets)


def route_tickets(tickets, *, labels, policy: str, model) -> list[dict]:
    tickets, labels = tuple(tickets), tuple(labels)
    if len(labels) < 2 or len(set(labels)) != len(labels) or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise ValueError("labels must contain at least two unique nonempty strings")
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("policy must be nonempty caller-supplied text")
    classify = text_ops.Classify.from_model(
        model,
        labels=labels,
        instructions=(
            "Route the supplied support ticket under this caller-owned policy. "
            "Abstain when the policy does not support one route.\n\nPOLICY:\n" + policy
        ),
    )
    values = tuple(text_ops.TextEncoder()(ticket.text) for ticket in tickets)
    results = classify.batch(values)
    routed = []
    for ticket, result in zip(tickets, results):
        record = {
            "id": ticket.id,
            "route": result.label,
            "abstained": result.abstained,
            "distribution": dict(result.distribution) if result.distribution is not None else None,
        }
        if result.confidence is not None:
            record["confidence"] = result.confidence
        routed.append(record)
    return routed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api", choices=("chat_completions", "responses"), default="chat_completions")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tickets", type=int, default=1000)
    parser.add_argument("--max-ticket-chars", type=int, default=10_000)
    parser.add_argument("--max-policy-chars", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_policy_chars < 1:
        parser.error("--max-policy-chars must be positive")
    with args.policy.open(encoding="utf-8") as source:
        policy = source.read(args.max_policy_chars + 1)
    if len(policy) > args.max_policy_chars:
        parser.error(f"policy exceeds --max-policy-chars={args.max_policy_chars}")
    tickets = load_tickets(
        args.input, max_tickets=args.max_tickets, max_ticket_chars=args.max_ticket_chars
    )
    model = OpenAICompatibleModel(
        base_url=args.base_url,
        model=args.model,
        api=args.api,
        api_key=os.environ.get(args.api_key_env),
        timeout=args.timeout,
    )
    records = route_tickets(tickets, labels=args.label, policy=policy, model=model)
    rendered = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
