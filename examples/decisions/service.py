"""A back-office decision service: HTTP API plus a small operator UI.

    python -m examples.decisions.service [--port 8795] [--tier cascade|rules|learned]

Routes, all ordinary JSON over HTTP — this is meant to look like any other backend:

    POST /triage            {text, charges?}        -> department, urgency, refund asked, gate
    POST /decide            {text, charges?}        -> the whole handler, including refund
    POST /rerank            {query, passages}       -> ranked with relevance scores
    POST /supports          {claim, passage}        -> holds | fails | unknown
    GET  /why?ticket=T-1                            -> the claim chain behind a decision
    POST /replay            {ticket}                -> re-decide stored state, report drift
    GET  /config                                    -> tier, thresholds and their basis
    GET  /trace                                     -> the last request's spans

Nothing here generates text. Every response is a typed value or an explicit refusal.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

import tensacode as tc
from tensacode.outcomes import Unknown

from . import decisions, tiers
from .audit import Audit, replay
from .domain import Charge, Passage, Ticket
from .gating import gates

PAGE = Path(__file__).parent / "ui.html"


def _charges(rows: list[dict] | None) -> tuple[Charge, ...]:
    now = datetime.now(timezone.utc)
    out = []
    for i, row in enumerate(rows or ()):
        at = row.get("at")
        when = datetime.fromisoformat(at) if isinstance(at, str) else now - timedelta(days=int(row.get("days_ago", 0)))
        out.append(
            Charge(
                row.get("id") or f"C-{i + 1}",
                row.get("merchant", "unknown"),
                float(row.get("amount_gbp", 0.0)),
                when,
                row.get("status", "settled"),
                row.get("duplicate_of"),
            )
        )
    return tuple(out)


class Service:
    """Holds the runtime, the gates and the audit store. One per process."""

    def __init__(self, tier: str, train_csv: Path | None) -> None:
        self.tier = tier
        self.report = None
        if tier == "rules":
            bindings = tiers.rules_only()
        elif tier == "learned":
            bindings, self.report = tiers.learned_only(train_csv)  # type: ignore[arg-type]
        elif tier == "cascade":
            bindings, self.report = tiers.cascade(train_csv)  # type: ignore[arg-type]
        else:  # the measured default: every tier reports a confidence the gate can use
            bindings, self.report = tiers.cascade_scored(train_csv)  # type: ignore[arg-type]
        self.runtime = tiers.runtime(bindings)
        self.gates = gates()
        self.audit = Audit()
        self.seq = 0

    def ticket(self, body: dict) -> Ticket:
        self.seq += 1
        return Ticket(
            body.get("ticket") or f"T-{self.seq}",
            body.get("customer", "someone@example.com"),
            body.get("text", ""),
            datetime.now(timezone.utc),
            _charges(body.get("charges")),
        )

    def triage(self, body: dict) -> dict:
        ticket = self.ticket(body)
        with tc.use(self.runtime):
            t = decisions.triage(ticket, gate=self.gates["routing"], audit=self.audit)
        return {
            "ticket": ticket.id,
            "intent": None if isinstance(t.intent, Unknown) else t.intent.name,
            "unknown": t.intent.reason if isinstance(t.intent, Unknown) else None,
            "department": None if isinstance(t.department, Unknown) else t.department.value,
            "urgency": None if isinstance(t.urgency, Unknown) else t.urgency.value,
            "refund_asked": t.refund_asked.status,
            "refund_why": list(t.refund_asked.reasons),
            "gate": {"action": t.routing.action, "confidence": t.routing.confidence, "why": t.routing.why, "basis": t.routing.basis},
            "confidence_kind": t.confidence.kind if t.confidence else None,
            "tier": t.tier,
            "ms": round(t.ms, 2),
        }

    def decide(self, body: dict) -> dict:
        ticket = self.ticket(body)
        with tc.use(self.runtime):
            return decisions.handle(ticket, audit=self.audit, table=self.gates)

    def rerank(self, body: dict) -> dict:
        passages = tuple(
            Passage(p.get("id") or f"P-{i + 1}", p.get("title", ""), p.get("text", ""))
            for i, p in enumerate(body.get("passages", ()))
        )
        with tc.use(self.runtime):
            ranked = decisions.rerank(body.get("query", ""), passages, limit=body.get("limit"))
        if isinstance(ranked, Unknown):
            return {"unknown": ranked.reason, "detail": ranked.detail}
        return {
            "ranked": [{"id": p.id, "title": p.title, "score": s.value, "kind": s.kind, "basis": s.basis} for p, s in ranked],
            "note": "relevance scores order this ranking only; the money gate refuses to threshold them",
        }

    def supports(self, body: dict) -> dict:
        p = body.get("passage") or {}
        passage = Passage(p.get("id", "P-1"), p.get("title", ""), p.get("text", ""))
        with tc.use(self.runtime):
            verdict = decisions.supports(body.get("claim", ""), passage)
        return {"status": verdict.status, "reasons": list(verdict.reasons), "evidence": [str(e) for e in verdict.evidence]}

    def why(self, ticket_id: str) -> dict:
        return {"why": self.audit.why(ticket_id), "chain": self.audit.explain(ticket_id)}

    def replay(self, ticket_id: str) -> dict:
        state = self.audit.inputs.get(ticket_id)
        if state is None:
            return {"error": f"no stored state for {ticket_id}"}
        again = replay(state, runtime=self.runtime)
        return {"replayed": again, "note": "same bindings: the tiers declare deterministic=True, so this must match"}

    def config(self) -> dict:
        return {
            "tier": self.tier,
            "implementations": [f"{i.name}@{i.version} ({i.op})" for i in self.runtime.implementations],
            "thresholds": {
                name: {"auto": g.t.auto, "confirm": g.t.confirm, "basis": g.t.basis, "requires_probability": g.require_probability}
                for name, g in self.gates.items()
            },
            "fit": None
            if self.report is None
            else {
                "threshold": self.report.threshold,
                "temperature": self.report.temperature,
                "validation_coverage": self.report.validation_coverage,
                "validation_selective_accuracy": self.report.validation_selective_accuracy,
                "basis": self.report.basis,
            },
        }

    def trace(self) -> dict:
        spans = self.runtime.trace.spans[-20:]
        return {
            "render": self.runtime.trace.render(since=max(0, len(self.runtime.trace.spans) - 20)),
            "model_calls": sum(
                1 for s in self.runtime.trace.spans for a in s.attempts if a.outcome == "answer" and a.implementation.startswith("chat:")
            ),
            "spans": len(self.runtime.trace.spans),
            "recent": [{"op": s.op, "answered_by": s.answered_by, "ms": round(s.total_ms, 2)} for s in spans],
        }


class _Threaded(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(service: Service, port: int) -> _Threaded:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def _send(self, code: int, body: Any, content_type: str = "application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body, indent=1, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            query = dict(re.findall(r"([^?=&]+)=([^&]*)", self.path))
            if path in ("/", "/ui"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/config":
                return self._send(200, service.config())
            if path == "/trace":
                return self._send(200, service.trace())
            if path == "/why":
                return self._send(200, service.why(query.get("ticket", "")))
            self._send(404, {"error": "no such route"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                return self._send(400, {"error": f"bad json: {exc}"})
            t0 = time.perf_counter()
            try:
                if path == "/triage":
                    out = service.triage(body)
                elif path == "/decide":
                    out = service.decide(body)
                elif path == "/rerank":
                    out = service.rerank(body)
                elif path == "/supports":
                    out = service.supports(body)
                elif path == "/replay":
                    out = service.replay(body.get("ticket", ""))
                else:
                    return self._send(404, {"error": "no such route"})
            except Exception as exc:  # noqa: BLE001 - a handler crash is a 500, not a dead server
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            out.setdefault("request_ms", round((time.perf_counter() - t0) * 1e3, 2))
            self._send(200, out)

    server = _Threaded(("127.0.0.1", port), Handler)
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--tier", default="cascade-scored", choices=["rules", "learned", "cascade", "cascade-scored"])
    ap.add_argument("--train", type=Path, default=Path("data/banking77_train.csv"))
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    service = Service(args.tier, args.train if args.tier != "rules" else None)
    server = serve(service, args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"decision service on {url} (tier={args.tier}, no model calls)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
