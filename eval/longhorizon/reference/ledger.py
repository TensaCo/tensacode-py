"""Reference solution for own/coding_ledger (exact cents, strict output formats)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

STORE = Path("/app/ledger.json")


def fail(reason: str) -> None:
    print(f"error: {reason}", file=sys.stderr)
    raise SystemExit(2)


def load() -> dict:
    if not STORE.exists():
        return {"next_id": 1, "expenses": []}
    try:
        return json.loads(STORE.read_text())
    except json.JSONDecodeError:
        fail("ledger.json is not valid JSON")


def save(data: dict) -> None:
    STORE.write_text(json.dumps(data, indent=1))


def cents(text: str) -> int:
    try:
        value = Decimal(text)
    except InvalidOperation:
        fail(f"{text!r} is not an amount")
    if -value.as_tuple().exponent > 2:
        fail("amounts have at most two decimal places")
    return int(value.scaleb(2).to_integral_value())


def money(c: int) -> str:
    return f"{Decimal(c) / 100:.2f}"


def parse_date(text: str | None) -> str:
    if text is None:
        return date.today().isoformat()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError:
        fail(f"{text!r} is not a date (YYYY-MM-DD)")


def ordered(expenses: list[dict]) -> list[dict]:
    return sorted(expenses, key=lambda e: (e["date"], e["id"]))


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("add", add_help=False)
    a.add_argument("amount")
    a.add_argument("category")
    a.add_argument("description", nargs="+")
    a.add_argument("--date")
    ls = sub.add_parser("list", add_help=False)
    ls.add_argument("--category")
    ls.add_argument("--month")
    bal = sub.add_parser("balance", add_help=False)
    bal.add_argument("--month")
    ex = sub.add_parser("export", add_help=False)
    ex.add_argument("path")
    de = sub.add_parser("delete", add_help=False)
    de.add_argument("id")
    args, unknown = ap.parse_known_args()
    if unknown or args.cmd is None:
        fail(f"unknown arguments {unknown}" if unknown else "no command")
    data = load()

    if args.cmd == "add":
        amount, when = cents(args.amount), parse_date(args.date)
        entry = {"id": data["next_id"], "date": when, "amount": amount, "category": args.category, "description": " ".join(args.description)}
        data["expenses"].append(entry)
        data["next_id"] += 1
        save(data)
        print(f"added #{entry['id']} {money(amount)} {entry['category']}")
    elif args.cmd == "list":
        rows = [e for e in ordered(data["expenses"]) if (not args.category or e["category"] == args.category) and (not args.month or e["date"].startswith(args.month))]
        for e in rows:
            print(f"#{e['id']} {e['date']} {money(e['amount'])} {e['category']} {e['description']}")
        print(f"total {money(sum(e['amount'] for e in rows))}")
    elif args.cmd == "balance":
        rows = [e for e in data["expenses"] if not args.month or e["date"].startswith(args.month)]
        totals: dict[str, int] = {}
        for e in rows:
            totals[e["category"]] = totals.get(e["category"], 0) + e["amount"]
        for name in sorted(totals):
            if totals[name]:
                print(f"{name} {money(totals[name])}")
        print(f"total {money(sum(totals.values()))}")
    elif args.cmd == "export":
        rows = ordered(data["expenses"])
        with open(args.path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "date", "amount", "category", "description"])
            for e in rows:
                writer.writerow([e["id"], e["date"], money(e["amount"]), e["category"], e["description"]])
        print(f"exported {len(rows)} rows")
    elif args.cmd == "delete":
        if not args.id.isdigit() or not any(e["id"] == int(args.id) for e in data["expenses"]):
            fail(f"no expense #{args.id}")
        data["expenses"] = [e for e in data["expenses"] if e["id"] != int(args.id)]
        save(data)
        print(f"deleted #{args.id}")


if __name__ == "__main__":
    main()
