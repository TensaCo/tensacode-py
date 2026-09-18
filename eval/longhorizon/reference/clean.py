"""Reference solution for own/data_cleaning."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

RAW, OUT = Path("/app/raw"), Path("/app/out")
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y")


def parse_amount(text: str) -> float | None:
    t = (text or "").strip()
    if not t or t.lower() in {"n/a", "na", "none", "-"}:
        return None
    negative = t.startswith("(") and t.endswith(")") or t.lstrip("€$").startswith("-") or t.startswith("-")
    digits = re.sub(r"[^0-9.]", "", t.replace(",", ""))
    if not digits or digits.count(".") > 1:
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def parse_date(text: str) -> str | None:
    t = (text or "").strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


rows_in = dropped = duplicates = 0
clean: dict[str, dict] = {}
for path in sorted(RAW.glob("*.csv")):
    for row in csv.DictReader(path.read_text().splitlines()):
        rows_in += 1
        amount, when = parse_amount(row["amount"]), parse_date(row["date"])
        if amount is None or when is None:
            dropped += 1
            continue
        oid = row["order_id"].strip()
        entry = {"order_id": oid, "date": when, "region": row["region"].strip().lower(), "category": row["category"].strip().lower(), "amount": round(amount, 2)}
        if oid in clean:
            duplicates += 1
            continue
        clean[oid] = entry

by_region: dict[str, float] = {}
by_month: dict[str, float] = {}
by_cat: dict[str, float] = {}
for e in clean.values():
    by_region[e["region"]] = round(by_region.get(e["region"], 0) + e["amount"], 2)
    by_month[e["date"][:7]] = round(by_month.get(e["date"][:7], 0) + e["amount"], 2)
    by_cat[e["category"]] = round(by_cat.get(e["category"], 0) + e["amount"], 2)

OUT.mkdir(parents=True, exist_ok=True)
answers = {"rows_in": rows_in, "rows_dropped": dropped, "duplicates_removed": duplicates,
           "net_revenue": round(sum(e["amount"] for e in clean.values()), 2), "revenue_by_region": by_region,
           "best_month": max(by_month, key=by_month.get), "top_category": max(by_cat, key=by_cat.get)}
(OUT / "answers.json").write_text(json.dumps(answers, indent=1))
with (OUT / "clean.csv").open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["order_id", "date", "region", "category", "amount"])
    for e in sorted(clean.values(), key=lambda e: (e["date"], e["order_id"])):
        writer.writerow([e["order_id"], e["date"], e["region"], e["category"], f"{e['amount']:.2f}"])
print(json.dumps(answers, indent=1))
