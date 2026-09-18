"""Constrained purchasing, as a mind: read a brief, survey the catalog, choose under hard constraints, revise on price changes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import tensorcode as tc
from tensorcode.backends.builtin import IN_PROCESS
from tensorcode.cognition import Fragment, Rule

from ..mind import BY_PRIORITY, Finish, MindSpec, Press, Wait, controls, knowledge, objects, one, order

V = tc.Var
BRIEF, CATALOG = tc.Ref("brief:main"), tc.Ref("catalog:store")
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


# ----------------------------------------------------- language: the brief


@dataclass(frozen=True)
class BriefText:
    text: str


@dataclass(frozen=True)
class Brief:
    min_size: int
    min_rating: float
    max_price: float
    deliver_by: date


_SIZE = re.compile(r"(?:at least|no smaller than)\s+(\d{2})|(\d{2})(?:-inch|\")\s*or larger|(\d{2})\"\+")
_RATING = re.compile(r"(?:rated|rating of at least)\s+(\d\.\d)|(\d\.\d)\+\s*stars")
_PRICE = re.compile(r"(?:under|max|budget|no more than|up to)\s+\$(\d+(?:\.\d\d)?)")
_BY = re.compile(r"(?:by|no later than)\s+(?:[A-Z][a-z]{2},\s+)?([A-Z][a-z]{2})[a-z]*\.?\s+(\d{1,2})")


@tc.implementation("parse", name="brief-rules", version="1", accepts=lambda r: isinstance(r.subject, BriefText) and r.target is Brief, profile=IN_PROCESS)
def parse_brief(request: tc.Request) -> Brief | tc.Unknown:
    t = request.subject.text
    size, rating, price, by = _SIZE.search(t), _RATING.search(t), _PRICE.search(t), _BY.search(t)
    if not (size and rating and price and by):
        return tc.Unknown("brief_incomplete", f"size={bool(size)} rating={bool(rating)} price={bool(price)} by={bool(by)}")
    first = lambda m: next(g for g in m.groups() if g)  # noqa: E731
    return Brief(int(first(size)), float(first(rating)), float(price[1]), date(2026, MONTHS[by[1].lower()], int(by[2])))


# ------------------------------------------------ language: a product card


@dataclass(frozen=True)
class CardText:
    name: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Product:
    name: str
    category: str
    size: int | None
    price: float
    rating: float
    in_stock: bool
    arrives: date


@tc.implementation("parse", name="product-card-rules", version="1", accepts=lambda r: isinstance(r.subject, CardText) and r.target is Product, profile=IN_PROCESS)
def parse_card(request: tc.Request) -> Product | tc.Unknown:
    card: CardText = request.subject
    text = " | ".join(card.lines)
    price, rating, arrives = re.search(r"\$(\d+\.\d\d)", text), re.search(r"(\d\.\d) ★", text), re.search(r"Arrives \w{3}, (\w{3}) (\d{1,2})", text)
    size = re.search(r'(\d{2})"', card.name)
    if not (price and rating and arrives):
        return tc.Unknown("card_incomplete")
    category = next((w for w in ("Monitor", "Keyboard", "Webcam") if w in card.name), "other")
    return Product(card.name, category, int(size[1]) if size else None, float(price[1]), float(rating[1]), "In stock" in text, date(2026, MONTHS[arrives[1].lower()], int(arrives[2])))


# ----------------------------------------------------- spontaneous thoughts


def _read_brief(b, mind):
    if b["x"].id.startswith("text:Brief#"):
        brief = tc.parse(BriefText(b["text"]), Brief)
        if isinstance(brief, Brief):
            yield knowledge([(tc.Claim(BRIEF, k, getattr(brief, k)), None) for k in ("min_size", "min_rating", "max_price", "deliver_by")], "doc:brief", "brief-rules@1")
        else:
            yield knowledge([(tc.Claim(BRIEF, "unreadable", brief.reason), None)], "doc:brief", "brief-rules@1")


def _read_card(b, mind):
    name = b["name"]
    lines = tuple(r.claim.object for r in mind.claims(predicate="reads") if r.claim.subject.id.startswith(f"text:{name}#"))
    product = tc.parse(CardText(name, lines), Product)
    if isinstance(product, tc.Unknown):
        return
    ref = tc.Ref(f"product:{name}")
    facts = tc.Ref(f"scope:product:{name}")
    claims = [(tc.Claim(ref, k, getattr(product, k), scope=facts), None) for k in ("category", "size", "rating", "in_stock", "arrives") if getattr(product, k) is not None]
    page = next((one(mind, p, "label") for p in controls(mind, section="Pages") if one(mind, p, "current")), None)
    if page:
        claims.append((tc.Claim(ref, "on_page", page, scope=facts), None))
    yield Fragment(tc.Ref(f"doc:card:{name}"), tuple(claims), snapshot_of=facts, method="product-card-rules@1")
    yield _price(name, product.price, f"doc:card:{name}")


def _price(name: str, price: float, source: str) -> Fragment:
    """Prices live in their own scope per product: a newer reading replaces the older belief."""
    scope = tc.Ref(f"scope:price:{name}")
    return Fragment(tc.Ref(source), ((tc.Claim(tc.Ref(f"product:{name}"), "price", price, scope=scope), None),), snapshot_of=scope, method="read-price")


def _read_dialog(b, mind):
    m = re.search(r"Price updated (.+?) is now \$(\d+\.\d\d)", b["text"])
    if m:
        yield _price(m[1], float(m[2]), "obs:price-dialog")


def _page_seen(b, mind):
    yield knowledge([(tc.Claim(CATALOG, "seen_page", b["label"]), None)], "obs:pages", "read")


def _outcome(b, mind):
    text = b["text"]
    if re.search(r"Order ORD-\d+ placed", text):
        yield knowledge([(tc.Claim(CATALOG, "ordered", True), None)], b["x"].id, "read")
    if "Reported: no suitable product" in text:
        yield knowledge([(tc.Claim(CATALOG, "reported", True), None)], b["x"].id, "read")


RULES = [
    Rule("read_brief", ((V("x"), "reads", V("text")),), _read_brief),
    Rule("read_product_card", ((V("b"), "label", "Add to cart"), (V("b"), "in", V("name"))), _read_card),
    Rule("read_price_update", ((tc.Ref("ui:dialog"), "shows", V("text")),), lambda b, m: _read_dialog({"text": b["text"]}, m)),
    Rule("page_seen", ((V("p"), "in", "Pages"), (V("p"), "current", True), (V("p"), "label", V("label"))), _page_seen),
    Rule("order_or_report_outcome", ((V("x"), "reads", V("text")),), _outcome),
    Rule("announced_outcome", ((V("x"), "announces", V("text")),), _outcome),
]


# ------------------------------------------------------------ deliberation


def _products(mind: tc.Store) -> list[tc.Ref]:
    return sorted({r.claim.subject for r in mind.claims(predicate="category")})


def _req(key):
    return lambda p, mind: (lambda v, need: tc.Unknown("not_read") if v is None or need is None else key[1](v, need))(one(mind, p, key[0]), one(mind, BRIEF, key[2]))


CONSTRAINTS = (
    tc.Constraint("is_a_monitor", lambda p, mind: one(mind, p, "category") == "Monitor"),
    tc.Constraint("big_enough", _req(("size", lambda v, n: v >= n, "min_size"))),
    tc.Constraint("rated_high_enough", _req(("rating", lambda v, n: v >= n, "min_rating"))),
    tc.Constraint("within_budget", _req(("price", lambda v, n: v <= n, "max_price"))),
    tc.Constraint("arrives_in_time", _req(("arrives", lambda v, n: v <= n, "deliver_by"))),
    tc.Constraint("in_stock", lambda p, mind: one(mind, p, "in_stock") is True),
)
CHEAPEST = tc.Objective("cheapest", "Lowest price; higher rating breaks ties", lambda p, mind: -one(mind, p, "price") + one(mind, p, "rating") / 1000)


def choose_product(mind: tc.Store) -> tc.Ref | tc.Unknown:
    """The brief ranks by price then rating. Products equal on both are equally correct answers:
    break that tie explicitly (earliest arrival, then name) instead of treating it as no answer."""
    best = tc.choose(_products(mind), objective=CHEAPEST, given=mind, constraints=CONSTRAINTS)
    if isinstance(best, tc.Unknown) and best.reason == "tie_within_margin" and best.candidates:
        top = best.candidates[0][1].value
        tied = [p for p, score in best.candidates if score.value == top]
        return min(tied, key=lambda p: (one(mind, p, "arrives"), p.id))
    return best


def _cart(mind: tc.Store) -> list[str]:
    return [m[1] for r in mind.claims(predicate="reads") if r.claim.subject.id.startswith("text:Cart#") and (m := re.match(r"(.+) — \$\d", r.claim.object))]


def intentions(mind: tc.Store) -> list[object]:
    if one(mind, CATALOG, "ordered") or one(mind, CATALOG, "reported"):
        return [Finish("order placed" if one(mind, CATALOG, "ordered") else "reported no match", priority=100)]
    pages = controls(mind, section="Pages")
    numbered = [p for p in pages if (one(mind, p, "label") or "").startswith("Page ")]
    unseen = [p for p in numbered if one(mind, p, "label") not in objects(mind, CATALOG, "seen_page")]
    dialog = one(mind, tc.Ref("ui:dialog"), "shows") or ""
    best = choose_product(mind) if not unseen else None
    best_name = best.id.removeprefix("product:") if isinstance(best, tc.Ref) else None

    if dialog:
        if dialog.startswith("Price updated"):
            want = "Add anyway" if best_name and best_name in dialog else "Keep shopping"
        elif dialog.startswith("Place order"):
            want = "Place order" if best_name and _cart(mind) == [best_name] else "Cancel"
        else:
            want = "Report" if unseen == [] and isinstance(best, tc.Unknown) else "Cancel"
        return [Press(b, f"dialog: {want}", priority=90) for b in controls(mind, role="button", label=want)]
    if one(mind, BRIEF, "min_size") is None:
        return [Wait(20, "reading the brief", priority=1)]
    if unseen:
        return [Press(unseen[0], f"survey {one(mind, unseen[0], 'label').lower()}", priority=70)]
    if isinstance(best, tc.Unknown):
        return [Press(b, f"no product qualifies ({best.reason})", priority=60) for b in controls(mind, role="button", label="No product meets the brief")]
    todo: list[object] = []
    for name in _cart(mind):
        if name != best_name:
            todo += [Press(b, f"remove {name}", priority=55) for b in controls(mind, role="button", label=f"Remove {name}")]
    if best_name not in _cart(mind):
        add = [b for b in controls(mind, role="button", label="Add to cart") if one(mind, b, "in") == best_name]
        if add:
            todo.append(Press(add[0], f"add cheapest qualifying: {best_name} (${one(mind, best, 'price'):.2f})", priority=50))
        else:
            where = one(mind, best, "on_page")
            todo += [Press(p, f"go to {where} for {best_name}", priority=45 + order(p, mind)) for p in numbered if one(mind, p, "label") == where]
    elif not todo:
        todo += [Press(b, "checkout", priority=40) for b in controls(mind, role="button", label="Checkout")]
    return todo or [Wait(25, "waiting for the page", priority=1)]


SPEC = MindSpec("shop", RULES, intentions, BY_PRIORITY, max_cycles=120)
BINDINGS = [parse_brief, parse_card]
