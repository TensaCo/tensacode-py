"""The economy, tracked properly.

Four goods: food, wood, stone, tools.

**Production** is per person per day and multiplicative in labour, land, skill and capital:

    yield = rate · (0.5 + skill) · health · weather · light · (1 + tools)^0.25   capped by the tile

Labour is the person-day; land is the tile's stock and capacity (crowding is real: people on one
patch share what is there); skill rises with repetition; tools are capital made in workshops from
wood and stone, and they wear out. Nothing here is a price — it is physical output.

**Inventories** are per household. Food is `p.wealth` (already conserved by the simulation); wood,
stone and tools live here. All of them decay.

**Exchange** starts as barter and is allowed to discover money. Each household holds an
*acceptability* estimate per good: how willing it is to take that good in payment for something it
does not itself want. A trade clears only if the buyer holds some good the seller will take —
either because the seller wants it, or because the seller expects to pass it on. Every time that
works, both sides' acceptability for that good rises a little and rivals fall, and neighbours copy
the good that is working. That is a positive feedback with no designated winner: whichever good
wins becomes the numéraire, and it is reported as money only once it settles most trades. Before
that, wanted trades fail for want of a medium — that count is measured, and it falls when money
appears.

**Prices** are not set anywhere. Each settlement runs a call auction per good: households post bids
(what a unit is worth to them, when that beats the going price) and asks (when it is worth less),
the book is crossed, and each pair meets in the middle. Realized ratios update the settlement's
price estimate, so prices are local and move with local scarcity.

**Routes** carry goods between settlements when the price gap beats the cost of the journey; a
share of the load is lost per tile travelled, so prices converge only as far as transport allows.

**Credit** is a claim on future goods. Unpaid debt defaults, and the creditor remembers it.

**Taxes** are levied by a settlement with a leader and a law, into the common store.

Conservation: production adds, consumption/decay/transport loss remove, everything else is a
transfer between holders. `tests/test_civ_economy.py` checks it to rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GOODS = ("food", "wood", "stone", "tools")
FOOD, WOOD, STONE, TOOLS = range(4)
DECAY = (0.0, 0.0025, 0.0002, 0.0045)  # per day; food spoilage is handled by the simulation
LOST_KEY = {WOOD: "wood_lost", STONE: "stone_lost", TOOLS: "tools_lost"}
# What a good is like to carry and keep. Physical properties, not a ranking of money: durability and
# portability tilt the feedback, they do not decide it.
DURABILITY = (0.35, 0.75, 0.97, 0.70)
PORTABILITY = (0.90, 0.50, 0.15, 0.80)


@dataclass
class Market:
    settlement: int
    price: np.ndarray  # food-equivalents per unit, kept as an accounting reference
    volume: np.ndarray
    history: list = field(default_factory=list)


class Economy:
    def __init__(self, sim) -> None:
        self.sim = sim
        cap = len(sim.p.alive)
        nv = max(1, len(sim.villages))
        self.stock = np.zeros((cap, 4), np.float64)  # column FOOD unused: food is p.wealth
        self.accept = 0.22 + 0.10 * sim.rng.random((cap, 4))
        self.markets = [Market(i, np.array([1.0, 0.8, 0.5, 3.0]), np.zeros(4)) for i in range(nv)]
        self.settled = np.zeros(4, np.float64)  # times each good was used as the medium of payment
        self.settled_recent = np.zeros(4, np.float64)
        self.debts: list = []
        self.route_flows: dict = {}
        self.stats: list = []
        self.money: str | None = None
        self.counters = {"trades": 0, "attempts": 0, "for_use": 0, "to_pass_on": 0, "failed_no_medium": 0,
                         "defaults": 0, "loans": 0, "caravans": 0, "tax": 0.0}
        for k in ("craft_wood", "craft_stone", "wood_lost", "stone_lost", "tools_made", "tools_lost"):
            sim.ledger.setdefault(k, 0.0)
        self._endow()

    def _endow(self) -> None:
        """The founders arrive holding something. All four goods are in the initial distribution, so
        none of them is money by default and the competition between them is real. This runs before
        the simulation takes its starting totals, so it is part of the initial condition, not income."""
        idx = self.sim.living()
        if len(idx) == 0:
            return
        for g, scale in ((WOOD, 1.1), (STONE, 0.5), (TOOLS, 0.30)):
            self.stock[idx, g] += self.sim.rng.gamma(1.4, scale, len(idx))

    # ------------------------------------------------------------ plumbing

    def add_market(self, settlement: int) -> None:
        """A new commons opens its own market, with the planet's prices as its starting guess."""
        while len(self.markets) <= settlement:
            ref = self.markets[0].price.copy() if self.markets else np.array([1.0, 0.8, 0.5, 3.0])
            self.markets.append(Market(len(self.markets), ref, np.zeros(4)))

    def grow(self, cap: int) -> None:
        if len(self.stock) >= cap:
            return
        stock = np.zeros((cap, 4), np.float64)
        stock[: len(self.stock)] = self.stock
        accept = 0.22 + 0.10 * self.sim.rng.random((cap, 4))
        accept[: len(self.accept)] = self.accept
        self.stock, self.accept = stock, accept

    def inherit(self, kid: int, parent: int) -> None:
        """A child picks up its parent's sense of what is worth taking in payment. Culture, not genes."""
        if kid < len(self.accept) and parent < len(self.accept):
            self.accept[kid] = np.clip(self.accept[parent] + 0.05 * (self.sim.rng.random(4) - 0.5), 0.02, 1.0)

    def on_death(self, i: int) -> None:
        """Goods pass to the settlement's common store; tools go with their owner."""
        v = self.sim.villages[int(self.sim.p.village[int(i)])]
        v.wood += float(self.stock[i, WOOD])
        v.stone += float(self.stock[i, STONE])
        self.sim.ledger["tools_lost"] += float(self.stock[i, TOOLS])
        self.stock[i] = 0.0

    def tool_bonus(self, idx: np.ndarray) -> np.ndarray:
        """The capital term of the production function."""
        return (1.0 + self.stock[idx, TOOLS]) ** 0.25

    def price(self, settlement: int, good: str | int) -> float:
        g = GOODS.index(good) if isinstance(good, str) else int(good)
        return round(float(self.markets[int(settlement) % len(self.markets)].price[g]), 3)

    def holding(self, i: int, g: int) -> float:
        return float(self.sim.p.wealth[i]) if g == FOOD else float(self.stock[i, g])

    def _move(self, g: int, src: int, dst: int, qty: float) -> None:
        if g == FOOD:
            self.sim.p.wealth[src] -= qty
            self.sim.p.wealth[dst] += qty
        else:
            self.stock[src, g] -= qty
            self.stock[dst, g] += qty

    # ------------------------------------------------------------ capital and decay

    def deposit(self, idx: np.ndarray, g: int, amount: np.ndarray) -> None:
        """What a household keeps of what it carried home (the rest went to the common store)."""
        self.stock[idx, g] += amount

    def make_tools(self, idx: np.ndarray) -> None:
        """Workshops turn wood and stone into tools. Skill raises the yield; the inputs are consumed."""
        from .sim import BUILD, WORKSHOP

        p = self.sim.p
        for v in self.sim.villages:
            benches = v.buildings.get(WORKSHOP, 0)
            if not benches:
                continue
            crew = idx[(p.village[idx] == v.id) & (p.task[idx] == BUILD)][: 4 * benches]
            if len(crew) == 0:
                continue
            wood = min(v.wood * 0.5, 0.35 * len(crew))
            stone = min(v.stone * 0.5, 0.15 * len(crew))
            if wood < 0.1:
                continue
            v.wood -= wood
            v.stone -= stone
            made = (0.30 * wood + 0.25 * stone) * float(np.mean(0.6 + p.skill_build[crew]))
            self.sim.ledger["craft_wood"] += wood
            self.sim.ledger["craft_stone"] += stone
            self.sim.ledger["tools_made"] += made
            self.stock[crew, TOOLS] += made / len(crew)
            p.skill_build[crew] = np.minimum(1.0, p.skill_build[crew] + 0.002)

    def decay(self, idx: np.ndarray) -> None:
        for g in (WOOD, STONE, TOOLS):
            loss = self.stock[idx, g] * DECAY[g]
            self.stock[idx, g] -= loss
            self.sim.ledger[LOST_KEY[g]] += float(loss.sum())

    # ------------------------------------------------------------ what a unit is worth to someone

    def wants(self, idx: np.ndarray) -> np.ndarray:
        """Marginal value of one more unit of each good, in food-equivalents, diminishing in holdings."""
        sim, p = self.sim, self.sim.p
        cold = np.clip((8 - sim.base_temp()[p.y[idx].astype(int) % sim.world.size]) / 12, 0, 1)
        out = np.empty((len(idx), 4))
        out[:, FOOD] = 1.0 + 1.6 * (1 - p.energy[idx])  # hunger makes food dear
        out[:, WOOD] = (0.55 + 0.9 * cold) / (1 + 0.7 * self.stock[idx, WOOD])
        out[:, STONE] = 0.34 / (1 + 0.9 * self.stock[idx, STONE])
        worker = 0.5 * (p.skill_build[idx] + p.skill_forage[idx])
        out[:, TOOLS] = (1.8 + 2.6 * worker) / (1 + 1.4 * self.stock[idx, TOOLS])
        return out

    # ------------------------------------------------------------ markets

    def clear_markets(self) -> None:
        """A call auction per settlement per good, settled in whatever good the seller will accept."""
        sim, p, rng = self.sim, self.sim.p, self.sim.rng
        idx = sim.living()
        if len(idx) == 0:
            return
        self.settled_recent *= 0.97
        for v in sim.villages:
            here = idx[p.village[idx] == v.id]
            if len(here) < 6:
                continue
            if len(here) > 240:  # a market day is a sample of the settlement, not everyone in it
                here = np.sort(rng.choice(here, 240, replace=False))
            m = self.markets[v.id]
            want = self.wants(here)
            row = {int(pid): k for k, pid in enumerate(here)}  # so a trade need not recompute wants
            for g in (WOOD, STONE, TOOLS):
                hold = self.stock[here, g]
                ref = float(m.price[g])
                val = want[:, g]
                bids = [(float(val[k]), int(here[k])) for k in np.flatnonzero((val > ref * 1.02) & (hold < 4.0))]
                asks = [(float(val[k]), int(here[k])) for k in np.flatnonzero((val < ref * 0.98) & (hold > 0.15))]
                if not bids or not asks:
                    # a market with only one side still says something: the price walks toward the
                    # side that is there, so a good nobody will buy gets cheaper.
                    if asks:
                        m.price[g] = round(float(max(0.02, 0.96 * ref + 0.04 * max(a for a, _ in asks))), 4)
                    elif bids:
                        m.price[g] = round(float(0.96 * ref + 0.04 * min(b for b, _ in bids)), 4)
                    m.volume[g] = 0.0
                    continue
                bids.sort(key=lambda t: (-t[0], t[1]))
                asks.sort(key=lambda t: (t[0], t[1]))
                traded, ratio_sum, n = 0.0, 0.0, 0
                for (bp, buyer), (ap, seller) in zip(bids[:40], asks[:40]):
                    if bp <= ap:
                        break  # the book no longer crosses
                    qty = min(0.6, float(self.stock[seller, g]) * 0.5)
                    if qty < 0.05:
                        continue
                    px = (bp + ap) / 2  # haggling meets in the middle
                    if not self._settle(buyer, seller, g, qty, px, v.id, want[row[seller]]):
                        continue
                    traded += qty
                    ratio_sum += px
                    n += 1
                if n:
                    m.price[g] = round(float(0.75 * m.price[g] + 0.25 * (ratio_sum / n)), 4)
                    m.volume[g] = traded
                    m.history.append((sim.day, g, float(m.price[g]), round(traded, 2)))
                    del m.history[:-600]

    def _settle(self, buyer: int, seller: int, g: int, qty: float, price: float, settlement: int,
                seller_wants: np.ndarray | None = None) -> bool:
        """Pay for `qty` of `g` with some good the seller will take. This is where money comes from."""
        owed = qty * price
        self.counters["attempts"] += 1
        # whatever the buyer holds that the seller will take, ranked by how keen the seller is on it
        best, best_score, best_afford = -1, 0.0, 0.0
        for cand in range(4):
            if cand == g:
                continue
            unit = float(self.markets[settlement].price[cand])
            afford = self.holding(buyer, cand) * unit
            if unit <= 0 or afford < 0.05 * owed:
                continue  # too little of it to be worth weighing out
            keenness = float(self.accept[seller, cand]) + 0.25 * DURABILITY[cand] * PORTABILITY[cand]
            if keenness > best_score:
                best, best_score, best_afford = cand, keenness, afford
        if best < 0:
            self.counters["failed_no_medium"] += 1
            return False
        if best_afford < owed:  # buy less rather than not at all — haggling over quantity
            qty *= best_afford / owed
            owed = best_afford
        unit = float(self.markets[settlement].price[best])
        pay = min(owed / unit, self.holding(buyer, best))
        if pay <= 1e-6 or qty <= 1e-6:
            return False
        self._move(g, seller, buyer, qty)
        self._move(best, buyer, seller, pay)
        # taken for its own sake, or to pass on? The second is money-like use.
        sw = seller_wants if seller_wants is not None else self.wants(np.array([seller]))[0]
        for_use = float(sw[best]) > unit
        self.counters["trades"] += 1
        self.counters["for_use" if for_use else "to_pass_on"] += 1
        self.settled[best] += 1
        self.settled_recent[best] += 1
        lift = 0.035 * (1 - float(self.accept[seller, best]))
        self.accept[seller, best] += lift
        self.accept[buyer, best] += 0.5 * lift
        for other in range(4):
            if other != best:
                self.accept[seller, other] *= 0.997
        return True

    def spread_habit(self) -> None:
        """People copy the good their neighbours manage to spend. Slow, imitative, and not decreed."""
        idx = self.sim.living()
        if len(idx) == 0 or self.settled_recent.sum() <= 0:
            return
        lead = int(np.argmax(self.settled_recent))
        share = float(self.settled_recent[lead] / self.settled_recent.sum())
        self.accept[idx, lead] += 0.02 * share * (1 - self.accept[idx, lead])
        np.clip(self.accept, 0.02, 1.0, out=self.accept)

    # ------------------------------------------------------------ routes

    def caravans(self) -> None:
        sim, p = self.sim, self.sim.p
        idx = sim.living()
        for a, va in enumerate(sim.villages):
            for b, vb in enumerate(sim.villages):
                if a == b or sim.day - max(va.last_raid_day, vb.last_raid_day) < 24:
                    continue
                here, there = idx[p.village[idx] == a], idx[p.village[idx] == b]
                if len(here) < 4 or len(there) < 4:
                    continue
                if float(p.alpha_p[here, b].mean()) < 0.4:
                    continue  # you do not send traders to people you do not count as people
                dist = float(sim.world.distance(np.array([va.cy]), np.array([va.cx]), np.array([vb.cy]), np.array([vb.cx]))[0])
                loss = min(0.35, 0.004 * dist)  # binds on the long routes only
                for g in (WOOD, STONE, TOOLS):
                    pa, pb = float(self.markets[a].price[g]), float(self.markets[b].price[g])
                    if pb <= pa * (1 + loss) + 0.03:
                        continue
                    sellers = here[self.stock[here, g] > 0.25]
                    buyers = there[np.asarray(p.wealth[there]) > 0.8]
                    if len(sellers) < 2 or len(buyers) < 2:
                        continue
                    load = min(float(self.stock[sellers, g].sum()) * 0.15,
                               0.2 * float(p.wealth[buyers].sum()) / max(0.05, pb), 60.0)
                    if load < 0.5:
                        continue
                    delivered = load * (1 - loss)
                    paid = delivered * (pa + pb) / 2
                    self.stock[sellers, g] -= load / len(sellers)
                    self.stock[buyers, g] += delivered / len(buyers)
                    p.wealth[buyers] -= paid / len(buyers)
                    p.wealth[sellers] += paid / len(sellers)
                    sim.ledger[LOST_KEY[g]] += load * loss
                    self.route_flows[(a, b)] = self.route_flows.get((a, b), 0.0) + delivered
                    self.counters["caravans"] += 1
                    self.markets[b].price[g] = round(float(0.8 * pb + 0.2 * pa * (1 + loss)), 4)
                    self.markets[a].price[g] = round(float(0.93 * pa + 0.07 * pb), 4)
                    if delivered > 6:
                        sim.log("trade", f"traders carried {delivered:.0f} {GOODS[g]} from {va.name} to {vb.name} at "
                                         f"{(pa + pb) / 2:.2f} food per unit — {loss:.0%} lost on the road",
                                tuple(int(s) for s in sellers[:2]))

    # ------------------------------------------------------------ credit and tax

    def note_debt(self, creditor: int, debtor: int, amount: float) -> bool:
        """Record an obligation for goods that have already changed hands (a gift asked for in hard times)."""
        creditor, debtor = int(creditor), int(debtor)
        if amount <= 0.25 or creditor == debtor:
            return False
        self.debts.append([creditor, debtor, float(amount) * 1.1, self.sim.day + 72, False])
        del self.debts[:-6000]
        self.counters["loans"] += 1
        return True

    def lend(self, creditor: int, debtor: int, amount: float) -> bool:
        p = self.sim.p
        creditor, debtor = int(creditor), int(debtor)
        amount = float(min(amount, p.wealth[creditor]))
        if amount <= 0.25 or creditor == debtor:
            return False
        p.wealth[creditor] -= amount
        p.wealth[debtor] += amount
        return self.note_debt(creditor, debtor, amount)

    def credit_round(self) -> None:
        """The hungry borrow from the comfortable. Lending is refused to anyone already in arrears —
        that is the only credit history there is, and it is enough to produce exclusion."""
        p, idx = self.sim.p, self.sim.living()
        if len(idx) == 0:
            return
        in_arrears = {d[1] for d in self.debts if d[4]}
        owing = {d[1] for d in self.debts if not d[4] and d[2] > 0.01}
        for v in self.sim.villages:
            here = idx[p.village[idx] == v.id]
            if len(here) < 6:
                continue
            # "short" and "comfortable" are relative to this settlement, not to an absolute line, so
            # credit exists in a rich world and in a poor one
            held = np.asarray(p.wealth[here], dtype=float)
            comfortable = max(1.2, float(np.percentile(held, 85)))
            short = [int(i) for i in here[(p.energy[here] < 0.7) & (held < max(0.3, comfortable * 0.1))]]
            rich = [int(i) for i in here[held >= comfortable]]
            if not short or not rich:
                continue
            self.sim.rng.shuffle(short)
            for borrower, lender in zip(short[:12], rich[:12]):
                if borrower in in_arrears or borrower in owing or borrower == lender:
                    continue
                minds = self.sim.minds
                if minds is not None and lender in minds.minds:
                    rel = minds.minds[lender].relation(borrower)
                    if rel["grudge"] > 0.4 or rel["trust"] < 0.25:
                        continue  # you do not lend to someone you have no faith in
                if self.lend(lender, borrower, min(2.5, float(p.wealth[lender]) * 0.2)):
                    owing.add(borrower)

    def settle_debts(self) -> None:
        p = self.sim.p
        for d in self.debts:
            creditor, debtor, owed, due, dead = d
            if dead or owed <= 0.01 or self.sim.day < due:
                continue
            if not p.alive[debtor] or not p.alive[creditor]:
                d[4] = True
                continue
            pay = float(min(p.wealth[debtor] * 0.6, owed))
            if pay > 0.1:
                p.wealth[debtor] -= pay
                p.wealth[creditor] += pay
                d[2] = owed - pay
                d[3] = self.sim.day + 48
                if d[2] <= 0.01:
                    d[4] = True
            else:
                d[4] = True
                self.counters["defaults"] += 1
                self.sim.log("debt", f"{self.sim.name(debtor)} could not repay {self.sim.name(creditor)}",
                             (debtor, creditor), notable=False)
                minds = self.sim.minds
                if minds is not None and creditor in minds.minds:
                    mc = minds.minds[creditor]
                    rel = mc.relation(debtor)
                    rel["grudge"] = min(1.0, rel["grudge"] + 0.3)
                    minds.on_default(mc, debtor)

    def collect_tax(self) -> None:
        """A settlement with a leader and a law takes a share of household food into the store."""
        p, idx = self.sim.p, self.sim.living()
        for v in self.sim.villages:
            if not v.law_share or v.leader < 0 or not p.alive[v.leader]:
                continue
            here = idx[p.village[idx] == v.id]
            if len(here) == 0:
                continue
            take = np.asarray(p.wealth[here]) * 0.04
            p.wealth[here] -= take
            v.food += float(take.sum())
            self.counters["tax"] += float(take.sum())

    # ------------------------------------------------------------ tracking

    @staticmethod
    def gini(values: np.ndarray) -> float:
        v = np.sort(np.asarray(values, dtype=np.float64))
        n = len(v)
        if n == 0 or v.sum() <= 0:
            return 0.0
        return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))

    def net_worth(self, idx: np.ndarray) -> np.ndarray:
        p = self.sim.p
        vil = p.village[idx]
        out = np.asarray(p.wealth[idx], dtype=np.float64).copy()
        for g in (WOOD, STONE, TOOLS):
            px = np.array([m.price[g] for m in self.markets])
            out += self.stock[idx, g] * px[vil]
        return out

    def record(self) -> None:
        from .sim import TASKS

        sim, p = self.sim, self.sim.p
        idx = sim.living()
        if len(idx) == 0:
            return
        total = float(self.settled.sum())
        lead = int(np.argmax(self.settled)) if total > 0 else FOOD
        share = float(self.settled[lead] / total) if total else 0.0
        self.money = GOODS[lead] if (total > 200 and share > 0.55) else None
        worth = self.net_worth(idx)
        labour = np.bincount(p.task[idx], minlength=12)
        self.stats.append({
            "day": sim.day,
            "price": {v.name: {GOODS[g]: round(float(self.markets[v.id].price[g]), 3) for g in (WOOD, STONE, TOOLS)} for v in sim.villages},
            "volume": {GOODS[g]: round(float(sum(m.volume[g] for m in self.markets)), 2) for g in (WOOD, STONE, TOOLS)},
            "held": {GOODS[g]: round(float(self.stock[idx, g].sum()), 1) for g in (WOOD, STONE, TOOLS)},
            "money": self.money,
            "money_share": round(share, 3),
            "accept": {GOODS[g]: round(float(self.accept[idx, g].mean()), 3) for g in range(4)},
            "gini": round(self.gini(worth), 3),
            "median_worth": round(float(np.median(worth)), 2),
            "deprived": round(float((p.energy[idx] < 0.35).mean()), 3),
            "tools_pc": round(float(self.stock[idx, TOOLS].mean()), 3),
            "debt": round(sum(d[2] for d in self.debts if not d[4]), 1),
            "defaults": self.counters["defaults"],
            "trades": self.counters["trades"],
            "failed": self.counters["failed_no_medium"],
            "labour": {TASKS[i]: int(c) for i, c in enumerate(labour) if c},
            "routes": {f"{sim.villages[a].name}->{sim.villages[b].name}": round(q, 1) for (a, b), q in sorted(self.route_flows.items())},
        })
        del self.stats[:-800]

    def totals(self) -> dict:
        p = self.sim.p
        alive = p.alive[: p.n]
        return {GOODS[g]: float(self.stock[: p.n][alive, g].sum()) for g in (WOOD, STONE, TOOLS)}
