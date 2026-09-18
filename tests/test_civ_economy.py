"""The economy's invariants: goods are conserved, stocks never go negative, prices come from trade,
money is not decreed, and debt is a claim that can fail."""

import numpy as np
import pytest

from research.civ_sim.economy import GOODS, STONE, TOOLS, WOOD, Economy
from research.civ_sim.sim import Simulation


def run(seed: int = 1, days: int = 30, people: int = 400) -> Simulation:
    s = Simulation(seed=seed, people=people, focal=0, size=64, villages=3, minds=False)
    s._start = s.totals()
    for _ in range(days):
        s.step()
    return s


def test_every_good_is_conserved_to_rounding():
    s = run(1, days=40)
    start, now, L, W = s._start, s.totals(), s.ledger, s.world.ledger
    errors = {
        "food": start["food"] + W["regrowth_food"] - L["eaten"] - L["spoiled"] - L["frost_loss"] - W["build_food"] - L.get("herd_lost", 0.0) - now["food"],
        "wood": start["wood"] + W["regrowth_wood"] - L["wood_burned"] - W["build_wood"] - L["craft_wood"] - L["wood_lost"] - now["wood"],
        "stone": start["stone"] - L["stone_used"] - L["craft_stone"] - L["stone_lost"] - now["stone"],
        "tools": start["tools"] + L["tools_made"] - L["tools_lost"] - now["tools"],
    }
    for good, err in errors.items():
        assert abs(err) <= max(1e-6, 1e-9 * abs(now[good])), f"{good} leaks: {err}"


def test_no_household_ever_holds_a_negative_amount():
    s = run(2, days=30)
    idx = s.living()
    assert s.economy.stock[idx].min() >= -1e-9
    assert s.p.wealth[idx].min() >= -1e-9
    assert min(v.wood for v in s.villages) >= -1e-9 and min(v.stone for v in s.villages) >= -1e-9


def test_trade_actually_happens_and_moves_goods():
    s = run(3, days=30)
    e = s.economy
    assert e.counters["trades"] > 50
    assert e.counters["attempts"] >= e.counters["trades"]
    assert float(e.stock[s.living(), WOOD].sum()) > 0


def test_prices_are_local_and_move():
    """Nothing writes a price. They start equal and end up different per settlement."""
    s = run(4, days=40)
    e = s.economy
    for good in (WOOD, STONE):
        prices = [e.price(v.id, good) for v in s.villages]
        assert len(set(prices)) > 1, f"{GOODS[good]} has one price everywhere: {prices}"
    assert e.price(0, WOOD) != 0.8  # the starting reference is gone


def test_money_is_discovered_not_assigned():
    """Whichever good ends up settling most trades is the money — and it is not always food."""
    s = run(5, days=60, people=600)
    e = s.economy
    assert e.settled.sum() > 0
    # acceptability moved away from where it started (all four within 0.22..0.32)
    mean = e.accept[s.living()].mean(axis=0)
    assert mean.max() > 0.45, f"no good became more acceptable than another: {mean}"
    top = int(np.argmax(e.settled))
    assert e.settled[top] / e.settled.sum() > 0.4
    assert e.money in (None,) + GOODS


def test_a_loan_is_repaid_or_defaults_and_the_creditor_remembers():
    s = run(6, days=6)
    idx = s.living()
    lender, borrower = int(idx[0]), int(idx[1])
    s.p.wealth[lender], s.p.wealth[borrower] = 20.0, 0.0
    before = float(s.p.wealth[lender] + s.p.wealth[borrower])
    defaults_before = s.economy.counters["defaults"]
    assert s.economy.lend(lender, borrower, 5.0)
    mine = s.economy.debts[-1]
    assert abs(float(s.p.wealth[lender] + s.p.wealth[borrower]) - before) < 1e-9  # a loan moves, not makes
    s.p.wealth[borrower] = 0.0  # the borrower has nothing when it comes due
    s.day += 100
    s.economy.settle_debts()
    assert s.economy.counters["defaults"] > defaults_before
    assert mine[4] is True and mine[2] > 0  # written off, still on the books as unpaid


def test_tools_raise_what_labour_produces():
    """Capital is not decoration: the same person with tools harvests more."""
    s = run(7, days=4)
    idx = s.living()[:20]
    bare = s.economy.tool_bonus(idx)
    s.economy.stock[idx, TOOLS] = 3.0
    assert float(s.economy.tool_bonus(idx).mean()) > float(bare.mean()) * 1.3


def test_gini_reads_zero_on_equality_and_high_on_concentration():
    assert Economy.gini(np.ones(50)) == pytest.approx(0.0, abs=1e-9)
    concentrated = np.zeros(50)
    concentrated[0] = 100.0
    assert Economy.gini(concentrated) > 0.9


def test_cropland_wears_out_and_fallow_ground_comes_back():
    s = run(8, days=1)
    w = s.world
    field = np.flatnonzero((w.building.reshape(-1) == 7) & (w.build_progress.reshape(-1) >= 1.0))
    if len(field) == 0:  # no field finished this early; plant one by hand
        w.building[10, 10], w.build_progress[10, 10], w.food_cap[10, 10] = 7, 1.0, 40.0
        w.fertility[10, 10] = w.fertility_base[10, 10] = 0.9
    before = float(w.fertility[10, 10])
    for _ in range(200):
        w.food[10, 10] = 0.0  # harvested every day, so it grows every day
        w.regrow(1, np.ones((1, w.size), np.float32), np.zeros((w.size, w.size), np.float32),
                 np.full((w.size, w.size), 15.0, np.float32))
    worked = float(w.fertility[10, 10])
    assert worked < before * 0.95, f"cropping did not wear the soil: {before} -> {worked}"
    w.building[10, 10] = 0  # left fallow
    for _ in range(400):
        w.regrow(1, np.ones((1, w.size), np.float32), np.zeros((w.size, w.size), np.float32),
                 np.full((w.size, w.size), 15.0, np.float32))
    assert float(w.fertility[10, 10]) > worked
    assert float(w.fertility[10, 10]) <= float(w.fertility_base[10, 10]) + 1e-9


def test_the_economy_is_deterministic_by_seed():
    a, b, c = run(9, days=20), run(9, days=20), run(10, days=20)

    def fingerprint(s):
        idx = s.living()
        return (round(float(s.economy.stock[idx].sum()), 6), s.economy.counters["trades"],
                tuple(round(float(m.price[g]), 4) for m in s.economy.markets for g in (WOOD, STONE, TOOLS)))

    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)


def test_a_short_commons_feeds_dependents_first_and_a_full_one_feeds_everyone():
    """When the store cannot cover everyone, children are served before able adults who brought
    nothing home. When it can cover everyone, it does — nobody starves beside a full granary."""
    p, idx = None, None

    def one_day(store: float):
        s = run(11, days=10, people=300)
        p, idx = s.p, s.living()
        age = s.age(idx)
        p.carry_food[idx] = 0.0  # nobody gathered today
        p.wealth[idx] = 0.0      # and nobody has anything put by
        for v in s.villages:
            v.food = store
        before = np.array(p.energy[idx], dtype=float)
        s._eat(idx)
        gained = np.array(p.energy[idx], dtype=float) - before
        adults = (age >= 12) & (age <= 58) & (p.sick[idx] <= 0.3)
        return gained, adults, age < 12

    lean, adults, kids = one_day(12.0)  # far less than the village needs
    assert adults.any() and kids.any()
    assert lean[kids].mean() > lean[adults].mean(), "rationing did not put children first"

    fat, adults, kids = one_day(5000.0)  # more than enough
    assert abs(fat[kids].mean() - fat[adults].mean()) < 0.02, "a full store still rationed"


def test_credit_appears_when_the_commons_runs_short():
    """Borrowing is not scheduled: it appears when the store cannot cover everyone and some
    households have nothing. In a fat year nobody borrows, and that is the correct behaviour."""
    s = run(12, days=40, people=300)
    e = s.economy
    before = e.counters["loans"]
    idx = s.living()
    for v in s.villages:
        v.food = 0.0  # a bad winter: the common store is empty
    s.p.wealth[idx] = 0.0
    s.p.energy[idx] = 0.3
    rich = idx[: max(2, len(idx) // 10)]
    s.p.wealth[rich] = 30.0  # a few households still have something put by
    for _ in range(3):
        s.economy.credit_round()
    assert e.counters["loans"] > before, "nobody lent to the hungry when the store was empty"
    assert e.counters["defaults"] <= e.counters["loans"]
    owed = [d for d in e.debts if not d[4]]
    assert owed and all(d[2] > 0 for d in owed)
