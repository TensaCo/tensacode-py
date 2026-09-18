"""The simulation's invariants: determinism, conservation, mortality, and that minds explain themselves."""

import numpy as np
import pytest

from research.civ_sim.sim import Simulation


def run(seed: int = 1, days: int = 14, minds: bool = False, people: int = 400, focal: int = 0) -> Simulation:
    s = Simulation(seed=seed, people=people, focal=focal, minds=minds, size=64, villages=3)
    for _ in range(days):
        s.step()
    return s


def fingerprint(s: Simulation) -> tuple:
    p, idx = s.p, s.living()
    return (len(idx), round(float(p.y[idx].sum()), 3), round(float(p.x[idx].sum()), 3), round(float(p.energy[idx].sum()), 3),
            round(float(s.world.food.sum()), 3), s.counters["births"], s.counters["deaths"], round(float(p.share[idx].sum()), 3))


def test_same_seed_gives_the_same_world():
    assert fingerprint(run(4)) == fingerprint(run(4))
    assert fingerprint(run(4)) != fingerprint(run(5))


def test_same_seed_gives_the_same_world_with_minds_too():
    a = run(11, days=6, minds=True, people=300, focal=10)
    b = run(11, days=6, minds=True, people=300, focal=10)
    assert fingerprint(a) == fingerprint(b)
    assert sorted(a.minds.minds) == sorted(b.minds.minds)
    assert a.counters["claims_transmitted"] == b.counters["claims_transmitted"]


def test_resources_are_conserved_to_rounding():
    """Every sink is named. Household goods and the workshop are part of the accounting
    (see tests/test_civ_economy.py for the same check over a longer run)."""
    s = run(2, days=20)
    start, now, L, W = s.world_total_start, s.totals(), s.ledger, s.world.ledger
    food_err = start["food"] + W["regrowth_food"] - L["eaten"] - L["spoiled"] - L["frost_loss"] - W["build_food"] - L.get("herd_lost", 0.0) - now["food"]
    wood_err = start["wood"] + W["regrowth_wood"] - L["wood_burned"] - W["build_wood"] - L["craft_wood"] - L["wood_lost"] - now["wood"]
    stone_err = start["stone"] - L["stone_used"] - L["craft_stone"] - L["stone_lost"] - now["stone"]
    tools_err = start["tools"] + L["tools_made"] - L["tools_lost"] - now["tools"]
    for err, total in ((food_err, now["food"]), (wood_err, now["wood"]), (stone_err, now["stone"]), (tools_err, now["tools"])):
        assert abs(err) <= max(1e-6, 1e-9 * total)


def test_stocks_never_go_negative():
    s = run(3, days=20)
    assert s.world.food.min() >= 0 and s.world.wood.min() >= 0 and s.world.stone.min() >= 0
    assert min(v.food for v in s.villages) >= -1e-9
    idx = s.living()
    assert s.p.wealth[idx].min() >= -1e-9 and s.p.energy[idx].min() >= 0 and s.p.health[idx].min() >= 0


def test_everyone_stays_on_the_planet():
    s = run(6, days=12)
    idx = s.living()
    assert s.p.y[idx].min() >= 0 and s.p.y[idx].max() < s.world.size
    assert s.p.x[idx].min() >= 0 and s.p.x[idx].max() < s.world.size


def test_a_death_removes_someone_and_is_remembered():
    s = run(7, days=6)
    idx = s.living()
    victim = int(idx[0])
    before = len(s.living())
    s.kill(victim, "illness")
    assert len(s.living()) == before - 1
    assert not s.p.alive[victim] and s.p.died[victim] == s.day
    assert any("died" in text for _, text in s.life.get(victim, []))


def test_people_sleep_at_night_and_work_in_the_day():
    s = run(8, days=6)
    idx = s.living()
    light = s.light_at(s.p.y[idx], s.p.x[idx])
    dark, bright = light < 0.2, light > 0.5
    if dark.any() and bright.any():
        assert s.p.asleep[idx][dark].mean() > s.p.asleep[idx][bright].mean()


@pytest.mark.parametrize("seed", [1, 2])
def test_minds_decide_and_can_explain_themselves(seed: int):
    s = run(seed, days=8, minds=True, people=300, focal=12)
    assert s.minds is not None and len(s.minds.minds) > 0
    explained = 0
    for pid, m in s.minds.minds.items():
        info = s.minds.inspect(pid)
        assert info["focal"] and info["claims"] > 0
        assert set(info["affect"]) >= {"valence", "arousal", "integration", "self_attention"}
        if info["explain"]:
            joined = "\n".join(info["explain"])
            assert "decided" in joined
            assert "choose:expected_valence" in joined or "rule:" in joined or "observed in" in joined
            explained += 1
        assert all("day" in t and "text" in t for t in info["thoughts"])
    assert explained > 0  # at least one mind can show the chain behind its last decision


def test_every_living_person_has_a_mind():
    """Full fidelity: no background tier, no cohorts, nobody left as a row of numbers."""
    s = run(9, days=6, minds=True, people=120)
    idx = s.living()
    assert len(s.minds.minds) == len(idx)
    assert set(int(i) for i in idx) == set(s.minds.minds)
    assert all(s.p.rich[i] >= 0 for i in idx)


def test_a_newborn_gets_a_mind_and_a_dead_person_loses_theirs():
    s = run(10, days=8, minds=True, people=120)
    idx = s.living()
    victim = int(idx[0])
    assert victim in s.minds.minds
    s.kill(victim, "illness")
    s.step()
    assert victim not in s.minds.minds
    assert len(s.minds.minds) == len(s.living())


def test_determinism_reaches_individual_beliefs_and_sentences():
    """Same seed, same world down to the grain of what each mind believes and said.

    Aggregate counters can match while a parser tie-break quietly moves which reading a listener
    took, so this hashes every claim, utterance and decision rather than trusting the totals.
    """
    import hashlib

    def fingerprint(seed: int):
        s = Simulation(seed=seed, people=120, minds=True)
        for _ in range(8):
            s.step()
        h = lambda xs: hashlib.sha256("\n".join(sorted(xs)).encode()).hexdigest()
        return (
            h(f"{r.claim.subject.id}|{r.claim.predicate}|{r.claim.object}"
              for m in s.minds.minds.values() for r in m.store._claims.values()),
            h(f"{ln['act']}|{ln['text']}" for m in s.minds.minds.values() for ln in m.said),
            h(f"{pid}:{m.decision}" for pid, m in s.minds.minds.items()),
        )

    assert fingerprint(5) == fingerprint(5)
    assert fingerprint(5) != fingerprint(6)
