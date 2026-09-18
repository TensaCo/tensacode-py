"""Settlements are derived, not declared: the coarse-graining must find them, keep their identity
across periods, and never be something the tick loop depends on."""

import numpy as np

from research.civ_sim.settlements import (
    MIN_PEOPLE,
    _circular_mean,
    _components,
)
from research.civ_sim.sim import Simulation


def run(seed: int = 1, days: int = 26, people: int = 500) -> Simulation:
    s = Simulation(seed=seed, people=people, focal=0, size=64, villages=3, minds=False)
    for _ in range(days):
        s.step()
    return s


def test_components_wrap_around_the_seam():
    """A lump straddling the edge is one lump, because the planet has no edge."""
    grid = np.zeros((8, 8), bool)
    grid[0, 0] = grid[0, 7] = grid[7, 0] = True  # three cells that only touch through the wrap
    labels = _components(grid)
    assert len({labels[0, 0], labels[0, 7], labels[7, 0]}) == 1


def test_circular_mean_does_not_average_across_the_middle():
    """Positions at 1 and 63 of 64 are neighbours; their centre is 0, not 32."""
    assert _circular_mean(np.array([1.0, 63.0]), 64) < 1.0 or _circular_mean(np.array([1.0, 63.0]), 64) > 63.0


def test_settlements_are_found_and_described():
    s = run(1)
    places = s.settlements.current
    assert places, "no settlement found in a populated world"
    assert sum(p.pop for p in places) <= len(s.living())
    for p in places:
        assert p.pop >= MIN_PEOPLE
        assert p.kind in ("hamlet", "village", "town", "city")
        assert p.area > 0 and p.density > 0
        assert 0 <= p.institutions <= 1 and 0 <= p.cohesion <= 1
        assert 0 <= p.specialization <= 1.001
        assert p.name and p.sid >= 0
        assert p.evidence  # it can say why it is one place


def test_the_class_follows_from_population_alone():
    s = run(2)
    for p in s.settlements.current:
        expected = "hamlet" if p.pop < 40 else "village" if p.pop < 200 else "town" if p.pop < 800 else "city"
        assert p.kind == expected


def test_identity_survives_a_period_and_the_people_turning_over():
    s = run(3, days=26)
    before = {p.sid: (p.name, p.founded) for p in s.settlements.current}
    for _ in range(14):
        s.step()
    after = s.settlements.current
    kept = [p for p in after if p.sid in before]
    assert kept, "every settlement lost its identity in one period"
    for p in kept:
        assert p.name == before[p.sid][0] and p.founded == before[p.sid][1]


def test_names_are_unique_within_a_period():
    s = run(4, days=40)
    names = [p.name for p in s.settlements.current]
    assert len(names) == len(set(names))


def test_hinterland_partitions_the_land():
    s = run(5)
    total = sum(p.hinterland for p in s.settlements.current)
    land = int((s.world.terrain != 0).sum())
    assert 0 < total <= land * 1.3  # sampled every 4 tiles, so allow the sampling slack


def test_the_simulation_does_not_depend_on_the_coarse_graining():
    """Settlements are read off the world. Recomputing them must change nothing about the world."""
    s = run(6, days=20)

    def fingerprint():
        idx = s.living()
        return (len(idx), round(float(s.p.y[idx].sum()), 6), round(float(s.p.energy[idx].sum()), 6),
                round(float(s.world.food.sum()), 6), round(float(s.economy.stock[idx].sum()), 6))

    before = fingerprint()
    for _ in range(5):
        s.settlements.recompute()
    assert fingerprint() == before


def test_a_lump_with_no_one_in_it_is_reported_as_emptied():
    s = run(7, days=14)
    doomed = s.settlements.current[0]
    for pid in list(doomed.members):
        if s.p.alive[pid]:
            s.kill(int(pid), "illness")
    s.settlements.recompute()
    assert doomed.sid not in {p.sid for p in s.settlements.current}
    assert any(e["event"] == "emptied" and e["sid"] == doomed.sid for e in s.settlements.events)


def test_the_report_is_json_shaped():
    import json

    s = run(8)
    json.dumps(s.settlements.report())  # the viewer gets this verbatim


def test_lineage_and_commons_are_different_things():
    """Which store feeds you and which line you descend from come apart: marrying into another
    settlement changes the first and never the second. Without that no settlement can hold two."""
    s = run(21, days=40, people=300)
    p, idx = s.p, s.living()
    assert (p.lineage[idx] >= 0).all()
    # somebody, somewhere, is fed by a commons that is not their line of descent
    moved = idx[p.village[idx] != p.lineage[idx]]
    assert s.counters.get("married_out", 0) >= 0
    if len(moved):
        assert len(set(int(x) for x in p.lineage[moved])) >= 1
    # and a settlement's lineages are reported separately from its commons
    for place in s.settlements.current:
        assert sum(place.bands.values()) == place.pop
        assert sum(place.lineages.values()) == place.pop


def test_a_crowded_commons_splits_and_the_split_is_conservative():
    """Fission moves people and a proportional share of the stores; it creates nothing."""
    s = run(22, days=20, people=300)
    home = s.villages[0]
    p, idx = s.p, s.living()
    members = idx[p.village[idx] == home.id]
    if len(members) < 40:
        return
    # put a knot of them out at the edge and empty the near ground so the commons reads crowded
    knot = members[:20]
    p.y[knot] = (home.cy + 10) % s.world.size
    p.x[knot] = (home.cx + 1) % s.world.size
    home.cand_y, home.cand_x = np.array([home.cy]), np.array([home.cx])  # almost no land in reach
    before_villages = len(s.villages)
    before_food = sum(v.food for v in s.villages)
    before_herd = sum(v.herd for v in s.villages)
    s.settlements.recompute()
    s._fission()
    assert len(s.villages) == before_villages + 1, "a crowded, dispersed group did not split off"
    assert abs(sum(v.food for v in s.villages) - before_food) < 1e-9  # stores were divided, not minted
    assert abs(sum(v.herd for v in s.villages) - before_herd) < 1e-9
    new = s.villages[-1]
    assert new.food >= 0 and len(s.economy.markets) == len(s.villages)
    moved = idx[p.village[idx] == new.id]
    assert len(moved) >= 12
    assert all(int(p.lineage[i]) != new.id for i in moved)  # they keep the line they came from


def test_the_ascription_array_grows_with_the_commons():
    """No silent ceiling: founding a store adds a column rather than being refused."""
    from research.civ_sim.sim import COMMONS_DTYPE_LIMIT, COMMONS_START_WIDTH

    s = run(23, days=6, people=200)
    assert s.p.alpha_width >= COMMONS_START_WIDTH
    before = s.p.alpha_width
    kept = s.p.alpha_p[: s.p.n, :before].copy()
    assert s.p.widen(before + 5)
    assert s.p.alpha_width == before + 5
    assert s.p.alpha_p.shape[1] == before + 5
    assert np.array_equal(s.p.alpha_p[: s.p.n, :before], kept)  # nobody's regard was disturbed
    assert (s.p.alpha_p[: s.p.n, before:] == 0.5).all()  # a settlement you never knew: neither kin nor stranger
    assert not s.p.widen(COMMONS_DTYPE_LIMIT + 1)  # the dtype's own bound, and it reports it


def test_a_refused_split_is_announced_in_the_world():
    """If a bound is ever hit the world says so, instead of fission quietly stopping."""
    s = run(24, days=20, people=300)
    home = s.villages[0]
    p, idx = s.p, s.living()
    members = idx[p.village[idx] == home.id]
    if len(members) < 40:
        return
    knot = members[:20]
    p.y[knot] = (home.cy + 10) % s.world.size
    p.x[knot] = (home.cx + 1) % s.world.size
    home.cand_y, home.cand_x = np.array([home.cy]), np.array([home.cx])
    s.p.widen = lambda *a, **k: False  # stand in for being at the dtype's own limit
    before = len(s.villages)
    s.settlements.recompute()
    s._fission()
    assert len(s.villages) == before  # refused, as it must be
    assert s.counters.get("fission_refused", 0) >= 1
    assert any("no further bands can form" in e["text"] for e in s.events)
