"""Settlements are not objects in the simulation. They are read off it.

Nothing in the tick loop knows what a town is. People walk, eat, bond, trade, and hand food to a
common store; that is all the mechanics there is. Every season this module looks at the result and
asks where the settlements are, by coarse-graining three topologies at once:

    spatial   where people actually are: a density field on the torus, thresholded, then connected
              components over wrapped cells. This finds lumps of people, not villages.
    social    who is tied to whom: bonds and parent/child links crossing between lumps. Two lumps
              standing next to each other with no ties between them stay two settlements; two lumps
              densely intermarried become one. That is the merge rule, and it is the social topology
              doing the work.
    economic  what the lump does: the mix of labour, what it holds, what it is worth, how unequal it
              is, and how much of the surrounding land it works.

Out of that come properties nobody assigned: population, area, density, the specialization mix,
median worth and its Gini, institutional capacity, cohesion, hinterland, and a size class
(hamlet / village / town / city) that a settlement crosses on its own.

Settlements are then matched to last season's by membership overlap, so one keeps its name and
identity while it grows, splits, merges, moves or dies — and those four events are logged with the
evidence that produced them. The identity is a correspondence over time, not a thing in the world.

The one thing that is *not* derived is the **common store**: a founding band keeps a shared granary
(`sim.villages`), and that is primitive, stated plainly here and in the docs. A settlement may hold
several bands, and one band's people may live in two settlements — which is exactly what makes the
coarse-graining non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL = 4  # tiles per coarse cell
MIN_PEOPLE = 6  # fewer than this and it is not a settlement, it is some people standing in a field
MERGE_RADIUS = 11.0  # only neighbouring lumps can be one settlement, however intermarried
CLASSES = ((40, "hamlet"), (200, "village"), (800, "town"), (10 ** 9, "city"))
SYLL_A = ("Ald", "Bren", "Cor", "Dun", "Esk", "Far", "Gell", "Hal", "Ith", "Kel", "Mor", "Nor", "Oss", "Pel", "Rhun", "Sel", "Tor", "Vann")
SYLL_B = ("mere", "holt", "alin", "marsh", "ridge", "fell", "wick", "gard", "combe", "stead", "ford", "haven", "keep", "reach")


@dataclass
class Settlement:
    sid: int
    name: str
    members: frozenset
    cy: float
    cx: float
    cells: tuple  # the coarse cells it occupies, for drawing the coarse-graining itself
    pop: int
    area: float  # tiles
    density: float
    radius: float
    bands: dict  # commons id -> how many people here are fed from it
    lineages: dict  # founding line -> how many people here descend from it
    labour: dict  # task name -> share
    specialization: float  # 0 = everyone does the same thing, 1 = fully specialized
    worth: float
    gini: float
    holdings: dict
    price: dict
    institutions: float  # 0..1: leader, law, granary, well, shrine, workshop, walls
    cohesion: float  # 0..1: ideological agreement minus grudge
    factions: int
    hinterland: int  # tiles closer to here than to any other settlement
    buildings: dict
    kind: str
    founded: int
    history: list = field(default_factory=list)  # (day, event, detail)
    ties: dict = field(default_factory=dict)  # other sid -> social tie strength
    evidence: dict = field(default_factory=dict)  # why this is one settlement and not two
    between: float = 0.0  # spread of median worth between the commons living here

    def label(self) -> str:
        return f"{self.name} ({self.kind}, {self.pop})"


def _components(occupied: np.ndarray) -> np.ndarray:
    """Connected components of occupied coarse cells, wrapping on both axes (union-find)."""
    n = occupied.shape[0]
    parent = np.arange(n * n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ys, xs = np.nonzero(occupied)
    for y, x in zip(ys, xs):
        here = y * n + x
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ny, nx = (y + dy) % n, (x + dx) % n
            if occupied[ny, nx]:
                union(here, ny * n + nx)
    labels = np.full((n, n), -1, np.int32)
    for y, x in zip(ys, xs):
        labels[y, x] = find(y * n + x)
    return labels


def _circular_mean(values: np.ndarray, size: int) -> float:
    """Mean position on a circle: you cannot average coordinates that wrap."""
    ang = values / size * 2 * np.pi
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((m / (2 * np.pi) * size) % size)


class Settlements:
    """Holds the last coarse-graining and the correspondence between periods."""

    def __init__(self, sim) -> None:
        self.sim = sim
        self.current: list[Settlement] = []
        self.next_sid = 0
        self.events: list[dict] = []
        self.day = -1

    # ------------------------------------------------------------ the coarse-graining

    def recompute(self) -> list[Settlement]:
        sim, p = self.sim, self.sim.p
        idx = sim.living()
        size = sim.world.size
        n = max(4, size // CELL)
        if len(idx) == 0:
            self.current = []
            return self.current
        cy = (p.y[idx].astype(int) // CELL) % n
        cx = (p.x[idx].astype(int) // CELL) % n
        counts = np.zeros((n, n), np.int32)
        np.add.at(counts, (cy, cx), 1)
        # a cell is part of a settlement if it is denser than the planet's mean occupied cell
        occupied_mean = counts[counts > 0].mean() if (counts > 0).any() else 0
        occupied = counts >= max(2, occupied_mean * 0.6)
        labels = _components(occupied)
        lab = labels[cy, cx]
        groups: dict[int, np.ndarray] = {}
        for label in np.unique(lab[lab >= 0]):
            members = idx[lab == label]
            if len(members) >= MIN_PEOPLE:
                groups[int(label)] = members
        if not groups:
            self.current = []
            return self.current
        groups = self._merge_by_social_ties(groups, size)
        built = [self._describe(label, members, labels, n) for label, members in groups.items()]
        built = self._match_to_previous(built)
        self.current = sorted(built, key=lambda s: -s.pop)
        self._hinterland()
        self.day = sim.day
        return self.current

    def _merge_by_social_ties(self, groups: dict, size: int) -> dict:
        """Lumps that are neighbours *and* densely tied become one settlement. The social topology
        decides; proximity only gives it the chance to."""
        p = self.sim.p
        keys = list(groups)
        where = {}
        for k in keys:
            where.update({int(i): k for i in groups[k]})
        # ties: bonds and parent/child links that cross from one lump to another
        cross: dict[tuple, int] = {}
        internal = {k: 0 for k in keys}
        for arr in (p.bond, p.mother, p.father):
            for i, other in ((int(i), int(arr[int(i)])) for k in keys for i in groups[k]):
                if other < 0:
                    continue
                a, b = where.get(i), where.get(other)
                if a is None or b is None:
                    continue
                if a == b:
                    internal[a] += 1
                else:
                    cross[(min(a, b), max(a, b))] = cross.get((min(a, b), max(a, b)), 0) + 1
        centre = {k: (_circular_mean(p.y[groups[k]], size), _circular_mean(p.x[groups[k]], size)) for k in keys}
        merged = {k: k for k in keys}

        def root(k):
            while merged[k] != k:
                k = merged[k]
            return k

        self._tie_evidence = {}
        for (a, b), ties in sorted(cross.items(), key=lambda kv: -kv[1]):
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            dist = float(self.sim.world.distance(centre[a][0], centre[a][1], centre[b][0], centre[b][1]))
            if dist > MERGE_RADIUS:
                continue
            small = min(len(groups[a]), len(groups[b]))
            # tied to each other about as much as to themselves: one place, not two
            if ties < max(3, 0.12 * small):
                continue
            merged[max(ra, rb)] = min(ra, rb)
            self._tie_evidence[min(ra, rb)] = {"merged_lumps": self._tie_evidence.get(min(ra, rb), {}).get("merged_lumps", 1) + 1,
                                               "crossing_ties": ties, "gap_tiles": round(dist, 1)}
        out: dict[int, list] = {}
        for k in keys:
            out.setdefault(root(k), []).append(groups[k])
        return {k: np.concatenate(v) for k, v in out.items()}

    def _describe(self, label: int, members: np.ndarray, labels: np.ndarray, n: int) -> Settlement:
        from .economy import GOODS, STONE, TOOLS, WOOD
        from .sim import TASKS
        from .world import BUILDING_NAMES, GRANARY, PALISADE, SHRINE, WELL, WORKSHOP

        sim, p, e = self.sim, self.sim.p, self.sim.economy
        size = sim.world.size
        cy = _circular_mean(p.y[members], size)
        cx = _circular_mean(p.x[members], size)
        dy = np.abs((p.y[members] - cy + size / 2) % size - size / 2)
        dx = np.abs((p.x[members] - cx + size / 2) % size - size / 2)
        radius = float(np.sqrt(dy ** 2 + dx ** 2).mean()) + 0.5
        cells = tuple((int(y), int(x)) for y, x in zip(*np.nonzero(labels == label))) if label in np.unique(labels) else ()
        area = max(1.0, float(len(cells) * CELL * CELL)) if cells else max(1.0, np.pi * radius ** 2)
        bands = {int(b): int(c) for b, c in zip(*np.unique(p.village[members], return_counts=True))}
        lineages = {int(b): int(c) for b, c in zip(*np.unique(p.lineage[members], return_counts=True))}
        # inequality *between* the commons inside one settlement, which only means anything once a
        # settlement can hold more than one
        between = 0.0
        if len(bands) > 1 and e is not None:
            per = [float(np.median(e.net_worth(members[p.village[members] == b]))) for b in bands]
            between = round(float(np.std(per) / max(1e-9, np.mean(per))), 3)
        work = np.bincount(p.task[members], minlength=12).astype(float)
        working = work.copy()
        working[[2, 11]] = 0  # resting and sleeping are not an occupation
        share = working / working.sum() if working.sum() > 0 else working
        labour = {TASKS[i]: round(float(s), 3) for i, s in enumerate(share) if s > 0.01}
        # specialization: how far the labour mix is from everyone doing the same thing
        nz = share[share > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(len(nz))) if len(nz) > 1 else 0.0
        worth = e.net_worth(members) if e is not None else np.asarray(p.wealth[members], dtype=float)
        main_band = max(bands, key=bands.get)
        v = sim.villages[main_band]
        insts = sum((v.leader >= 0 and p.alive[v.leader], bool(v.law_share), v.buildings.get(GRANARY, 0) > 0,
                     v.buildings.get(WELL, 0) > 0, v.buildings.get(SHRINE, 0) > 0, v.buildings.get(WORKSHOP, 0) > 0,
                     v.buildings.get(PALISADE, 0) >= 4)) / 7
        spread = float(np.std(p.share[members])) + float(np.std(p.martial[members]))
        grudge = 0.0
        if getattr(sim, "minds", None) is not None:
            rels = [r["grudge"] for pid in members.tolist() if (m := sim.minds.minds.get(int(pid)))
                    for r in m.relations.values()]
            grudge = float(np.mean(rels)) if rels else 0.0
        cohesion = float(np.clip(1 - spread - 0.5 * grudge, 0, 1))
        pop = len(members)
        kind = next(name for limit, name in CLASSES if pop < limit)
        holdings = {GOODS[g]: round(float(e.stock[members, g].sum()), 1) for g in (WOOD, STONE, TOOLS)} if e is not None else {}
        price = {GOODS[g]: e.price(main_band, g) for g in (WOOD, STONE, TOOLS)} if e is not None else {}
        return Settlement(
            sid=-1, name="", members=frozenset(int(i) for i in members), cy=cy, cx=cx, cells=cells, pop=pop,
            area=area, density=round(pop / area, 3), radius=round(radius, 2), bands=bands, lineages=lineages, labour=labour,
            specialization=round(entropy, 3), worth=round(float(np.median(worth)), 2),
            gini=round(float(e.gini(worth)) if e is not None else 0.0, 3), holdings=holdings, price=price,
            institutions=round(insts, 3), cohesion=round(cohesion, 3), between=between,
            factions=int(1 + (spread > 0.45) + (grudge > 0.35) + (len(bands) > 1)), hinterland=0,
            buildings={BUILDING_NAMES[b]: c for b, c in sorted(v.buildings.items()) if b and c}, kind=kind,
            founded=sim.day,
            evidence=self._tie_evidence.get(label, {}) | {"lumps": 1 + self._tie_evidence.get(label, {}).get("merged_lumps", 1) - 1,
                                                          "cells": len(cells), "threshold": "denser than 0.6x the mean occupied cell"},
        )

    # ------------------------------------------------------------ identity over time

    def _match_to_previous(self, built: list) -> list:
        """Who is the same settlement as last season? Membership overlap decides, so a place keeps
        its name while its people turn over — and splits, merges and deaths are named events."""
        sim = self.sim
        old = self.current
        self._named_this_pass = []
        claimed: dict[int, list] = {}
        for s in built:
            best, best_score = None, 0.0
            for o in old:
                if not o.members:
                    continue
                overlap = len(s.members & o.members) / len(s.members | o.members)
                if overlap > best_score:
                    best, best_score = o, overlap
            if best is not None and best_score >= 0.25:
                claimed.setdefault(best.sid, []).append((s, best, best_score))
            else:
                claimed.setdefault(-1, []).append((s, None, 0.0))
        out = []
        for sid, group in claimed.items():
            if sid == -1:
                for s, _, _ in group:
                    self._adopt_new(s)
                    out.append(s)
                continue
            group.sort(key=lambda t: -t[2])
            for rank, (s, o, score) in enumerate(group):
                if rank == 0:
                    s.sid, s.name, s.founded, s.history = o.sid, o.name, o.founded, list(o.history)
                    if s.kind != o.kind:
                        self._event(s, "grew" if s.pop > o.pop else "shrank", f"{o.kind} -> {s.kind} at {s.pop} people")
                    moved = float(sim.world.distance(s.cy, s.cx, o.cy, o.cx))
                    if moved > 6:
                        self._event(s, "moved", f"centre shifted {moved:.0f} tiles")
                    if len(group) > 1:
                        self._event(s, "split", f"{len(group)} settlements now where {o.name} was (overlap {score:.0%})")
                    if s.pop > o.pop * 1.6 and rank == 0 and len(group) == 1:
                        self._event(s, "absorbed", f"population {o.pop} -> {s.pop}")
                else:
                    self._adopt_new(s, parent=o)
                out.append(s)
        for s in out:
            for o in old:
                if o.sid != s.sid and o.members and len(s.members & o.members) / max(1, len(o.members)) > 0.5:
                    self._event(s, "merged", f"{o.name} ({o.pop}) is now part of {s.name}")
        live = {s.sid for s in out}
        for o in old:
            if o.sid not in live and o.pop >= MIN_PEOPLE:
                self.events.append({"day": sim.day, "sid": o.sid, "name": o.name, "event": "emptied",
                                    "detail": f"{o.name} has no one left"})
                sim.log("settlement", f"{o.name} was abandoned")
        del self.events[:-300]
        return out

    def _adopt_new(self, s: Settlement, parent: Settlement | None = None) -> None:
        sim = self.sim
        s.sid = self.next_sid
        self.next_sid += 1
        rng = sim.rng
        taken = {o.name for o in self.current} | {o.name for o in getattr(self, "_named_this_pass", [])}
        for attempt in range(40):
            if parent is not None:  # a place that split off is named after its parent, then drifts
                name = parent.name[: max(3, len(parent.name) // 2)] + SYLL_B[rng.integers(len(SYLL_B))]
            else:
                name = SYLL_A[rng.integers(len(SYLL_A))] + SYLL_B[rng.integers(len(SYLL_B))]
            if name not in taken:
                break
            if attempt > 20:
                name = f"{name} {self.next_sid}"  # two places really did end up wanting one name
                break
        s.name = name
        self._named_this_pass = list(getattr(self, "_named_this_pass", [])) + [s]
        if parent is not None:
            s.history = [(sim.day, "split from", parent.name)]
            self._event(s, "founded", f"split from {parent.name}")
        else:
            self._event(s, "founded", f"{s.pop} people, {s.kind}")

    def _event(self, s: Settlement, kind: str, detail: str) -> None:
        s.history.append((self.sim.day, kind, detail))
        del s.history[:-20]
        self.events.append({"day": self.sim.day, "sid": s.sid, "name": s.name, "event": kind, "detail": detail})
        if kind in ("founded", "split", "merged", "grew", "emptied"):
            self.sim.log("settlement", f"{s.name}: {kind} — {detail}")

    def _hinterland(self) -> None:
        """How much land is worked from here: tiles nearer this settlement than any other."""
        if not self.current:
            return
        size = self.sim.world.size
        step = 4
        ys = np.arange(0, size, step)
        gy, gx = np.meshgrid(ys, ys, indexing="ij")
        best = np.full(gy.shape, np.inf)
        owner = np.full(gy.shape, -1, np.int32)
        for k, s in enumerate(self.current):
            d = self.sim.world.distance(gy, gx, s.cy, s.cx)
            closer = d < best
            best, owner = np.where(closer, d, best), np.where(closer, k, owner)
        land = self.sim.world.terrain[::step, ::step] != 0
        for k, s in enumerate(self.current):
            s.hinterland = int(((owner == k) & land).sum() * step * step)

    # ------------------------------------------------------------ output

    def report(self) -> list:
        return [{
            "sid": s.sid, "name": s.name, "kind": s.kind, "pop": s.pop, "y": round(s.cy, 1), "x": round(s.cx, 1),
            "radius": s.radius, "area": round(s.area, 1), "density": s.density, "hinterland": s.hinterland,
            "bands": s.bands, "lineages": s.lineages, "between_band_inequality": s.between,
            "labour": s.labour, "specialization": s.specialization, "worth": s.worth,
            "gini": s.gini, "holdings": s.holdings, "price": s.price, "institutions": s.institutions,
            "cohesion": s.cohesion, "factions": s.factions, "buildings": s.buildings, "founded": s.founded,
            "cells": [[y, x] for y, x in s.cells[:400]], "evidence": s.evidence,
            "history": [{"day": d, "event": k, "detail": t} for d, k, t in s.history[-6:]],
        } for s in self.current]

    def of(self, pid: int) -> Settlement | None:
        for s in self.current:
            if int(pid) in s.members:
                return s
        return None
