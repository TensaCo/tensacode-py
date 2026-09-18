"""A toroidal planet: a tile grid that wraps on both axes, with terrain, resources, ground state and buildings.

No edges: every distance, neighbour lookup and step uses wrapped (toroidal) metrics, and the
noise that makes the terrain is periodic, so continents run across the seams. Resources are
float64 so the conservation ledger holds to rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WATER, GRASS, FOREST, STONE, MOUNTAIN = 0, 1, 2, 3, 4
TERRAIN_NAMES = ("water", "grass", "forest", "stone", "mountain")
DAYS_PER_SEASON = 12
DAYS_PER_YEAR = 4 * DAYS_PER_SEASON
SEASONS = ("spring", "summer", "autumn", "winter")
REGROWTH = np.array([1.15, 1.35, 0.85, 0.3])

# buildings (one per tile)
NONE, HOUSE, GRANARY, WELL, SHRINE, WORKSHOP, PALISADE, FIELD, STOCKPILE, MONUMENT, HEARTH, WATCHTOWER = range(12)
BUILDING_NAMES = ("", "house", "granary", "well", "shrine", "workshop", "palisade", "field", "stockpile", "monument", "hearth", "watchtower")
BUILD_COST = {HOUSE: (5.0, 4.0), GRANARY: (10.0, 14.0), WELL: (4.0, 4.0), SHRINE: (6.0, 10.0), WORKSHOP: (8.0, 10.0),
              PALISADE: (2.0, 8.0), FIELD: (1.0, 0.0), STOCKPILE: (2.0, 3.0), MONUMENT: (10.0, 0.0), HEARTH: (2.0, 3.0), WATCHTOWER: (4.0, 8.0)}


def periodic_noise(rng: np.random.Generator, size: int, octaves: int = 5) -> np.ndarray:
    """Fractal value noise that tiles seamlessly: the lattice wraps, so opposite edges match."""
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        grid = rng.random((n, n))
        g = np.pad(grid, ((0, 1), (0, 1)), mode="wrap")
        t = np.linspace(0, n, size, endpoint=False)
        i0 = t.astype(int)
        f = t - i0
        f = f * f * (3 - 2 * f)
        a, b = g[np.ix_(i0, i0)], g[np.ix_(i0, i0 + 1)]
        c, d = g[np.ix_(i0 + 1, i0)], g[np.ix_(i0 + 1, i0 + 1)]
        fy, fx = f[:, None], f[None, :]
        out += amp * (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def wrap_delta(a, b, size: int):
    """Shortest signed difference a - b on a circle of circumference ``size``."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return d - size * np.round(d / size)


def torus_distance(y1, x1, y2, x2, size: int):
    return np.hypot(wrap_delta(y1, y2, size), wrap_delta(x1, x2, size))


@dataclass
class World:
    size: int
    elevation: np.ndarray
    terrain: np.ndarray
    food_cap: np.ndarray
    wood_cap: np.ndarray
    food: np.ndarray
    wood: np.ndarray
    stone: np.ndarray
    fertility: np.ndarray  # current soil quality: cropping draws it down, fallow restores it
    fertility_base: np.ndarray  # what the soil recovers toward
    building: np.ndarray  # uint8 building id per tile
    build_owner: np.ndarray  # int8 village id, -1 for none
    build_progress: np.ndarray  # float32, 1.0 = finished
    trample: np.ndarray  # path wear where people walk
    tilled: np.ndarray  # worked soil (fields): raises capacity, decays
    burn: np.ndarray  # burn scars from raids
    ledger: dict = field(default_factory=lambda: {"regrowth_food": 0.0, "regrowth_wood": 0.0, "build_food": 0.0, "build_wood": 0.0, "build_stone": 0.0})

    @classmethod
    def generate(cls, rng: np.random.Generator, size: int = 160) -> "World":
        elev = periodic_noise(rng, size, 5)
        ridges = 1 - np.abs(periodic_noise(rng, size, 3) - 0.5) * 2
        elev = 0.7 * elev + 0.3 * ridges
        elev = (elev - elev.min()) / (elev.max() - elev.min())
        moist = periodic_noise(rng, size, 4)
        woods = periodic_noise(rng, size, 4)
        terrain = np.full((size, size), GRASS, np.uint8)
        terrain[woods > 0.58] = FOREST
        terrain[elev > 0.74] = STONE
        terrain[elev > 0.88] = MOUNTAIN
        terrain[elev < 0.32] = WATER
        water = terrain == WATER
        near = np.zeros_like(water)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                near |= np.roll(np.roll(water, dy, 0), dx, 1)  # roll wraps: right on a torus
        fert = np.clip(0.4 + 0.5 * moist + 0.3 * near - 0.4 * np.clip(elev - 0.5, 0, 1), 0, 1)
        fert = np.where(terrain == GRASS, fert, fert * 0.3)
        food_cap = np.where(terrain == GRASS, 45.0 * fert, np.where(terrain == FOREST, 12.0 * fert, 0.0))
        wood_cap = np.where(terrain == FOREST, 12.0, 0.0)
        stone = np.where(terrain == STONE, 45.0, np.where(terrain == MOUNTAIN, 20.0, 0.0))
        z = lambda: np.zeros((size, size), np.float32)  # noqa: E731
        return cls(size, elev, terrain, food_cap, wood_cap, food_cap * 0.85, wood_cap * 0.9, stone.astype(np.float64), fert, fert.copy(),
                   np.zeros((size, size), np.uint8), np.full((size, size), -1, np.int8), z(), z(), z(), z())

    # ------------------------------------------------------------ dynamics

    def effective_food_cap(self) -> np.ndarray:
        field = (self.building == FIELD) & (self.build_progress >= 1.0)
        soil = np.where(self.fertility_base > 0, self.fertility / np.maximum(self.fertility_base, 1e-6), 1.0)
        return self.food_cap * np.clip(soil, 0.15, 1.2) * (1 + 1.2 * self.tilled) * np.where(field, 2.2, 1.0) * (1 - 0.5 * self.burn)

    def regrow(self, season: int, light: np.ndarray, rain: np.ndarray, temperature: np.ndarray) -> None:
        """Logistic regrowth scaled by season, daylight, rain and frost; ground state decays."""
        m = REGROWTH[season]
        weather = np.clip(0.45 + 0.9 * light, 0, 1.6) * np.clip(0.5 + 1.2 * rain, 0.5, 1.5) * np.clip((temperature + 4) / 18, 0.05, 1.3)
        cap_f = np.maximum(self.effective_food_cap(), 1e-9)
        add_f = np.where(self.food_cap > 0, m * weather * (0.3 * self.food * (1 - self.food / cap_f) + 0.022 * cap_f), 0.0)
        # a planted, worked field grows a crop each day rather than regrowing like wild forage
        fields = (self.building == FIELD) & (self.build_progress >= 1.0)
        add_f = np.where(fields, m * weather * 0.32 * cap_f, add_f)
        add_f = np.clip(add_f, 0, np.maximum(cap_f - self.food, 0))
        cap_w = np.maximum(self.wood_cap, 1e-9)
        add_w = np.where(self.wood_cap > 0, m * (0.05 * self.wood * (1 - self.wood / cap_w) + 0.006 * self.wood_cap), 0.0)
        add_w = np.clip(add_w, 0, self.wood_cap - self.wood)
        self.food += add_f
        self.wood += add_w
        self.ledger["regrowth_food"] += float(add_f.sum())
        self.ledger["regrowth_wood"] += float(add_w.sum())
        # Soil: a crop takes nutrients out of the ground, and ground left alone gets them back. This is
        # what puts a ceiling on the land — a village that crops everything hard exhausts it. A cropped
        # field loses about 0.2% of its base fertility a day and every tile recovers 0.4% of what it is
        # missing, so permanent cropland settles near half its natural quality: five years to get
        # there, and as long again to come back if it is left fallow.
        drain = np.where(fields, 0.0020 * np.clip(add_f / np.maximum(cap_f, 1e-9) / 0.32, 0, 1.5), 0.0) * self.fertility_base
        recover = 0.004 * (self.fertility_base - self.fertility)
        self.fertility = np.clip(self.fertility - drain + recover, 0.0, self.fertility_base)
        self.trample *= 0.994
        self.tilled *= 0.996
        self.burn *= 0.9985

    def walkable(self, y, x) -> np.ndarray:
        t = self.terrain[np.asarray(y).astype(int) % self.size, np.asarray(x).astype(int) % self.size]
        return (t != WATER) & (t != MOUNTAIN)

    def distance(self, y1, x1, y2, x2):
        return torus_distance(y1, x1, y2, x2, self.size)

    def step_toward(self, y, x, ty, tx, speed):
        """One wrapped step toward a target; returns new (y, x) and remaining distance."""
        dy, dx = wrap_delta(ty, y, self.size), wrap_delta(tx, x, self.size)
        d = np.hypot(dy, dx)
        s = np.minimum(1.0, speed / np.maximum(d, 1e-6))
        return (y + dy * s) % self.size, (x + dx * s) % self.size, d
