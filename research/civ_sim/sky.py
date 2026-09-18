"""Sky over a toroidal planet: rotation (so local time runs with longitude), two moons, four planets.

Everything is an analytic function of the world clock, so it is deterministic, periodic and cheap.
Positions are angles in turns [0, 1). Light at a tile column is a function of local solar angle
plus moonlight; eclipses and conjunctions are found by angular proximity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .world import DAYS_PER_YEAR

HOURS_PER_DAY = 24.0
MOONS = ({"name": "Bel", "period": 9.4, "size": 1.0, "brightness": 0.8}, {"name": "Ara", "period": 27.3, "size": 0.55, "brightness": 0.35})
PLANETS = ({"name": "Tal", "period": 0.62 * DAYS_PER_YEAR}, {"name": "Wen", "period": 1.9 * DAYS_PER_YEAR},
           {"name": "Sarn", "period": 4.3 * DAYS_PER_YEAR}, {"name": "Hume", "period": 11.8 * DAYS_PER_YEAR})


@dataclass(frozen=True)
class SkyState:
    day: int
    hour: float
    season: int
    sun_longitude: float  # turn of the planet: which longitude has noon
    moon_phase: tuple  # per moon, 0 = new, 0.5 = full
    moon_longitude: tuple
    planet_longitude: tuple
    eclipse: str | None  # "solar" | "lunar" | None
    conjunction: tuple  # names of planets within 2 degrees

    @property
    def night(self) -> bool:
        return not 6 <= self.hour < 18

    def moonlight(self) -> float:
        """0..1 from the phases of both moons (full moon nights are much brighter)."""
        return float(sum(m["brightness"] * max(0.0, 1 - abs(p - 0.5) * 2) for m, p in zip(MOONS, self.moon_phase)))

    def describe(self) -> str:
        bits = [f"{int(self.hour):02d}:{int(self.hour % 1 * 60):02d}"]
        for m, p in zip(MOONS, self.moon_phase):
            names = ("new", "waxing", "full", "waning")
            bits.append(f"{m['name']} {names[int((p % 1) * 4)]}")
        if self.eclipse:
            bits.append(f"{self.eclipse} eclipse")
        if self.conjunction:
            bits.append("conjunction of " + " and ".join(self.conjunction))
        return " · ".join(bits)


def state(day: int, hour: float) -> SkyState:
    t = day + hour / HOURS_PER_DAY
    sun_lon = (hour / HOURS_PER_DAY) % 1.0
    moon_lon = tuple(((t / m["period"]) % 1.0) for m in MOONS)
    phase = tuple(((t / m["period"] + 0.5) % 1.0) for m in MOONS)
    planet_lon = tuple(((t / p["period"]) % 1.0) for p in PLANETS)
    ecl = None
    if abs(((phase[0] - 0.0 + 0.5) % 1.0) - 0.5) < 0.012 and abs(((phase[1] + 0.5) % 1.0) - 0.5) < 0.05:
        ecl = "solar"  # both moons near new and aligned with the sun
    elif abs(phase[0] - 0.5) < 0.008 and abs(phase[1] - 0.5) < 0.06:
        ecl = "lunar"
    conj = []
    for i in range(len(PLANETS)):
        for j in range(i + 1, len(PLANETS)):
            if abs(((planet_lon[i] - planet_lon[j] + 0.5) % 1.0) - 0.5) < 0.002:
                conj += [PLANETS[i]["name"], PLANETS[j]["name"]]
    return SkyState(day, hour, (day // 12) % 4, sun_lon, phase, moon_lon, planet_lon, ecl, tuple(dict.fromkeys(conj)))


def light_field(sky: SkyState, size: int) -> np.ndarray:
    """Light per tile column: the sun sweeps longitude, so dawn runs around the world."""
    lon = (np.arange(size) / size)
    ang = (lon - sky.sun_longitude + 0.5) % 1.0 - 0.5  # 0 = local noon
    daylight = np.clip(np.cos(ang * 2 * math.pi) * 1.6 + 0.35, 0, 1)
    return (daylight + (1 - np.clip(daylight * 3, 0, 1)) * 0.22 * sky.moonlight()).astype(np.float32)


def daily_light(day: int, size: int) -> np.ndarray:
    """Mean light over a day for each column (used by regrowth, which ticks once a day)."""
    acc = np.zeros(size, np.float32)
    for h in range(0, 24, 3):
        acc += light_field(state(day, h), size)
    return acc / 8


def season_temperature(day: int, size: int) -> np.ndarray:
    """Base temperature in °C by latitude (torus rows) and season."""
    lat = np.cos(np.arange(size) / size * 2 * math.pi)  # two temperate bands, two cold bands
    season_shift = math.cos((day / DAYS_PER_YEAR) * 2 * math.pi) * 7
    return (9 + 11 * lat - season_shift).astype(np.float32)
