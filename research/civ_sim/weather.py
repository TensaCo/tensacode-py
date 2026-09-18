"""Weather as a drifting field on the torus, not per-tile noise.

A coarse grid carries pressure, moisture and temperature anomaly. Wind comes from the pressure
gradient (with a prevailing easterly), moisture is advected by the wind, and rain falls where
moist air meets high ground or converging flow. Everything wraps, so systems cross the seams.
Upsampled to tiles for the simulation and the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sky as skymod
from .world import periodic_noise


def _roll(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(a, dy, 0), dx, 1)


@dataclass
class Weather:
    size: int  # tile grid size
    n: int  # coarse grid size
    pressure: np.ndarray
    moisture: np.ndarray
    anomaly: np.ndarray  # temperature anomaly °C
    rain: np.ndarray  # coarse rain intensity 0..1
    snow: np.ndarray
    cloud: np.ndarray
    elevation_coarse: np.ndarray
    day: int = 0
    events: tuple = ()  # ("drought", "heatwave", "storm", "frost", "flood") active today

    @classmethod
    def create(cls, rng: np.random.Generator, world, n: int | None = None) -> "Weather":
        n = n or max(8, world.size // 4)
        while world.size % n:  # the coarse grid has to divide the tile grid exactly
            n -= 1
        p = periodic_noise(rng, n, 3) - 0.5
        m = periodic_noise(rng, n, 3)
        k = world.size // n
        elev = world.elevation.reshape(n, k, n, k).mean((1, 3))
        z = np.zeros((n, n), np.float32)
        return cls(world.size, n, p.astype(np.float32), m.astype(np.float32), (periodic_noise(rng, n, 2) - 0.5).astype(np.float32) * 4, z.copy(), z.copy(), z.copy(), elev.astype(np.float32))

    def step(self, day: int, rng: np.random.Generator, base_temp: np.ndarray) -> None:
        """One day of weather: advect, orographic lift, rain out, and note extreme events."""
        self.day = day
        n = self.n
        # wind from the pressure gradient, plus a prevailing easterly; wraps on both axes
        dpx = (_roll(self.pressure, 0, -1) - _roll(self.pressure, 0, 1)) * 0.5
        dpy = (_roll(self.pressure, -1, 0) - _roll(self.pressure, 1, 0)) * 0.5
        wx = np.clip(-dpx * 6 + 0.55, -1.6, 1.6)
        wy = np.clip(-dpy * 6, -1.6, 1.6)
        # semi-Lagrangian advection of moisture along the wind (bilinear, wrapped)
        ys, xs = np.mgrid[0:n, 0:n]
        sy, sx = (ys - wy) % n, (xs - wx) % n
        y0, x0 = sy.astype(int), sx.astype(int)
        fy, fx = sy - y0, sx - x0
        y1, x1 = (y0 + 1) % n, (x0 + 1) % n
        adv = (self.moisture[y0, x0] * (1 - fy) * (1 - fx) + self.moisture[y1, x0] * fy * (1 - fx)
               + self.moisture[y0, x1] * (1 - fy) * fx + self.moisture[y1, x1] * fy * fx)
        self.moisture = adv.astype(np.float32)
        # pressure evolves slowly with its own drift and a little noise; smoothing keeps systems coherent
        blur = lambda a: (a + _roll(a, 1, 0) + _roll(a, -1, 0) + _roll(a, 0, 1) + _roll(a, 0, -1)) / 5  # noqa: E731
        self.pressure = (0.985 * blur(self.pressure) + 0.06 * (rng.random((n, n)).astype(np.float32) - 0.5)).astype(np.float32)
        self.pressure -= self.pressure.mean()
        # evaporation over lowland/water, orographic lift over hills, convergence rain in lows
        self.moisture += (0.05 * (0.6 - self.elevation_coarse) + 0.03).astype(np.float32)
        convergence = np.clip(-(_roll(wx, 0, -1) - _roll(wx, 0, 1) + _roll(wy, -1, 0) - _roll(wy, 1, 0)) * 0.5, 0, None)
        lift = np.clip(self.elevation_coarse - 0.45, 0, None) * 1.4
        precip = np.clip((self.moisture - 0.45) * (0.5 + lift + 1.2 * convergence + np.clip(-self.pressure * 4, 0, 1.5)), 0, None)
        precip = np.minimum(precip, self.moisture)
        self.moisture -= precip
        self.moisture = np.clip(self.moisture, 0, 1.6).astype(np.float32)
        self.anomaly = (0.9 * self.anomaly + 0.5 * self.pressure * 6 + 0.25 * (rng.random((n, n)).astype(np.float32) - 0.5)).astype(np.float32)
        temp = base_temp[:, None] * np.ones((1, n), np.float32) if base_temp.ndim == 1 else base_temp
        temp = temp[:: max(1, len(temp) // n)][:n] if temp.shape[0] != n else temp
        coarse_temp = temp + self.anomaly
        self.rain = np.where(coarse_temp > 1.0, precip, 0).astype(np.float32)
        self.snow = np.where(coarse_temp <= 1.0, precip, 0).astype(np.float32)
        self.cloud = np.clip(self.moisture * 1.3 + precip * 2, 0, 1).astype(np.float32)
        ev = []
        if float(self.rain.mean()) < 0.006 and float(self.moisture.mean()) < 0.45:
            ev.append("drought")
        if float(coarse_temp.mean()) > 20:
            ev.append("heatwave")
        if float(coarse_temp.mean()) < 1:
            ev.append("frost")
        if float(precip.max()) > 0.85:
            ev.append("storm")
        if float(self.rain.mean()) > 0.25:
            ev.append("flood")
        self.events = tuple(ev)

    # ------------------------------------------------------------ sampling

    def tile(self, field: np.ndarray) -> np.ndarray:
        """Upsample a coarse field to the tile grid (nearest; cheap and adequate)."""
        k = self.size // self.n
        return np.repeat(np.repeat(field, k, 0), k, 1)

    def at(self, y, x, field: np.ndarray):
        yy = (np.asarray(y).astype(int) % self.size) * self.n // self.size
        xx = (np.asarray(x).astype(int) % self.size) * self.n // self.size
        return field[yy, xx]

    def report(self, y: int, x: int, base_temp: np.ndarray) -> dict:
        r, s = float(self.at(y, x, self.rain)), float(self.at(y, x, self.snow))
        t = float(base_temp[int(y) % self.size]) + float(self.at(y, x, self.anomaly))
        sky_words = "snowing" if s > 0.02 else "raining" if r > 0.02 else "cloudy" if float(self.at(y, x, self.cloud)) > 0.55 else "clear"
        return {"sky": sky_words, "rain": round(r, 3), "snow": round(s, 3), "temperature_c": round(t, 1), "events": list(self.events)}


def weather_words(rain: float, snow: float, temp: float, cloud: float) -> str:
    if snow > 0.02:
        return "snow"
    if rain > 0.25:
        return "downpour"
    if rain > 0.02:
        return "rain"
    if temp > 24:
        return "heat"
    if temp < 1:
        return "frost"
    if cloud > 0.55:
        return "grey skies"
    return "clear skies"
