"""The toroidal planet and its sky: wrapped metrics, seamless terrain, periodic and deterministic sky."""

import numpy as np

from research.civ_sim import sky as skymod
from research.civ_sim.weather import Weather
from research.civ_sim.world import DAYS_PER_YEAR, World, periodic_noise, torus_distance, wrap_delta


def world(seed: int = 3, size: int = 64) -> World:
    return World.generate(np.random.default_rng(seed), size)


def test_distance_wraps_and_takes_the_short_way_round():
    n = 64
    assert torus_distance(0, 1, 0, 63, n) == 2.0  # across the seam, not 62 the long way
    assert torus_distance(2, 2, 62, 62, n) == float(np.hypot(4, 4))
    assert float(wrap_delta(1, 63, n)) == 2.0 and float(wrap_delta(63, 1, n)) == -2.0
    # never longer than half the world in either axis
    rng = np.random.default_rng(0)
    a, b = rng.random((2, 500, 2)) * n
    d = torus_distance(a[:, 0], a[:, 1], b[:, 0], b[:, 1], n)
    assert d.max() <= np.hypot(n / 2, n / 2) + 1e-9


def test_a_step_across_the_seam_is_short_and_lands_inside():
    w = world()
    y, x, dist = w.step_toward(np.array([0.5]), np.array([0.5]), np.array([0.5]), np.array([float(w.size) - 0.5]), 2.0)
    assert dist[0] == 1.0  # one tile away through the seam
    assert 0 <= float(y[0]) < w.size and 0 <= float(x[0]) < w.size
    assert float(x[0]) > w.size - 2  # it stepped backwards through the wrap, not forwards across the map


def test_terrain_is_continuous_across_both_seams():
    n = 128
    for field in (periodic_noise(np.random.default_rng(7), n, 5), world(5, n).elevation):
        seam_x = np.abs(field[:, 0] - field[:, -1])
        seam_y = np.abs(field[0, :] - field[-1, :])
        inner_x = np.abs(field[:, 1] - field[:, 0])
        inner_y = np.abs(field[1, :] - field[0, :])
        # the jump at the seam is no bigger than a normal neighbouring step
        assert seam_x.mean() <= inner_x.mean() * 1.6 + 1e-6
        assert seam_y.mean() <= inner_y.mean() * 1.6 + 1e-6


def test_weather_drifts_and_wraps_without_a_seam():
    w = world(11, 80)
    wx = Weather.create(np.random.default_rng(11), w, n=20)
    before = wx.moisture.copy()
    for d in range(12):
        wx.step(d, np.random.default_rng(d), w.elevation[:20, 0] * 0 + 12.0)
    assert not np.allclose(before, wx.moisture)  # systems move
    assert np.isfinite(wx.moisture).all() and wx.moisture.min() >= 0
    seam = np.abs(wx.cloud[:, 0] - wx.cloud[:, -1]).mean()
    inner = np.abs(wx.cloud[:, 1] - wx.cloud[:, 0]).mean()
    assert seam <= inner * 2.0 + 1e-6


def test_sky_is_deterministic_periodic_and_local_time_follows_longitude():
    a, b = skymod.state(10, 6.0), skymod.state(10, 6.0)
    assert a == b
    # the sun comes back to the same longitude every day, and the moons to the same phase after their period
    assert abs(skymod.state(10, 6.0).sun_longitude - skymod.state(40, 6.0).sun_longitude) < 1e-9
    period = skymod.MOONS[0]["period"]  # 9.4 days: one full period is a whole day plus a fraction
    p0 = skymod.state(0, 0.0).moon_phase[0]
    p1 = skymod.state(int(period), (period % 1) * skymod.HOURS_PER_DAY).moon_phase[0]
    assert abs(((p1 - p0 + 0.5) % 1.0) - 0.5) < 1e-6
    light = skymod.light_field(skymod.state(5, 12.0), 64)
    assert light.argmax() != 0  # noon is at some longitude, and dawn sweeps around
    rolled = skymod.light_field(skymod.state(5, 18.0), 64)
    assert np.argmax(rolled) != np.argmax(light)
    assert light.min() >= 0 and light.max() <= 1.001


def test_light_is_continuous_across_the_seam():
    light = skymod.light_field(skymod.state(3, 9.0), 96)
    seam = abs(float(light[0]) - float(light[-1]))
    inner = abs(float(light[1]) - float(light[0]))
    assert seam <= inner * 2.0 + 1e-6


def test_eclipses_are_reproducible_and_rare():
    days = [d for d in range(4 * DAYS_PER_YEAR) if skymod.state(d, 0.0).eclipse]
    again = [d for d in range(4 * DAYS_PER_YEAR) if skymod.state(d, 0.0).eclipse]
    assert days == again
    assert len(days) < 4 * DAYS_PER_YEAR * 0.2  # an omen, not the weather


def test_planted_fields_yield_more_than_wild_ground():
    from research.civ_sim.world import FIELD

    w = world(2, 48)
    grass = np.argwhere((w.terrain == 1) & (w.food_cap > 5))[0]
    field = np.argwhere((w.terrain == 1) & (w.food_cap > 5))[1]
    w.building[tuple(field)] = FIELD
    w.build_progress[tuple(field)] = 1.0
    w.food[tuple(grass)] = w.food[tuple(field)] = 1.0
    light = np.ones((1, w.size), np.float32)
    w.regrow(1, light, np.zeros((w.size, w.size), np.float32) + 0.1, np.zeros((w.size, w.size), np.float32) + 14)
    assert w.food[tuple(field)] > w.food[tuple(grass)] * 1.5
