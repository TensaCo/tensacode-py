"""People on a toroidal planet, at full fidelity, under a real sky and drifting weather.

**Every person is a mind.** There is no background tier and no cohorts: each living person carries a
tensacode `Store` (minds.py) with percepts and appraisals as claims with provenance, a thought
stream, episodic memory that decays and consolidates, affect read off their own processing, theory
of mind, conversations that move claims as English sentences, and `tc.choose` over intentions under
norms and an ascription-weighted harm constraint. The body still lives in NumPy arrays — position,
needs, genome, skills, kin — because that is just an efficient layout for the same state, not a
lower level of detail.

What that costs is population. A mind is ~3 ms of thinking per simulated day, so the largest
watchable world is a few hundred people rather than thousands; `--people` is the knob and
`docs/civ-sim/slice-results.md` has the measured table. Smaller and complete beats larger and thin.

The day has four phases (night, morning, afternoon, evening); the world's physics runs once a day,
minds wake for the phases they are awake for. Deterministic by seed. Resources are conserved: every
change goes through the ledger.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import sky as skymod
from .weather import Weather
from .world import (BUILD_COST, BUILDING_NAMES, DAYS_PER_SEASON, DAYS_PER_YEAR, FIELD, GRANARY, GRASS, HEARTH, HOUSE, NONE, PALISADE,
                    SEASONS, SHRINE, WATCHTOWER, WELL, WORKSHOP, World, wrap_delta)

def _circular_mean_1d(values: np.ndarray, size: int) -> float:
    """Mean position on a wrapped axis. You cannot average coordinates that come round again."""
    ang = np.asarray(values, dtype=float) / size * 2 * np.pi
    m = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
    return float((m / (2 * np.pi) * size) % size)


# How many common stores can coexist. This is not a design limit any more: the ascription array
# grows a column whenever one is founded (`People.widen`). The only real bound is the dtype of
# `village`/`lineage`, and reaching it is announced in the world rather than passing in silence.
COMMONS_START_WIDTH = 8
COMMONS_DTYPE_LIMIT = 126  # int8 holds the id
PRESTIGE_NOISE = 0.2  # how much of standing is luck; the incumbency margin is derived from it
FORAGE, WOOD, REST, SOCIAL, RITUAL, RAID, QUARRY, CARE, BUILD, FARM, GUARD, SLEEP = range(12)
TASKS = ("forage", "gather wood", "rest", "talk with neighbours", "keep a ritual", "raid", "quarry stone", "care for kin", "build", "work the fields", "stand guard", "sleep")
MOTIFS = ("calm", "joy", "flow", "desire", "fear", "anger", "grief", "shame", "boredom", "suffering", "attachment", "awe")
PHASES = ("night", "morning", "afternoon", "evening")
PHASE_HOURS = (2.0, 8.0, 14.0, 20.0)
SYLL = ("ka", "lo", "mi", "ra", "tu", "se", "an", "do", "ve", "ni", "ya", "ze", "ol", "ib", "ur", "em", "sa", "ko", "li", "ta")
VILLAGE_NAMES = ("Aldmere", "Brenholt", "Coralin", "Dunmarsh", "Everlow", "Fenwick")
ROLES = ("gatherer", "farmer", "woodcutter", "mason", "guard", "trader", "healer", "leader")

GAIN, COUPLING, ASCRIPTION, FERTILITY, LONGEVITY, SKILL, TEMPER, SOCIAL_G, SKIN, HAIR, FACE, BUILD_G, EYES, NOSE, BROW, MOUTH = range(16)


def person_name(i: int) -> str:
    a, b, c = SYLL[(i * 7) % 20], SYLL[(i // 20 * 11 + 3) % 20], SYLL[(i // 400 * 13 + 5) % 20]
    return (a + b + (c if i % 3 else "")).capitalize()


@dataclass
class Village:
    id: int
    name: str
    cy: int
    cx: int
    food: float = 0.0
    wood: float = 0.0
    stone: float = 0.0
    leader: int = -1
    law_share: bool = False
    adherence_hist: list = field(default_factory=list)
    monuments: int = 0
    herd: float = 0.0  # livestock: a food store with legs. Grows on surplus, eaten in a lean year.
    last_raid_day: int = -999
    raids_led: int = 0
    plan: list = field(default_factory=list)
    buildings: dict = field(default_factory=dict)
    calendar: dict = field(default_factory=dict)
    cand_y: np.ndarray | None = None
    cand_x: np.ndarray | None = None
    wood_y: np.ndarray | None = None
    wood_x: np.ndarray | None = None
    stone_y: np.ndarray | None = None
    stone_x: np.ndarray | None = None


@dataclass
class Raid:
    attacker: int
    target: int
    warriors: np.ndarray
    started: int
    at_night: bool = False


class People:
    FIELDS = {
        "alive": np.bool_, "x": np.float32, "y": np.float32, "tx": np.float32, "ty": np.float32, "village": np.int8, "female": np.bool_,
        "born": np.int32, "energy": np.float32, "health": np.float32, "warmth": np.float32, "carry_food": np.float64, "carry_wood": np.float64,
        "carry_stone": np.float64, "wealth": np.float64, "share": np.float32, "martial": np.float32, "valence": np.float32,
        "arousal": np.float32, "viability": np.float32, "ms": np.float32, "burden": np.float32, "bond": np.int32, "mother": np.int32,
        "father": np.int32, "task": np.int8, "last_birth": np.int32, "children": np.int16, "rich": np.int16, "motif": np.int8,
        "raids": np.int16, "died": np.int32, "cause": np.int8, "role": np.int8, "skill_forage": np.float32, "skill_build": np.float32,
        "skill_fight": np.float32, "asleep": np.bool_, "sick": np.float32,
        # `village` is which COMMON STORE feeds you; `lineage` is which founding line you are of.
        # They start equal and come apart: marrying into another settlement changes the first and
        # never the second, and a band that splits off keeps the lineage it came from. Dialect,
        # kinship reckoning and who counts as "us" follow lineage; food follows the commons.
        "lineage": np.int8,
    }

    def __init__(self, cap: int, commons: int = COMMONS_START_WIDTH) -> None:
        self.cap = 0
        self.n = 0
        self.alpha_width = max(commons, COMMONS_START_WIDTH)
        self.grow(cap)

    def widen(self, commons: int, fill: float = 0.5) -> bool:
        """Make room for another common store. Returns False only at the dtype's own limit.

        A new column is how this person regards a settlement that did not exist when they were born.
        It starts at `fill` — neither kin nor stranger — and the founding then sets the real values.
        """
        if commons <= self.alpha_width:
            return True
        if commons > COMMONS_DTYPE_LIMIT:
            return False
        wide = np.full((self.cap, commons), fill, np.float32)
        wide[:, : self.alpha_width] = self.alpha_p
        self.alpha_p = wide
        self.alpha_width = commons
        return True

    def grow(self, cap: int) -> None:
        for name, dt in self.FIELDS.items():
            old = getattr(self, name, None)
            new = np.zeros(cap, dt)
            if old is not None:
                new[: self.cap] = old
            setattr(self, name, new)
        for name, width, dt in (("genome", 16, np.uint8), ("alpha_p", self.alpha_width, np.float32)):
            old = getattr(self, name, None)
            new = np.zeros((cap, width), dt)
            if old is not None:
                new[: self.cap, : old.shape[1]] = old
            setattr(self, name, new)
        for arr in (self.bond, self.mother, self.father, self.rich, self.died):
            arr[self.cap:] = -1
        self.cap = cap

    def add(self, k: int) -> np.ndarray:
        if self.n + k > self.cap:
            self.grow(max(self.cap * 2, self.n + k))
        ids = np.arange(self.n, self.n + k)
        self.n += k
        return ids

    def gene(self, idx, locus: int) -> np.ndarray:
        return self.genome[np.asarray(idx), locus].astype(np.float32) / 255.0


class Simulation:
    def __init__(self, seed: int = 1, *, villages: int = 3, people: int = 240, focal: int = 0, size: int = 72,
                 rich_every: int = 1, minds: bool = True) -> None:
        """Full fidelity: every person carries a tensacode mind. `people` is the only knob that
        trades scale against what the machine can do; `focal` is accepted and ignored, kept so older
        call sites and the measurement harness still run."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.world = World.generate(self.rng, size)
        self.weather = Weather.create(self.rng, self.world)
        self.day = 0
        self.phase = 1
        self.p = People(max(16_000, people * 2), commons=max(COMMONS_START_WIDTH, villages + 2))
        self.villages: list[Village] = []
        self.raids: list[Raid] = []
        self.events: list[dict] = []
        self.life: dict[int, list] = {}
        self.series: list[dict] = []
        self.transcript: list[dict] = []
        self.ledger = {"eaten": 0.0, "spoiled": 0.0, "wood_burned": 0.0, "stone_used": 0.0, "frost_loss": 0.0}
        self.counters = {"births": 0, "deaths": 0, "raids": 0, "trade_volume": 0.0, "sanctions": 0, "bonds": 0, "promotions": 0, "laws": 0,
                         "monuments": 0, "migrants": 0, "buildings": 0, "conversations": 0, "claims_transmitted": 0, "festivals": 0, "omens": 0, "utterances": 0, "not_understood": 0, "ambiguous": 0, "misheard": 0}
        self.prices: list[float] = []
        self.timing = {"background_ms": 0.0, "minds_ms": 0.0, "society_ms": 0.0, "weather_ms": 0.0, "economy_ms": 0.0, "ticks": 0}
        self.rich_every = rich_every
        from . import language as lang

        self.base_lexicon = lang.base_lexicon()
        # one dialect per founding line. Splitting off a new commons does not create a new dialect;
        # drifting apart over generations does.
        self.dialects = [lang.dialect(self.base_lexicon, i, self.rng) for i in range(villages)]
        self.deaths_today = np.zeros(villages, np.int32)
        self.dirty_tiles: set = set()
        self.economy = None
        self._place_villages(villages)
        self._populate(people)
        from .economy import Economy
        from .settlements import Settlements

        self.economy = Economy(self)
        self.settlements = Settlements(self)
        import tensacode as tc
        from tensacode.backends.builtin import UtilityChooser

        self.runtime = tc.Runtime([UtilityChooser()])
        self.minds = None
        self.settlements.recompute()
        if minds:
            from .minds import Minds

            self.minds = Minds(self)
        self.world_total_start = self.totals()

    # ------------------------------------------------------------ clock

    @property
    def sky(self):
        return skymod.state(self.day, PHASE_HOURS[self.phase])

    @property
    def year(self) -> int:
        return self.day // DAYS_PER_YEAR

    @property
    def season(self) -> int:
        return (self.day // DAYS_PER_SEASON) % 4

    def base_temp(self) -> np.ndarray:
        return skymod.season_temperature(self.day, self.world.size)

    def light_at(self, y, x) -> np.ndarray:
        return skymod.light_field(self.sky, self.world.size)[np.asarray(x).astype(int) % self.world.size]

    def weather_at(self, y, x) -> tuple:
        w = self.weather
        return (float(w.at(y, x, w.rain)), float(w.at(y, x, w.snow)),
                float(self.base_temp()[int(y) % self.world.size] + w.at(y, x, w.anomaly)), float(w.at(y, x, w.cloud)))

    # ------------------------------------------------------------ setup

    def _place_villages(self, k: int) -> None:
        w = self.world
        size = w.size
        smooth = np.where(w.terrain == GRASS, w.food_cap, 0.0)
        for _ in range(3):
            smooth = (smooth + np.roll(smooth, 4, axis=0) + np.roll(smooth, -4, axis=0) + np.roll(smooth, 4, axis=1) + np.roll(smooth, -4, axis=1)) / 5
        smooth[w.terrain != GRASS] = -1
        chosen: list[tuple[int, int]] = []
        for flat in np.argsort(-smooth, axis=None):
            cy, cx = divmod(int(flat), size)
            if all(float(w.distance(cy, cx, a, b)) > size * 0.26 for a, b in chosen):
                chosen.append((cy, cx))
            if len(chosen) == k:
                break
        for i, (cy, cx) in enumerate(chosen):
            v = Village(i, self._village_name(i), cy, cx)
            self._pick_sites(v)
            self.villages.append(v)
            self._queue_named(v, [GRANARY, HEARTH, WELL] + [HOUSE] * 5 + [FIELD] * 14 + [SHRINE, WORKSHOP])

    def _village_name(self, i: int) -> str:
        """Names for the founding bands. The hand-written list runs out; the world does not."""
        if i < len(VILLAGE_NAMES):
            return VILLAGE_NAMES[i]
        a = ("Ald", "Bren", "Cor", "Dun", "Esk", "Far", "Gell", "Hal", "Ith", "Kel", "Mor", "Nor")
        b = ("mere", "holt", "alin", "marsh", "ridge", "fell", "wick", "gard", "combe", "stead")
        j = i - len(VILLAGE_NAMES)
        return a[j % len(a)] + b[(j // len(a)) % len(b)]

    def _pick_sites(self, v: Village) -> None:
        """Which tiles this commons works: the food, wood and stone within reach of its centre."""
        w = self.world
        yy, xx = np.mgrid[0:w.size, 0:w.size]
        d = w.distance(yy, xx, v.cy, v.cx)
        for kind, mask, radius in (("cand", w.food_cap > 1.0, 16), ("wood", w.wood_cap > 0, 20), ("stone", w.stone > 0, 26)):
            m = mask & (d <= radius)
            if not m.any():
                m = mask & (d <= radius * 2)
            ty, tx = np.nonzero(m)
            setattr(v, f"{kind}_y", ty)
            setattr(v, f"{kind}_x", tx)

    def _queue_named(self, v: Village, kinds: list) -> None:
        w = self.world
        spots = []
        for r in range(1, 7):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    y, x = (v.cy + dy) % w.size, (v.cx + dx) % w.size
                    if w.terrain[y, x] == GRASS and w.building[y, x] == NONE and (y, x) not in [(b[1], b[2]) for b in v.plan]:
                        spots.append((y, x))
        self.rng.shuffle(spots)
        for b, (y, x) in zip(kinds, spots):
            v.plan.append((b, y, x))

    def _populate(self, n: int) -> None:
        p, rng = self.p, self.rng
        ids = p.add(n)
        k = len(self.villages)
        vil = (np.arange(n) % k).astype(np.int8)
        p.alive[ids] = True
        p.village[ids] = vil
        p.lineage[ids] = vil
        p.female[ids] = rng.random(n) < 0.5
        age_years = rng.gamma(2.2, 11.0, n).clip(0, 70)
        p.born[ids] = (-age_years * DAYS_PER_YEAR).astype(np.int32)
        p.genome[ids] = rng.integers(30, 226, (n, 16), dtype=np.uint8)
        for v in self.villages:
            m = ids[vil == v.id]
            p.y[m] = (v.cy + rng.normal(0, 3, len(m))) % self.world.size
            p.x[m] = (v.cx + rng.normal(0, 3, len(m))) % self.world.size
        p.tx[ids], p.ty[ids] = p.x[ids], p.y[ids]
        p.energy[ids], p.health[ids], p.warmth[ids] = 0.85, 0.9, 0.8
        p.share[ids] = np.clip(rng.normal(0.55, 0.15, n), 0, 1)
        p.martial[ids] = np.clip(0.25 + 0.4 * p.gene(ids, TEMPER) + rng.normal(0, 0.1, n), 0, 1)
        base = 0.3 + 0.7 * p.gene(ids, ASCRIPTION)
        for j in range(k):
            p.alpha_p[ids, j] = np.where(vil == j, 1.0, base)
        p.viability[ids] = 0.8
        p.last_birth[ids] = -999
        p.skill_forage[ids] = 0.2 + 0.4 * p.gene(ids, SKILL)
        p.skill_build[ids] = 0.15 + 0.3 * p.gene(ids, SKILL)
        p.skill_fight[ids] = 0.1 + 0.4 * p.gene(ids, TEMPER)
        p.role[ids] = rng.integers(0, 6, n)
        for v in self.villages:
            c = int((vil == v.id).sum())
            v.food, v.wood, v.stone = 14.0 * c, 8.0 * c, 20.0
        self._bond(initial=True)

    # ------------------------------------------------------------ helpers

    def age(self, idx) -> np.ndarray:
        return (self.day - self.p.born[np.asarray(idx)]) / DAYS_PER_YEAR

    def living(self) -> np.ndarray:
        return np.flatnonzero(self.p.alive[: self.p.n])

    def log(self, kind: str, text: str, who: tuple = (), notable: bool = True) -> None:
        ev = {"day": self.day, "year": self.year, "season": SEASONS[self.season], "phase": PHASES[self.phase], "kind": kind, "text": text, "who": [int(x) for x in who]}
        if notable:
            self.events.append(ev)
            del self.events[:-4000]
        for x in who:
            self.life.setdefault(int(x), []).append((self.day, text))

    def name(self, i) -> str:
        return f"{person_name(int(i))} of {self.villages[int(self.p.village[int(i)])].name}"

    def village_of(self, i) -> Village:
        return self.villages[int(self.p.village[int(i)])]

    def totals(self) -> dict:
        p, w = self.p, self.world
        alive = p.alive[: p.n]
        held = self.economy.totals() if self.economy is not None else {"wood": 0.0, "stone": 0.0, "tools": 0.0}
        return {
            "food": float(w.food.sum() + sum(v.food + v.herd * self.FOOD_PER_ANIMAL for v in self.villages)
                          + p.carry_food[: p.n][alive].sum() + p.wealth[: p.n][alive].sum()),
            "wood": float(w.wood.sum() + sum(v.wood for v in self.villages) + p.carry_wood[: p.n][alive].sum() + held["wood"]),
            "stone": float(w.stone.sum() + sum(v.stone for v in self.villages) + p.carry_stone[: p.n][alive].sum() + held["stone"]),
            "tools": float(held["tools"]),
        }

    # ------------------------------------------------------------ the day

    def step(self) -> None:
        import tensacode as tc

        with tc.use(self.runtime):
            self._step()
        self.runtime.trace.spans.clear()

    def _step(self) -> None:
        t0 = time.perf_counter()
        self.weather.step(self.day, self.rng, self.base_temp())
        light = skymod.daily_light(self.day, self.world.size)
        temp_tiles = self.base_temp()[:, None] + self.weather.tile(self.weather.anomaly)
        self.world.regrow(self.season, light[None, :], self.weather.tile(self.weather.rain), temp_tiles)
        self._frost_damage()
        self.timing["weather_ms"] += (time.perf_counter() - t0) * 1e3

        for phase in range(4):
            self.phase = phase
            t2 = time.perf_counter()
            idx = self.living()
            self.rebuild_bins(idx)
            self._sleep_and_wake(idx)
            awake = idx[~self.p.asleep[idx]]
            self._choose_tasks(awake)
            t3 = time.perf_counter()
            if self.minds is not None:
                self.minds.tick(phase)
            t4 = time.perf_counter()
            awake = self.living()
            awake = awake[~self.p.asleep[awake]]
            self._move(awake)
            self._gather(awake)
            self._build(awake)
            self.timing["background_ms"] += (t3 - t2) * 1e3 + (time.perf_counter() - t4) * 1e3
            self.timing["minds_ms"] += (t4 - t3) * 1e3

        t5 = time.perf_counter()
        self.phase = 3  # the day's bookkeeping happens in the evening, so the sky and sleep agree
        idx = self.living()
        self._eat(idx)
        self._health_and_death(idx)
        self._bond()
        self._births()
        idx = self.living()
        self._culture_and_affect(idx)
        self._herds(idx)
        self._economy_day(idx)
        if self.day % 6 == 0:
            self._trade()
        if self.day % DAYS_PER_SEASON == DAYS_PER_SEASON - 1:
            self._institutions()
        if self.day % 12 == 11:
            self.settlements.recompute()  # settlements are read off the world, not kept in it
            self._fission()
        if self.day % DAYS_PER_YEAR == DAYS_PER_YEAR - 1:
            self._language_change()
        if self.day % 8 == 5:
            self._raid_decisions()
        self._resolve_raids()
        if self.minds is not None:
            self.minds.refill()
        self._record()
        self.timing["society_ms"] += (time.perf_counter() - t5) * 1e3
        self.timing["ticks"] += 1
        self.day += 1

    def _economy_day(self, idx: np.ndarray) -> None:
        """Craft, decay, hold market, settle debts. Prices and money come out of this, not into it."""
        e = self.economy
        if e is None or len(idx) == 0:
            return
        t = time.perf_counter()
        e.make_tools(idx)
        e.decay(idx)
        e.clear_markets()
        e.spread_habit()
        if self.day % 6 == 3:
            e.caravans()
        e.credit_round()
        e.settle_debts()
        if self.day % DAYS_PER_SEASON == 0:
            e.collect_tax()
        e.record()
        self.timing["economy_ms"] += (time.perf_counter() - t) * 1e3

    # one animal is worth this much food: fattened on surplus grain, slaughtered when the store runs out
    FOOD_PER_ANIMAL = 12.0

    def _herds(self, idx: np.ndarray) -> None:
        """Livestock as the oldest kind of savings: surplus grain walks around on four legs until a
        lean season, and then it is eaten. Every animal is food that was really put in, so the
        conservation ledger covers them like any other store."""
        counts = np.bincount(self.p.village[idx], minlength=len(self.villages)).clip(1)
        for v in self.villages:
            per_head = v.food / counts[v.id]
            # a herd needs hands: about one and a half head per person is as much as a village can
            # tend. Without this cap the settlement turned its whole surplus into cattle and then
            # starved beside it — 2,366 head for 159 people in one run before the cap existed.
            room = max(0.0, 1.5 * counts[v.id] - v.herd)
            if per_head > 10.0 and room > 0.5:  # fatten: grain the village will not eat walks away
                spend = min(v.food - 10.0 * counts[v.id], 0.05 * v.food, room * self.FOOD_PER_ANIMAL)
                if spend > 0.5:
                    v.food -= spend
                    v.herd += spend / self.FOOD_PER_ANIMAL
                    self.ledger["to_herd"] = self.ledger.get("to_herd", 0.0) + spend
            elif per_head < 8.0 and v.herd >= 1.0:  # slaughter before anyone goes hungry
                take = min(v.herd, max(1.0, 0.08 * v.herd))
                v.herd -= take
                v.food += take * self.FOOD_PER_ANIMAL
                self.ledger["from_herd"] = self.ledger.get("from_herd", 0.0) + take * self.FOOD_PER_ANIMAL
                if take >= 2:
                    self.log("herd", f"{v.name} slaughtered {int(take)} of the herd to get through a lean season")
            loss = v.herd * 0.001  # animals die of their own accord
            v.herd -= loss
            self.ledger["herd_lost"] = self.ledger.get("herd_lost", 0.0) + loss * self.FOOD_PER_ANIMAL

    def _frost_damage(self) -> None:
        w, wx = self.world, self.weather
        cold = (wx.tile(wx.snow) > 0.02) & (w.food > 0)
        loss = np.where(cold, w.food * 0.04, 0.0)
        w.food -= loss
        self.ledger["frost_loss"] += float(loss.sum())

    def rebuild_bins(self, idx: np.ndarray, cell: int = 8) -> None:
        """Bucket living people by coarse tile, so minds can ask who is nearby cheaply."""
        p = self.p
        self.bin_cell = cell
        n = self.world.size // cell
        key = (p.y[idx].astype(int) // cell % n) * n + (p.x[idx].astype(int) // cell % n)
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        edges = np.searchsorted(sorted_key, np.arange(n * n + 1))
        self._bin_idx, self._bin_edges, self._bin_n = idx[order], edges, n

    def neighbours(self, pid: int, radius: float = 6.0) -> np.ndarray:
        """People within ``radius`` tiles on the torus, from the coarse bins."""
        p, w = self.p, self.world
        n, cell = self._bin_n, self.bin_cell
        by, bx = int(p.y[pid]) // cell % n, int(p.x[pid]) // cell % n
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                b = ((by + dy) % n) * n + ((bx + dx) % n)
                out.append(self._bin_idx[self._bin_edges[b]:self._bin_edges[b + 1]])
        cand = np.concatenate(out) if out else np.array([], int)
        if len(cand) == 0:
            return cand
        d = w.distance(p.y[cand], p.x[cand], p.y[pid], p.x[pid])
        keep = (d < radius) & (cand != pid)
        return cand[keep][np.argsort(d[keep])]

    def _sleep_and_wake(self, idx: np.ndarray) -> None:
        p = self.p
        light = self.light_at(p.y[idx], p.x[idx])
        dark = light < 0.22
        age = self.age(idx)
        guard = dark & (p.martial[idx] > 0.55) & (p.health[idx] > 0.5) & (age >= 16) & (age < 55)
        asleep = dark & ~guard & (p.task[idx] != RAID)
        p.asleep[idx] = asleep
        p.task[idx[asleep]] = SLEEP
        awake_guards = idx[guard & (p.task[idx] != RAID)]
        p.task[awake_guards] = GUARD
        p.asleep[idx[~dark]] = False

    def _local_prices(self, vil: np.ndarray) -> dict:
        """Each person's own settlement's price for each good, or a flat 1.0 with no economy."""
        e = self.economy
        if e is None:
            return {g: np.ones(len(vil)) for g in ("wood", "stone", "tools")}
        return {g: np.array([e.price(v.id, g) for v in self.villages])[vil] for g in ("wood", "stone", "tools")}

    def _choose_tasks(self, idx: np.ndarray) -> None:
        p, rng = self.p, self.rng
        age = self.age(idx)
        # At full fidelity every task comes from somebody's own decision. This heuristic is only a
        # fallback for people whose mind has not decided anything yet — a newborn on its first day,
        # or anyone not yet reached by a thinking phase.
        undecided = np.ones(len(idx), bool)
        if self.minds is not None:
            for k, pid in enumerate(idx):
                m = self.minds.minds.get(int(pid))
                if m is not None and m.decision_id is not None:
                    undecided[k] = False
        free = undecided & (p.task[idx] != RAID) & ~p.asleep[idx]
        counts = np.bincount(p.village[idx], minlength=len(self.villages)).clip(1)
        vil = p.village[idx]
        food_pc = (np.array([v.food for v in self.villages]) / counts)[vil]
        wood_pc = (np.array([v.wood for v in self.villages]) / counts)[vil]
        task = np.full(len(idx), REST, np.int8)
        adult = (age >= 12) & (age < 62)
        hungry = (p.energy[idx] < 0.85) | (food_pc < 8)
        task[adult & hungry] = FORAGE
        has_fields = np.array([v.buildings.get(FIELD, 0) > 0 for v in self.villages])[vil]
        task[adult & hungry & has_fields & (rng.random(len(idx)) < 0.55)] = FARM
        # what work pays: the local price of what the work produces, and what you are personally
        # better at than the alternative. Between them they produce specialization.
        px = self._local_prices(vil)
        advantage = np.clip(p.skill_build[idx] - p.skill_forage[idx], -1, 1)
        need_wood = adult & (wood_pc < (7 if self.season >= 1 else 4)) & (p.energy[idx] > 0.35) & (
            rng.random(len(idx)) < 0.22 * (0.5 + px["wood"]) * (1 + 0.6 * np.maximum(advantage, 0)))
        task[need_wood] = WOOD
        planning = np.array([bool(v.plan) for v in self.villages])[vil]
        builders = adult & planning & ~need_wood & (p.energy[idx] > 0.4) & (
            rng.random(len(idx)) < 0.2 * (1 + 0.8 * np.maximum(advantage, 0)))
        has_shop = np.array([v.buildings.get(WORKSHOP, 0) > 0 for v in self.villages])[vil]
        crafters = adult & has_shop & ~need_wood & ~builders & (p.energy[idx] > 0.45) & (
            rng.random(len(idx)) < 0.10 * px["tools"] / 2 * (1 + 1.2 * np.maximum(advantage, 0)))
        task[builders | crafters] = BUILD
        burdened = adult & ~hungry & ~need_wood & ~builders & ~crafters & (rng.random(len(idx)) < 0.3 * (0.4 + px["stone"]) + 0.2 * p.burden[idx])
        task[burdened] = QUARRY
        social = adult & (task == REST) & (rng.random(len(idx)) < 0.35)
        task[social] = SOCIAL
        sel = idx[free]
        p.task[sel] = task[free]
        self._retarget(sel)

    def land_crowding(self, v: Village, pop: int) -> float:
        """People per unit of workable land within this commons' reach.

        Measured, not invented: across a run this sits at 0.8–0.9 while a settlement is comfortable
        and passes 1.0 as it fills its ground (observed 1.12 and 1.49 for the two largest commons at
        day 1,500 of seed 1). Both the outward drift of the working radius and the decision to open a
        second store read this same number, so there is one notion of "crowded" in the model.
        """
        if v.cand_y is None or len(v.cand_y) == 0:
            return 0.0
        near_cap = float(self.world.food_cap[v.cand_y, v.cand_x].sum())
        return pop / max(1.0, near_cap / 90.0)

    def working_range(self, who: np.ndarray, village_id: int, crowding: float) -> np.ndarray:
        """How far from the common store a person is content to work, and so to live.

        A crowded settlement pushes its working radius outward: the ground near the store is already
        claimed, so some people take the far patches and end up living out there. Each person's
        preference is a fixed function of who they are, scaled by how crowded their store is, so it
        is deterministic and stable rather than a daily coin toss — which is what lets a group of
        outliers persist long enough to be recognised as a place of its own.
        """
        h = ((np.asarray(who, dtype=np.int64) * 2654435761) % 1000) / 1000.0
        return 1.5 + 14.0 * h * min(1.0, crowding)

    def _retarget(self, sel: np.ndarray) -> None:
        p, rng, w = self.p, self.rng, self.world
        if len(sel) == 0:
            return
        living = self.living()
        pop = np.bincount(p.village[living], minlength=len(self.villages)).clip(1)
        for v in self.villages:
            m = sel[p.village[sel] == v.id]
            if len(m) == 0:
                continue
            crowding = self.land_crowding(v, int(pop[v.id]))
            tk = p.task[m]
            for task, ys, xs, stock in ((FORAGE, v.cand_y, v.cand_x, w.food), (WOOD, v.wood_y, v.wood_x, w.wood), (QUARRY, v.stone_y, v.stone_x, w.stone)):
                g = m[tk == task]
                if len(g) == 0 or ys is None or len(ys) == 0:
                    continue
                cur = stock[p.ty[g].astype(int) % w.size, p.tx[g].astype(int) % w.size]
                need = g[cur < 2.0]
                if len(need) == 0:
                    continue
                pick = rng.integers(0, len(ys), (len(need), 6))
                dist = w.distance(ys[pick], xs[pick], p.y[need, None], p.x[need, None])
                from_home = w.distance(ys[pick], xs[pick], v.cy, v.cx)
                want = self.working_range(need, v.id, crowding)[:, None]
                ring = np.exp(-np.abs(from_home - want) / 6.0)
                val = stock[ys[pick], xs[pick]] / (1 + 0.3 * dist) * ring
                best = pick[np.arange(len(need)), val.argmax(1)]
                p.ty[need], p.tx[need] = ys[best] + 0.5, xs[best] + 0.5
            farmers = m[tk == FARM]
            if len(farmers):
                fy, fx = np.nonzero((w.building == FIELD) & (w.build_owner == v.id) & (w.build_progress >= 1.0))
                if len(fy):
                    pick = rng.integers(0, len(fy), len(farmers))
                    p.ty[farmers], p.tx[farmers] = fy[pick] + 0.5, fx[pick] + 0.5
                else:
                    p.task[farmers] = FORAGE
            crew = m[tk == BUILD]
            if len(crew) and v.plan:
                _, by, bx = v.plan[0]
                p.ty[crew], p.tx[crew] = by + 0.5, bx + 0.5
            home = m[(tk == REST) | (tk == SOCIAL) | (tk == RITUAL) | (tk == CARE) | (tk == SLEEP) | (tk == GUARD)]
            if len(home):
                p.ty[home] = (v.cy + rng.normal(0, 2.5, len(home))) % w.size
                p.tx[home] = (v.cx + rng.normal(0, 2.5, len(home))) % w.size

    def _move(self, idx: np.ndarray) -> None:
        p, w, wx = self.p, self.world, self.weather
        rain = wx.at(p.y[idx], p.x[idx], wx.rain)
        snow = wx.at(p.y[idx], p.x[idx], wx.snow)
        mud = 1 - np.clip(0.5 * rain + 1.2 * snow, 0, 0.5)
        speed = np.where(p.task[idx] == RAID, 3.6, 3.2) * mud * np.clip(p.health[idx] + 0.3, 0.3, 1.2)
        y, x, _ = w.step_toward(p.y[idx], p.x[idx], p.ty[idx], p.tx[idx], speed)
        moved = (np.abs(wrap_delta(y, p.y[idx], w.size)) + np.abs(wrap_delta(x, p.x[idx], w.size))) > 0.05
        p.y[idx], p.x[idx] = y, x
        ty, tx = y.astype(int) % w.size, x.astype(int) % w.size
        np.add.at(w.trample, (ty[moved], tx[moved]), 0.02)
        np.clip(w.trample, 0, 1, out=w.trample)

    def _gather(self, idx: np.ndarray) -> None:
        p, w, wx = self.p, self.world, self.weather
        arrived = w.distance(p.ty[idx], p.tx[idx], p.y[idx], p.x[idx]) < 1.4
        mult = np.clip(1 - 0.35 * wx.at(p.y[idx], p.x[idx], wx.rain) - 0.8 * wx.at(p.y[idx], p.x[idx], wx.snow), 0.25, 1.0)
        light = self.light_at(p.y[idx], p.x[idx])
        for task, stock, carry, rate in ((FORAGE, w.food, p.carry_food, 4.5), (FARM, w.food, p.carry_food, 6.0), (WOOD, w.wood, p.carry_wood, 1.6), (QUARRY, w.stone, p.carry_stone, 1.0)):
            m = arrived & (p.task[idx] == task)
            g = idx[m]
            if len(g) == 0:
                continue
            skill = (p.skill_forage[g] if task in (FORAGE, FARM) else p.skill_build[g]) + 0.5
            capital = self.economy.tool_bonus(g) if self.economy is not None else 1.0
            demand = (rate * skill * capital * np.clip(p.health[g], 0.2, 1) * mult[m] * np.clip(0.4 + light[m], 0.4, 1.3)).astype(np.float64)
            gy, gx = p.y[g].astype(int) % w.size, p.x[g].astype(int) % w.size
            flat = stock.reshape(-1)
            got = np.zeros(len(g))
            left = demand.copy()
            for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                if not left.any():
                    break
                tile = ((gy + dy) % w.size) * w.size + ((gx + dx) % w.size)
                dsum = np.bincount(tile, weights=left, minlength=flat.size)
                touched = np.flatnonzero(dsum)
                taken = np.minimum(dsum[touched], flat[touched])
                ratio = np.zeros(flat.size)
                ratio[touched] = taken / dsum[touched]
                share = left * ratio[tile]
                got += share
                left -= share
                flat[touched] -= taken
            np.maximum(flat, 0, out=flat)
            carry[g] += got
            if task == FARM:
                fy, fx = p.y[g].astype(int) % w.size, p.x[g].astype(int) % w.size
                np.add.at(w.tilled, (fy, fx), 0.05)
                np.clip(w.tilled, 0, 1, out=w.tilled)
                p.skill_forage[g] = np.minimum(1.0, p.skill_forage[g] + 0.002)
            elif task == FORAGE:
                p.skill_forage[g] = np.minimum(1.0, p.skill_forage[g] + 0.001)
            else:
                p.skill_build[g] = np.minimum(1.0, p.skill_build[g] + 0.001)

    def _build(self, idx: np.ndarray) -> None:
        p, w = self.p, self.world
        for v in self.villages:
            if not v.plan:
                continue
            b, by, bx = v.plan[0]
            crew = idx[(p.village[idx] == v.id) & (p.task[idx] == BUILD) & (w.distance(p.y[idx], p.x[idx], by + 0.5, bx + 0.5) < 1.6)]
            if len(crew) == 0:
                continue
            if w.build_progress[by, bx] <= 0.0:
                cost_f, cost_w = BUILD_COST[b]
                if v.food < cost_f + 5 or v.wood < cost_w:
                    continue
                v.food -= cost_f
                v.wood -= cost_w
                w.ledger["build_food"] += cost_f
                w.ledger["build_wood"] += cost_w
                w.building[by, bx] = b
                w.build_owner[by, bx] = v.id
            w.build_progress[by, bx] = min(1.0, float(w.build_progress[by, bx]) + float((0.09 * (0.5 + p.skill_build[crew])).sum()))
            p.skill_build[crew] = np.minimum(1.0, p.skill_build[crew] + 0.003)
            self.dirty_tiles.add((int(by), int(bx)))
            if w.build_progress[by, bx] >= 1.0:
                v.plan.pop(0)
                v.buildings[b] = v.buildings.get(b, 0) + 1
                self.counters["buildings"] += 1
                if b == FIELD:
                    w.tilled[by, bx] = max(float(w.tilled[by, bx]), 0.4)
                elif b in (GRANARY, SHRINE, WORKSHOP, WATCHTOWER, WELL):
                    self.log("build", f"{v.name} finished a {BUILDING_NAMES[b]}", tuple(int(c) for c in crew[:2]))

    def _eat(self, idx: np.ndarray) -> None:
        p, w, wx = self.p, self.world, self.weather
        age = self.age(idx)
        need = np.where(age < 12, 0.2, np.where(age > 60, 0.26, 0.34))
        eaten = np.minimum(p.carry_food[idx], need)
        p.carry_food[idx] -= eaten
        law = np.array([v.law_share for v in self.villages])[p.village[idx]]
        contrib = np.where(law, np.maximum(0.85, p.share[idx]), 0.35 + 0.6 * p.share[idx])
        dep = p.carry_food[idx] * contrib
        p.wealth[idx] += p.carry_food[idx] - dep
        p.carry_food[idx] = 0.0
        # wood and stone split the same way food does: what you hand to the common store, and what you
        # keep as your own — the private half is what there is to trade.
        from .economy import STONE as G_STONE, WOOD as G_WOOD

        for arr, attr, good in ((p.carry_wood, "wood", G_WOOD), (p.carry_stone, "stone", G_STONE)):
            kept = arr[idx] * 0.5  # the sharing norm is about food; timber and stone are half yours
            if self.economy is not None:
                self.economy.deposit(idx, good, kept)
            else:
                kept = np.zeros(len(idx))
            amt = np.bincount(p.village[idx], weights=arr[idx] - kept, minlength=len(self.villages))
            for v in self.villages:
                setattr(v, attr, getattr(v, attr) + float(amt[v.id]))
            arr[idx] = 0.0
        depv = np.bincount(p.village[idx], weights=dep, minlength=len(self.villages))
        for v in self.villages:
            v.food += float(depv[v.id])
        rest = need - eaten
        from_w = np.minimum(p.wealth[idx], rest)
        p.wealth[idx] -= from_w
        eaten += from_w
        rest -= from_w
        # The common store rations only when it has to. If it can cover everyone who is short, it
        # does, and nobody starves beside a full granary. When requests exceed what is there, the
        # order matters: children, the old and the sick are served first whatever they brought in,
        # and an able adult who brought nothing home is served last. So a *lean* day is a hungry
        # evening — which is what makes anyone need to borrow — and a fat one is not.
        dependent = (age < 12) | (age > 58) | (p.sick[idx] > 0.3)
        priority = np.where(dependent, 1.0, np.clip(0.3 + 1.4 * dep / np.maximum(need, 1e-9), 0.3, 1.0))
        asked = np.bincount(p.village[idx], weights=rest, minlength=len(self.villages))
        rationing = np.array([v.food < asked[v.id] for v in self.villages])
        weighted = np.where(rationing[p.village[idx]], rest * priority, rest)
        req = np.bincount(p.village[idx], weights=weighted, minlength=len(self.villages))
        grant = np.zeros(len(self.villages))
        for v in self.villages:
            give = min(v.food, float(req[v.id]))
            grant[v.id] = give / req[v.id] if req[v.id] > 0 else 0.0
            v.food -= give
        eaten += weighted * grant[p.village[idx]]
        self.counters["rationed_days"] = self.counters.get("rationed_days", 0) + int(rationing.sum())
        self.ledger["eaten"] += float(eaten.sum())
        p.energy[idx] = np.clip(p.energy[idx] + 0.12 * (eaten / need - 0.85), 0, 1)
        # warmth: cold costs health unless there is fuel and shelter
        temp = self.base_temp()[p.y[idx].astype(int) % w.size] + wx.at(p.y[idx], p.x[idx], wx.anomaly)
        sheltered = np.array([1.0 if (v.buildings.get(HEARTH, 0) or v.buildings.get(HOUSE, 0)) else 0.0 for v in self.villages])[p.village[idx]]
        cold_need = np.clip((8 - temp) / 12, 0, 1)
        wreq = np.bincount(p.village[idx], weights=0.04 * cold_need, minlength=len(self.villages))
        short = np.zeros(len(self.villages))
        for v in self.villages:
            burn = min(v.wood, float(wreq[v.id]))
            v.wood -= burn
            self.ledger["wood_burned"] += burn
            short[v.id] = 1 - burn / wreq[v.id] if wreq[v.id] > 0 else 0.0
        cold = cold_need * (0.35 + 0.65 * short[p.village[idx]]) * np.where(sheltered > 0, 0.45, 1.0)
        p.warmth[idx] = np.clip(1 - cold, 0, 1)
        p.health[idx] -= (0.010 * cold).astype(np.float32)
        rain = wx.at(p.y[idx], p.x[idx], wx.rain)
        wells = np.array([min(1, v.buildings.get(WELL, 0)) for v in self.villages])[p.village[idx]]
        risk = (0.004 * cold + 0.003 * rain * (1 - p.energy[idx])) * (1 - 0.4 * wells)
        p.sick[idx] = np.clip(p.sick[idx] * 0.9 + (self.rng.random(len(idx)) < risk) * 0.6, 0, 1)
        p.health[idx] -= (0.02 * p.sick[idx]).astype(np.float32)
        np.clip(p.health, 0, 1, out=p.health)
        for v in self.villages:
            s = v.food * (0.003 if v.buildings.get(GRANARY, 0) else 0.01)
            v.food -= s
            self.ledger["spoiled"] += s
        s = p.wealth[idx] * 0.006
        p.wealth[idx] -= s
        self.ledger["spoiled"] += float(s.sum())

    def _health_and_death(self, idx: np.ndarray) -> None:
        p, rng = self.p, self.rng
        starving = p.energy[idx] < 0.2
        p.health[idx] = np.clip(p.health[idx] + np.where(starving, -0.025, 0.005), 0, 1)
        age = self.age(idx)
        longevity = 0.7 + 0.6 * (1 - p.gene(idx, LONGEVITY))
        annual = 0.0011 * np.exp(0.075 * age) * longevity + np.where(age < 5, 0.03, 0.0)
        hazard = annual / DAYS_PER_YEAR + np.where(p.health[idx] < 0.3, 0.03 * (0.3 - p.health[idx]) / 0.3, 0.0)
        doomed = (rng.random(len(idx)) < hazard) | (p.health[idx] <= 0.0)
        for i, a, e, sick in zip(idx[doomed], age[doomed], p.energy[idx][doomed], p.sick[idx][doomed]):
            cause = "starvation" if e < 0.25 else ("illness" if sick > 0.3 else ("old age" if a > 52 else "illness"))
            self.kill(int(i), cause)

    def kill(self, i: int, cause: str) -> None:
        p = self.p
        if not p.alive[i]:
            return
        p.alive[i] = False
        p.died[i] = self.day
        causes = ("starvation", "old age", "illness", "raid", "cold")
        p.cause[i] = causes.index(cause) if cause in causes else 2
        self.counters["deaths"] += 1
        self.deaths_today[int(p.village[i])] += 1
        v = self.village_of(i)
        v.food += float(p.wealth[i] + p.carry_food[i])
        v.wood += float(p.carry_wood[i])
        v.stone += float(p.carry_stone[i])
        p.wealth[i] = p.carry_food[i] = p.carry_wood[i] = p.carry_stone[i] = 0.0
        if self.economy is not None:
            self.economy.on_death(int(i))
        partner = int(p.bond[i])
        if partner >= 0:
            p.bond[partner] = -1
            p.ms[partner] = min(1.0, p.ms[partner] + 0.8)
        kin = [int(k) for k in (p.mother[i], p.father[i]) if k >= 0 and p.alive[k]]
        kids = [int(c) for c in np.flatnonzero(((p.mother[: p.n] == i) | (p.father[: p.n] == i)) & p.alive[: p.n])]
        for k in kin + kids:
            p.ms[k] = min(1.0, p.ms[k] + 0.6)
            p.arousal[k] = min(1.0, p.arousal[k] + 0.4)
        notable = p.rich[i] >= 0 or v.leader == i
        self.log("death", f"{self.name(i)} died of {cause} at {int(self.age([i])[0])}", (i,), notable)
        if self.minds is not None:
            self.minds.on_death(i, [partner] + kin + kids, cause)
        if v.leader == i:
            v.leader = -1

    def _bond(self, initial: bool = False) -> None:
        p, rng = self.p, self.rng
        idx = self.living()
        age = self.age(idx)
        cand = idx[(p.bond[idx] < 0) & (age >= 16) & (age <= 50)]
        if len(cand) == 0:
            return
        for v in self.villages:
            c = cand[p.village[cand] == v.id]
            women = rng.permutation(c[p.female[c]])
            men = rng.permutation(c[~p.female[c]])
            k = min(len(women), len(men))
            if k == 0:
                continue
            wv, mv = women[:k], men[:k]
            kin = ((p.mother[wv] == p.mother[mv]) & (p.mother[wv] >= 0)) | ((p.father[wv] == p.father[mv]) & (p.father[wv] >= 0)) | (p.mother[wv] == mv) | (p.father[mv] == wv)
            compat = np.exp(-2.0 * np.abs(p.share[wv] - p.share[mv]) - 2.0 * np.abs(p.martial[wv] - p.martial[mv]))
            chance = 0.85 if initial else 0.025 * (0.5 + p.gene(wv, SOCIAL_G))
            ok = (~kin) & (rng.random(k) < chance * compat)
            for a, b in zip(wv[ok], mv[ok]):
                p.bond[a], p.bond[b] = b, a
                if not initial:
                    self.counters["bonds"] += 1
                    self.log("bond", f"{self.name(a)} and {person_name(int(b))} became partners", (int(a), int(b)), p.rich[a] >= 0 or p.rich[b] >= 0)
                    if self.minds is not None:
                        self.minds.on_bond(int(a), int(b))
        if not initial:
            self._marry_out(cand)

    def _marry_out(self, cand: np.ndarray) -> None:
        """People marry across settlements, and one of the pair moves to the other's household.

        This is the only way a band's membership mixes, and without it no settlement can ever hold
        more than one band however close together they live — which is what the coarse-graining
        measurements showed before this existed. Who moves is decided by which store is fuller, not
        by a rule about which sex moves.
        """
        p, rng = self.p, self.rng
        if len(self.villages) < 2 or len(cand) == 0:
            return
        free = cand[p.bond[cand] < 0]
        women, men = free[p.female[free]], free[~p.female[free]]
        if len(women) == 0 or len(men) == 0:
            return
        counts = np.bincount(p.village[self.living()], minlength=len(self.villages)).clip(1)
        per_head = np.array([v.food for v in self.villages]) / counts
        for w in rng.permutation(women)[:8]:
            others = men[p.village[men] != p.village[w]]
            if len(others) == 0:
                continue
            # you have to have met: only people whose settlements count each other as people, and
            # who are within a day's walk, marry out
            reach = others[self.world.distance(p.y[others], p.x[others], p.y[w], p.x[w]) < 26]
            reach = reach[p.alpha_p[reach, int(p.village[w])] > 0.55]
            if len(reach) == 0:
                continue
            mate = int(reach[rng.integers(len(reach))])
            compat = float(np.exp(-2.0 * abs(p.share[w] - p.share[mate]) - 2.0 * abs(p.martial[w] - p.martial[mate])))
            if rng.random() > 0.05 * compat:
                continue
            p.bond[w], p.bond[mate] = mate, w
            mover, stayer = (int(w), mate) if per_head[int(p.village[mate])] >= per_head[int(p.village[w])] else (mate, int(w))
            old = self.villages[int(p.village[mover])]
            new = self.village_of(stayer)
            p.village[mover] = new.id
            p.alpha_p[mover, new.id] = 1.0
            p.alpha_p[mover, old.id] = max(0.85, float(p.alpha_p[mover, old.id]))
            p.ty[mover] = (new.cy + rng.normal(0, 2)) % self.world.size
            p.tx[mover] = (new.cx + rng.normal(0, 2)) % self.world.size
            self.counters["bonds"] += 1
            self.counters["married_out"] = self.counters.get("married_out", 0) + 1
            self.log("bond", f"{person_name(int(w))} of {old.name if mover == int(w) else new.name} and "
                             f"{person_name(mate)} married across settlements; {person_name(mover)} moved to {new.name}",
                     (int(w), mate))
            if self.minds is not None:
                self.minds.on_bond(int(w), mate)

    def _births(self) -> None:
        p, rng = self.p, self.rng
        idx = self.living()
        age = self.age(idx)
        mothers = idx[p.female[idx] & (p.bond[idx] >= 0) & (age >= 16) & (age <= 40) & (p.energy[idx] > 0.55) & (self.day - p.last_birth[idx] > int(1.4 * DAYS_PER_YEAR))]
        if len(mothers) == 0:
            return
        mothers = mothers[rng.random(len(mothers)) < 0.42 / DAYS_PER_YEAR * (0.5 + p.gene(mothers, FERTILITY)) * p.energy[mothers]]
        if len(mothers) == 0:
            return
        fathers = p.bond[mothers]
        kids = p.add(len(mothers))
        if self.economy is not None:
            self.economy.grow(p.cap)
            for kid, mother in zip(kids, mothers):
                self.economy.inherit(int(kid), int(mother))
        p.alive[kids] = True
        p.born[kids] = self.day
        p.village[kids] = p.village[mothers]
        p.lineage[kids] = p.lineage[mothers]  # you are of your mother's line wherever you are fed
        p.female[kids] = rng.random(len(kids)) < 0.5
        p.mother[kids], p.father[kids] = mothers, fathers
        p.y[kids], p.x[kids] = p.y[mothers], p.x[mothers]
        p.ty[kids], p.tx[kids] = p.y[mothers], p.x[mothers]
        p.energy[kids], p.health[kids], p.warmth[kids] = 0.8, 0.85, 0.8
        pick = rng.random((len(kids), 16)) < 0.5
        g = np.where(pick, p.genome[mothers], p.genome[fathers]).astype(np.int16)
        mut = rng.random((len(kids), 16)) < 0.03
        g = np.where(mut, g + rng.normal(0, 25, (len(kids), 16)).astype(np.int16), g)
        p.genome[kids] = np.clip(g, 0, 255).astype(np.uint8)
        p.share[kids] = np.clip((p.share[mothers] + p.share[fathers]) / 2 + rng.normal(0, 0.05, len(kids)), 0, 1)
        p.martial[kids] = np.clip((p.martial[mothers] + p.martial[fathers]) / 2 + rng.normal(0, 0.05, len(kids)), 0, 1)
        p.alpha_p[kids] = (p.alpha_p[mothers] + p.alpha_p[fathers]) / 2
        p.bond[kids], p.rich[kids], p.died[kids] = -1, -1, -1
        p.last_birth[kids] = -999
        p.last_birth[mothers] = self.day
        p.children[mothers] += 1
        p.children[fathers] += 1
        p.viability[kids] = 0.8
        p.skill_forage[kids] = 0.1 + 0.2 * p.gene(kids, SKILL)
        p.skill_build[kids] = 0.05 + 0.2 * p.gene(kids, SKILL)
        p.role[kids] = rng.integers(0, 6, len(kids))
        self.counters["births"] += len(kids)
        for m, f, kid in zip(mothers, fathers, kids):
            notable = p.rich[m] >= 0 or p.rich[f] >= 0
            self.log("birth", f"{person_name(int(kid))} was born to {person_name(int(m))} and {person_name(int(f))} in {self.village_of(kid).name}", (int(m), int(f), int(kid)), notable)
            if self.minds is not None and notable:
                self.minds.on_birth(int(kid), int(m), int(f))

    def _culture_and_affect(self, idx: np.ndarray) -> None:
        p, rng, wx = self.p, self.rng, self.weather
        vil = p.village[idx]
        k = len(self.villages)
        counts = np.bincount(vil, minlength=k).clip(1)
        coupling = 0.2 + 0.8 * p.gene(idx, COUPLING)
        gain = 0.3 + 0.7 * p.gene(idx, GAIN)
        safety = np.ones(len(idx), np.float32)
        for r in self.raids:
            safety[vil == r.target] = 0.5
        gloom = np.clip(wx.at(p.y[idx], p.x[idx], wx.cloud) - 0.6, 0, 1)
        viab = np.minimum.reduce([p.energy[idx], p.health[idx], safety, p.warmth[idx]]) * 0.5 + (p.energy[idx] + p.health[idx]) * 0.25
        delta = viab - p.viability[idx]
        p.valence[idx] = np.clip(0.8 * p.valence[idx] + gain * delta * 4 + 0.05 * (viab - 0.6) - 0.02 * gloom, -1, 1)
        p.arousal[idx] = np.clip(0.9 * p.arousal[idx] + np.abs(delta) * 3 * gain, 0, 1)
        p.viability[idx] = viab
        vmean = np.bincount(vil, weights=p.valence[idx], minlength=k) / counts
        p.valence[idx] += (0.05 * coupling * (vmean[vil] - p.valence[idx])).astype(np.float32)
        p.ms[idx] *= 0.97
        toll = np.array([min(0.1, 0.02 * self.deaths_today[v.id]) for v in self.villages])
        p.ms[idx] = np.clip(p.ms[idx] + toll[vil], 0, 1)
        self.deaths_today[:] = 0
        attach = p.bond[idx] >= 0
        shrine = np.array([min(1, v.buildings.get(SHRINE, 0) + v.monuments) for v in self.villages])[vil]
        p.burden[idx] = np.clip(0.985 * p.burden[idx] + 0.02 * p.ms[idx] * (1.2 - p.viability[idx]) - np.where(attach, 0.002, 0.0) - 0.002 * shrine, 0, 1)
        lead_s = np.array([p.share[v.leader] if v.leader >= 0 else 0.5 for v in self.villages])
        lead_m = np.array([p.martial[v.leader] if v.leader >= 0 else 0.3 for v in self.villages])
        law_v = np.array([v.law_share for v in self.villages])
        famine = (np.bincount(vil, weights=p.energy[idx], minlength=k) / counts) < 0.4
        commons = np.where(famine[vil], -0.002, 0.0)
        trust = np.bincount(vil, weights=p.share[idx], minlength=k) / counts
        mmean = np.bincount(vil, weights=p.martial[idx], minlength=k) / counts
        model = np.empty(len(idx), np.int64)
        for v in self.villages:
            sel = np.flatnonzero(vil == v.id)
            if len(sel):
                model[sel] = idx[sel[rng.integers(0, len(sel), len(sel))]]

        def payoff(who):
            t, mm = trust[p.village[who]], mmean[p.village[who]]
            return (p.energy[who] + p.health[who] + p.share[who] * (0.3 * (1 - t) - 0.3 * t) - 0.35 * (1 - p.share[who]) * law_v[p.village[who]]
                    + p.martial[who] * (0.25 * (1 - mm) - 0.45 * mm) + 0.05 * np.minimum(p.raids[who], 4))

        pull = (0.02 * coupling * np.clip(payoff(model) - payoff(idx), 0, 1)).astype(np.float32)
        p.share[idx] = np.clip(p.share[idx] + pull * (p.share[model] - p.share[idx]) + 0.0004 * (lead_s[vil] - p.share[idx]) + commons + rng.normal(0, 0.006, len(idx)), 0, 1)
        threatened = safety < 1
        p.martial[idx] = np.clip(p.martial[idx] + pull * (p.martial[model] - p.martial[idx]) + 0.0004 * (lead_m[vil] - p.martial[idx])
                                 + 0.001 * (0.25 + 0.4 * p.gene(idx, TEMPER) - p.martial[idx])
                                 + np.where(threatened, 0.01 * p.burden[idx], 0.0) + np.where(famine[vil], 0.001, 0.0) + rng.normal(0, 0.006, len(idx)), 0, 1)
        food_pc = np.array([v.food for v in self.villages]) / counts
        for a_ in range(k):
            if food_pc[a_] < 2.0:
                sel = idx[vil == a_]
                for b_ in range(k):
                    if b_ != a_ and food_pc[b_] > 6.0:
                        p.alpha_p[sel, b_] -= (0.004 * (0.5 + p.martial[sel])).astype(np.float32)
        for v in self.villages:
            if v.leader >= 0 and p.alive[v.leader]:
                sel = vil == v.id
                for j in range(k):
                    if j != v.id:
                        p.alpha_p[idx[sel], j] += (0.003 * coupling[sel] * (p.alpha_p[v.leader, j] - p.alpha_p[idx[sel], j])).astype(np.float32)
        np.clip(p.alpha_p, 0, 1, out=p.alpha_p)
        if self.day % 3 == 0:
            self._migrate(idx, vil, counts, food_pc, k)
        va, ar, bu = p.valence[idx], p.arousal[idx], p.burden[idx]
        motif = np.zeros(len(idx), np.int8)
        motif[(va > 0.15) & (ar < 0.5)] = 1
        motif[(va < -0.15) & (ar > 0.45)] = 4
        motif[(va < -0.2) & (p.ms[idx] > 0.4)] = 6
        motif[(va < -0.3) & (ar < 0.3)] = 9
        motif[p.task[idx] == RAID] = 5
        motif[(bu > 0.6) & (va < 0)] = 7
        attached = (p.bond[idx] >= 0) & (va > 0.05) & (ar < 0.35)
        motif[attached & (motif == 1)] = 10
        rich = p.rich[idx] >= 0
        p.motif[idx[~rich]] = motif[~rich]

    def _migrate(self, idx, vil, counts, food_pc, k) -> None:
        p = self.p
        for a_ in range(k):
            if food_pc[a_] >= 1.0:
                continue
            starving = idx[(vil == a_) & (p.energy[idx] < 0.3) & (p.rich[idx] < 0)]
            hosts = [b_ for b_ in range(k) if b_ != a_ and food_pc[b_] > 5.0 and counts[b_] > 5]
            if len(starving) == 0 or not hosts:
                continue
            goers = starving[self.rng.random(len(starving)) < 0.15]
            for b_ in sorted(hosts, key=lambda b_: -food_pc[b_]):
                ok = goers[p.alpha_p[goers, b_] > 0.5]
                if len(ok) == 0:
                    continue
                hv = self.villages[b_]
                p.village[ok] = b_
                p.alpha_p[ok, b_], p.alpha_p[ok, a_] = 1.0, 0.9
                p.ty[ok] = (hv.cy + self.rng.normal(0, 3, len(ok))) % self.world.size
                p.tx[ok] = (hv.cx + self.rng.normal(0, 3, len(ok))) % self.world.size
                self.counters["migrants"] += len(ok)
                if len(ok) >= 20:
                    self.log("migration", f"{len(ok)} hungry people left {self.villages[a_].name} for {hv.name}", tuple(int(i) for i in ok[:2]))
                goers = np.setdiff1d(goers, ok)

    def _trade(self) -> None:
        p = self.p
        idx = self.living()
        vil = p.village[idx]
        k = len(self.villages)
        counts = np.bincount(vil, minlength=k).clip(1)
        for a in range(k):
            for b in range(a + 1, k):
                va, vb = self.villages[a], self.villages[b]
                if self.day - max(va.last_raid_day, vb.last_raid_day) < 36 or not (vil == a).any() or not (vil == b).any():
                    continue
                if min(float(p.alpha_p[idx[vil == a], b].mean()), float(p.alpha_p[idx[vil == b], a].mean())) < 0.45:
                    continue
                mrs_a = (va.food / counts[a] + 1) / (va.wood / counts[a] + 1)
                mrs_b = (vb.food / counts[b] + 1) / (vb.wood / counts[b] + 1)
                if abs(math.log(mrs_a / mrs_b)) < 0.2:
                    continue
                price = math.sqrt(mrs_a * mrs_b)
                seller_food, buyer = (va, vb) if mrs_a > mrs_b else (vb, va)
                wood_q = min(0.08 * buyer.wood, 0.08 * seller_food.food / price)
                if wood_q < 1.0:
                    continue
                food_q = wood_q * price
                seller_food.food -= food_q
                buyer.food += food_q
                buyer.wood -= wood_q
                seller_food.wood += wood_q
                self.counters["trade_volume"] += food_q
                self.prices.append(price)
                both = idx[(vil == a) | (vil == b)]
                other = np.where(p.village[both] == a, b, a)
                p.alpha_p[both, other] = np.clip(p.alpha_p[both, other] + 0.01, 0, 1)
                p.martial[both] = np.clip(p.martial[both] - 0.002, 0, 1)
                if wood_q > 25:
                    self.log("trade", f"{seller_food.name} traded {food_q:.0f} food for {wood_q:.0f} wood with {buyer.name} at {price:.2f} food per wood")

    def _institutions(self) -> None:
        p = self.p
        idx = self.living()
        age = self.age(idx)
        for v in self.villages:
            m = idx[(p.village[idx] == v.id) & (age >= 18)]
            if len(m) == 0:
                continue
            standing = (0.02 * self.age(m) + 0.05 * np.log1p(p.wealth[m]) + 0.1 * np.minimum(p.raids[m], 4) + 0.3 * p.share[m]
                        + 0.1 * p.children[m] + 0.3 * (p.rich[m] >= 0) - 0.02 * np.maximum(self.age(m) - 60, 0) ** 1.5)
            prestige = standing + PRESTIGE_NOISE * self.rng.random(len(m))
            # An incumbent is not unseated by noise. How much a challenger must be ahead by is
            # *derived* from how noisy the prestige measure is, not chosen: the noise term is
            # uniform on [0, PRESTIGE_NOISE), whose standard deviation is PRESTIGE_NOISE/sqrt(12),
            # so a challenger has to lead by more than two of those to be a real challenger rather
            # than a lucky draw. Before this, leaders changed five times in two years.
            leader = int(m[prestige.argmax()])  # with no incumbent, luck settles a near-tie
            if v.leader >= 0 and p.alive[v.leader] and (m == v.leader).any():
                # displacement is judged on standing with the luck taken out, and the challenger
                # has to lead by more than the luck could have explained
                margin = 2.0 * PRESTIGE_NOISE / math.sqrt(12.0)
                if float(standing.max()) - float(standing[m == v.leader][0]) <= margin:
                    leader = int(v.leader)
                else:
                    leader = int(m[standing.argmax()])
            if leader != v.leader:
                old = v.leader
                v.leader = leader
                p.role[leader] = 7
                self.log("leader", f"{self.name(leader)} now leads {v.name}" + (f", after {person_name(old)}" if old >= 0 else ""), (leader,))
            adherence = float((p.share[m] >= 0.5).mean())
            v.adherence_hist = (v.adherence_hist + [adherence])[-3:]
            if not v.law_share and len(v.adherence_hist) >= 2 and min(v.adherence_hist[-2:]) > 0.62:
                v.law_share = True
                self.counters["laws"] += 1
                self.log("law", f"{v.name} made sharing the law; hoarding is punished ({adherence:.0%} already shared)", (leader,))
            elif v.law_share and len(v.adherence_hist) >= 2 and max(v.adherence_hist[-2:]) < 0.4:
                v.law_share = False
                self.log("law", f"{v.name} repealed its sharing law ({adherence:.0%} still shared)", (leader,))
            if v.law_share:
                hoarders = m[(p.wealth[m] > 12) & (p.share[m] < 0.4)]
                for h in hoarders:
                    fine = float(p.wealth[h]) * 0.5
                    p.wealth[h] -= fine
                    v.food += fine
                    self.counters["sanctions"] += 1
                    p.ms[h] = min(1.0, p.ms[h] + 0.1)
                    p.share[h] = min(1.0, p.share[h] + 0.05)
                    if self.minds is not None:
                        self.minds.on_sanction(int(h), v)
                if len(hoarders):
                    self.log("sanction", f"{v.name} fined {len(hoarders)} hoarders under the sharing law", tuple(int(h) for h in hoarders[:3]), notable=len(hoarders) >= 3)
            burden = float(p.burden[m].mean())
            if v.stone >= 40 and burden > 0.08:
                v.stone -= 40
                self.ledger["stone_used"] += 40
                v.monuments += 1
                self.counters["monuments"] += 1
                p.burden[m] *= 0.75
                self.log("monument", f"{v.name} raised monument #{v.monuments}; the sense of burden eased", (leader,))
            if not v.plan:
                pop = int((p.village[idx] == v.id).sum())
                need = []
                if v.buildings.get(HOUSE, 0) * 12 < pop:
                    need += [HOUSE, HOUSE]
                if v.buildings.get(FIELD, 0) * 22 < pop:
                    need += [FIELD] * 6
                if self.day - v.last_raid_day < 120 and v.buildings.get(PALISADE, 0) < 6:
                    need += [PALISADE] * 3
                if self.day - v.last_raid_day < 200 and v.buildings.get(WATCHTOWER, 0) < 1:
                    need += [WATCHTOWER]
                if v.buildings.get(GRANARY, 0) < 1 + pop // 400:
                    need += [GRANARY]
                if v.buildings.get(WELL, 0) < 1 + pop // 500:
                    need += [WELL]
                if need:
                    self._queue_named(v, need)

    def _fission(self) -> None:
        """A group living out at the edge of its commons' range founds a store of its own.

        This is what makes a settlement something other than a renamed founding band. It is keyed on
        the *commons' own members*, not on the coarse-graining, and that is deliberate: a dispersed
        population is still one connected lump of people, so a settlement that splits its granary in
        two stays one settlement — which is precisely the multi-commons place we were unable to make
        before. The conditions are the ones that split a village: enough people to keep a store, far
        enough out to be walking past each other's fields, and either a crowded store behind them or
        a grievance against whoever runs it.
        """
        p = self.p
        idx = self.living()
        if len(idx) == 0:
            return
        size = self.world.size
        for home in list(self.villages):
            members = idx[p.village[idx] == home.id]
            if len(members) < 40:
                continue
            out = members[self.world.distance(p.y[members], p.x[members], home.cy, home.cx) > 7.0]
            if len(out) < 12:
                continue
            # are the outliers together, or just scattered? Take the densest knot of them.
            cy = _circular_mean_1d(p.y[out], size)
            cx = _circular_mean_1d(p.x[out], size)
            knot = out[self.world.distance(p.y[out], p.x[out], cy, cx) < 6.0]
            if len(knot) < 12 or len(knot) > 0.6 * len(members):
                continue
            crowded = self.land_crowding(home, len(members)) > 1.0
            grievance = 0.0
            if self.minds is not None and home.leader >= 0:
                gr = [m.relation(int(home.leader))["grudge"] for i in knot if (m := self.minds.minds.get(int(i)))]
                grievance = float(np.mean(gr)) if gr else 0.0
            if not crowded and grievance < 0.3:
                continue
            if not p.widen(len(self.villages) + 1):
                # the only hard bound left, and the world says so rather than quietly stopping
                if not self.counters.get("fission_refused"):
                    self.log("fission", f"{len(knot)} people of {home.name} wanted their own store, but the world "
                                        f"cannot hold more than {COMMONS_DTYPE_LIMIT} of them; no further bands can form")
                    if self.minds is not None:
                        for i in knot[:20]:
                            m = self.minds.minds.get(int(i))
                            if m is not None:
                                self.minds._remember(m, "fission", "we wanted our own store and could not have one", 0.6, ())
                self.counters["fission_refused"] = self.counters.get("fission_refused", 0) + 1
                return
            share = len(knot) / len(members)
            new = Village(id=len(self.villages), name=self._fission_name(home.name),
                          cy=int(round(cy)) % size, cx=int(round(cx)) % size)
            new.food, new.wood = home.food * share, home.wood * share
            new.stone, new.herd = home.stone * share, home.herd * share
            home.food -= new.food
            home.wood -= new.wood
            home.stone -= new.stone
            home.herd -= new.herd
            self.villages.append(new)
            self.deaths_today = np.append(self.deaths_today, 0)
            if self.economy is not None:
                self.economy.add_market(new.id)
            p.village[knot] = new.id
            p.alpha_p[knot, new.id] = 1.0
            p.alpha_p[knot, home.id] = np.maximum(0.6, p.alpha_p[knot, home.id])
            p.alpha_p[:, new.id] = np.maximum(p.alpha_p[:, new.id], 0.55)
            self._pick_sites(new)
            self._queue_named(new, [GRANARY, HEARTH, WELL, HOUSE, HOUSE, FIELD, FIELD, FIELD])
            self.counters["fissions"] = self.counters.get("fissions", 0) + 1
            why = "the store could not stretch" if crowded else f"they fell out with {person_name(int(home.leader))}"
            self.log("fission", f"{len(knot)} people of {home.name} opened their own store at {new.name} "
                                f"({why}); they took {new.food:.0f} food and {new.herd:.0f} head with them",
                     tuple(int(g) for g in knot[:3]))
            if self.minds is not None:
                for i in knot[:40]:
                    m = self.minds.minds.get(int(i))
                    if m is not None:
                        self.minds._remember(m, "fission", f"we opened our own store at {new.name}", 0.8, ())
            return  # one split per season is plenty

    def _fission_name(self, parent: str) -> str:
        taken = {v.name for v in self.villages}
        for suffix in ("stead", "combe", "wick", "ford", "holt", "mere", "fell", "gard"):
            name = parent[: max(3, len(parent) // 2)] + suffix
            if name not in taken:
                return name
        return f"{parent} {len(self.villages)}"

    def _language_change(self) -> None:
        """A generation of drift: sound change in each dialect, and borrowing between trading neighbours."""
        from . import language as lang

        p = self.p
        idx = self.living()
        # a dialect belongs to a line of descent, and borrows from the lines its speakers live among
        for li in range(len(self.dialects)):
            here = idx[p.lineage[idx] == li]
            if len(here) == 0:
                continue
            neighbours = [int(o) for o in np.unique(p.lineage[idx]) if int(o) != li]
            partner = None
            for o in neighbours:
                shared = here[np.isin(p.village[here], np.unique(p.village[idx[p.lineage[idx] == o]]))]
                if len(shared) > 0 or float(p.alpha_p[here, min(o, self.p.alpha_p.shape[1] - 1)].mean()) > 0.6:
                    partner = o
                    break
            borrow = self.dialects[partner] if partner is not None and self.rng.random() < 0.5 else None
            changed = lang.drift(self.dialects[li], self.rng, borrow_from=borrow, rate=0.06)
            if changed and self.rng.random() < 0.3:
                name = next((v.name for v in self.villages if v.id == li), f"line {li}")
                self.log("language", f"the {name} line now says “{self.dialects[li].say('food')}” for food"
                                     + (", borrowed from the people they live among" if borrow is not None else ""))

    def _raid_decisions(self) -> None:
        import tensacode as tc

        p = self.p
        idx = self.living()
        vil = p.village[idx]
        age = self.age(idx)
        k = len(self.villages)
        counts = np.bincount(vil, minlength=k).clip(1)
        food_pc = np.array([v.food for v in self.villages]) / counts
        moonlight = self.sky.moonlight()
        for v in self.villages:
            if v.leader < 0 or not p.alive[v.leader] or any(r.attacker == v.id for r in self.raids) or self.day - v.last_raid_day < 48:
                continue
            fighters = idx[(vil == v.id) & (age >= 16) & (age <= 50) & (p.health[idx] > 0.5) & (p.martial[idx] > 0.35)]
            options = [("stay", -1)] + [("raid", t.id) for t in self.villages if t.id != v.id and counts[t.id] > 5]
            leader = v.leader
            need = max(0.0, 6 - food_pc[v.id]) / 6
            mart = float(p.martial[idx[vil == v.id]].mean()) if (vil == v.id).any() else 0.0

            def utility(opt, _g, v=v, need=need, mart=mart, fighters=fighters):
                if opt[0] == "stay":
                    return 0.0
                t = opt[1]
                defenders = int(((vil == t) & (age >= 16) & (age <= 60) & (p.martial[idx] > 0.15)).sum())
                walls = self.villages[t].buildings.get(PALISADE, 0) * 0.08 + self.villages[t].buildings.get(WATCHTOWER, 0) * 0.15
                prize = min(food_pc[t], 20) / 20
                ratio = len(fighters) / (defenders + 1)
                return 2.5 * need * prize * mart * min(ratio * 2, 2.0) + 0.3 * float(p.martial[leader]) + 0.15 * (1 - moonlight) - 0.25 - walls

            harm = tc.Constraint("harm_forbidden_by_ascription", lambda opt, _g, leader=leader: opt[0] == "stay" or float(p.alpha_p[leader, opt[1]]) * 0.8 <= 0.42)
            choice = tc.choose(options, objective=tc.Objective("village_interest", "food security against risk and walls", utility), constraints=(harm,))
            if self.minds is not None:
                self.minds.on_leader_decision(leader, v, options, choice, need, mart)
            if isinstance(choice, tc.Unknown) or choice[0] == "stay" or len(fighters) < 3:
                continue
            target = choice[1]
            warriors = fighters[np.argsort(-p.martial[fighters])[: max(3, int(len(fighters) * 0.6))]]
            tv = self.villages[target]
            p.task[warriors] = RAID
            p.ty[warriors] = (tv.cy + self.rng.normal(0, 1.5, len(warriors))) % self.world.size
            p.tx[warriors] = (tv.cx + self.rng.normal(0, 1.5, len(warriors))) % self.world.size
            self.raids.append(Raid(v.id, target, warriors, self.day, moonlight < 0.3))
            v.last_raid_day = self.day
            self.log("raid_start", f"{self.name(leader)} led {len(warriors)} warriors of {v.name} against {tv.name}" + (" under a dark moon" if moonlight < 0.3 else ""), (leader,))

    def _resolve_raids(self) -> None:
        p, rng, w = self.p, self.rng, self.world
        done = []
        for r in self.raids:
            war = r.warriors[p.alive[r.warriors]]
            tv, av = self.villages[r.target], self.villages[r.attacker]
            if len(war) == 0:
                done.append(r)
                continue
            if float(w.distance(p.y[war], p.x[war], tv.cy, tv.cx).mean()) > 3.0 and self.day - r.started < 12:
                continue
            idx = self.living()
            age = self.age(idx)
            defenders = idx[(p.village[idx] == r.target) & (age >= 16) & (age <= 60) & (p.health[idx] > 0.4) & (w.distance(p.y[idx], p.x[idx], tv.cy, tv.cx) < 12)]
            A = len(war) * float((p.martial[war] + 0.5 + p.skill_fight[war]).mean())
            D = (len(defenders) * 0.35 * float((p.martial[defenders] + 0.5 + p.skill_fight[defenders]).mean()) * (1.2 + 0.08 * tv.buildings.get(PALISADE, 0))) if len(defenders) else 0.0
            att_dead = war[rng.random(len(war)) < 0.3 * D / (A + D + 1e-9)]
            def_dead = defenders[rng.random(len(defenders)) < 0.12 * A / (A + D + 1e-9)] if len(defenders) else np.array([], int)
            for i in list(att_dead) + list(def_dead):
                self.kill(int(i), "raid")
            survivors = war[p.alive[war]]
            success = A > 0.8 * D
            loot = min(tv.food, 5.0 * len(survivors)) if success else 0.0
            tv.food -= loot
            av.food += loot
            p.raids[survivors] += 1
            p.skill_fight[survivors] = np.minimum(1.0, p.skill_fight[survivors] + 0.05)
            self.counters["raids"] += 1
            av.raids_led += 1
            if success:
                ys, xs = np.mgrid[-3:4, -3:4]
                yy, xx = (tv.cy + ys.ravel()) % w.size, (tv.cx + xs.ravel()) % w.size
                w.burn[yy, xx] = np.minimum(1.0, w.burn[yy, xx] + 0.5)
                self.dirty_tiles.update(zip(yy.tolist(), xx.tolist()))
            victims = idx[(p.village[idx] == r.target) & p.alive[idx]]
            p.alpha_p[victims, r.attacker] = np.clip(p.alpha_p[victims, r.attacker] - 0.25 * (0.3 + 0.7 * p.gene(victims, GAIN)), 0, 1)
            p.ms[victims] = np.clip(p.ms[victims] + 0.35, 0, 1)
            p.arousal[victims] = np.clip(p.arousal[victims] + 0.5, 0, 1)
            p.alpha_p[survivors, r.target] = np.clip(p.alpha_p[survivors, r.target] - 0.05, 0, 1)
            tv.last_raid_day = self.day
            p.task[survivors] = REST
            p.ty[survivors] = (av.cy + rng.normal(0, 2, len(survivors))) % w.size
            p.tx[survivors] = (av.cx + rng.normal(0, 2, len(survivors))) % w.size
            self.log("raid", f"Raiders of {av.name} {'took ' + str(int(loot)) + ' food' if success else 'were driven off'} at {tv.name}: {len(att_dead)} raiders and {len(def_dead)} defenders died",
                     tuple(int(i) for i in list(att_dead[:2]) + list(def_dead[:2])))
            if self.minds is not None:
                self.minds.on_raid(r, success, loot, att_dead, def_dead)
            done.append(r)
        self.raids = [r for r in self.raids if r not in done]

    def _record(self) -> None:
        if self.day % 4:
            return
        p = self.p
        idx = self.living()
        counts = np.bincount(p.village[idx], minlength=len(self.villages))
        self.series.append({
            "day": self.day, "pop": counts.tolist(), "food": [round(v.food, 1) for v in self.villages], "wood": [round(v.wood, 1) for v in self.villages],
            "valence": round(float(p.valence[idx].mean()), 3) if len(idx) else 0,
            "martial": round(float((p.martial[idx] > 0.5).mean()), 3) if len(idx) else 0,
            "communal": round(float((p.share[idx] > 0.5).mean()), 3) if len(idx) else 0,
            "rain": round(float(self.weather.rain.mean()), 4), "temp": round(float(self.base_temp().mean() + self.weather.anomaly.mean()), 2),
            "price": round(self.prices[-1], 3) if self.prices else None,
            "births": self.counters["births"], "deaths": self.counters["deaths"], "raids": self.counters["raids"], "trade": round(self.counters["trade_volume"], 1),
            **self._economy_series(idx),
        })
        del self.series[:-2000]

    def _economy_series(self, idx: np.ndarray) -> dict:
        e = self.economy
        if e is None or len(idx) == 0:
            return {}
        from .economy import STONE as G_STONE, TOOLS as G_TOOLS, WOOD as G_WOOD

        worth = e.net_worth(idx)
        return {
            "p_wood": [e.price(v.id, G_WOOD) for v in self.villages],
            "p_stone": [e.price(v.id, G_STONE) for v in self.villages],
            "p_tools": [e.price(v.id, G_TOOLS) for v in self.villages],
            "gini": round(e.gini(worth), 3),
            "worth": round(float(np.median(worth)), 2),
            "deprived": round(float((self.p.energy[idx] < 0.35).mean()), 3),
            "tools_pc": round(float(e.stock[idx, G_TOOLS].mean()), 3),
            "trades": e.counters["trades"], "failed_trades": e.counters["failed_no_medium"],
            "money": e.money, "money_share": round(float(e.settled.max() / e.settled.sum()) if e.settled.sum() else 0.0, 3),
            "debt": round(sum(d[2] for d in e.debts if not d[4]), 1), "defaults": e.counters["defaults"],
        }
