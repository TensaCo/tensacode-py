"""Focal people: one tensacode mind each, at the grain of thoughts, memories and conversations.

Each waking phase of the day, for each focal person:

    perceive     needs, weather, sky, buildings, who is nearby -> a snapshot Fragment (scope:percept)
    remember     events since last time arrive as knowledge claims; salient ones become episodes
    think        appraisal rules derive appraisals, wants and meanings, each with provenance;
                 rules that cross modules are gated by coupling κ
    read affect  valence, arousal, integration, effective rank, counterfactual weight, self-salience
    recall       the most salient relevant episode is recalled and cited in the decision
    choose       tc.choose over intentions under constraints; the decision is a claim with premises
    speak        in the evening, whoever is near talks: claims are realized as English, heard,
                 parsed back (lossily), and integrated with "X told me" provenance
    consolidate  repeated episodes become semantic beliefs (trust, grudge, fear); the rest fade

Everything is templates and rules; no language model anywhere. The affect readings and the
"integration" proxy are structural measures, not evidence that anything is experienced.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

import tensacode as tc
from tensacode.cognition import Fragment, Rule, Thought, explain, integrate, think
from tensacode.records import Evidence, Patch, Tell

from . import language as lang
from . import talk
from .sim import (BUILD, CARE, COUPLING, FARM, FORAGE, GAIN, GUARD, MOTIFS, PHASES, QUARRY, RAID, REST, RITUAL, SKILL, SLEEP, SOCIAL, TEMPER,
                  WOOD, person_name)
from .world import DAYS_PER_YEAR, GRANARY, HEARTH, SEASONS, SHRINE, WELL

V = tc.Var
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
PERCEPT = tc.Ref("scope:percept")
MODULE = {
    "hunger": "perception", "health": "perception", "stores": "perception", "season": "perception", "near": "perception", "burden": "perception",
    "law": "perception", "wealth": "perception", "weather": "perception", "time": "perception", "sees": "perception", "cold": "perception",
    "dead": "event", "raided": "event", "born_to": "event", "fined": "event", "fought": "event", "told_me": "event", "sky": "event",
    "kin_of": "social", "bonded_to": "social", "member_of": "self", "adheres": "self", "name": "self", "role": "self", "promoted": "self",
    "trusts": "social", "grudge_against": "social", "grateful_to": "social", "models": "social",
    "appraises": "appraisal", "motif": "appraisal", "wants": "plan", "means": "narrative", "decided": "plan", "recalls": "memory",
    "remembers": "memory", "believes_hostile": "belief", "knows_how": "skill", "keeps_feast": "belief",
}
PROTOTYPES = {
    "calm": (0.05, 0.15, 0.3, 0.3, 0.1, 0.3, 0.4), "joy": (0.5, 0.45, 0.5, 0.7, 0.2, 0.3, 0.5), "flow": (0.25, 0.35, 0.6, 0.3, 0.1, 0.1, 0.8),
    "desire": (0.1, 0.5, 0.4, 0.3, 0.7, 0.4, 0.6), "fear": (-0.5, 0.75, 0.4, 0.3, 0.7, 0.5, 0.4), "anger": (-0.45, 0.85, 0.3, 0.2, 0.5, 0.5, 0.8),
    "grief": (-0.55, 0.35, 0.4, 0.2, 0.8, 0.6, 0.3), "shame": (-0.4, 0.5, 0.4, 0.3, 0.3, 0.9, 0.8), "boredom": (-0.1, 0.05, 0.1, 0.15, 0.05, 0.6, 0.2),
    "suffering": (-0.7, 0.4, 0.2, 0.1, 0.3, 0.6, 0.3), "attachment": (0.45, 0.3, 0.6, 0.5, 0.3, 0.2, 0.5), "awe": (0.4, 0.8, 0.7, 0.8, 0.3, 0.05, 0.3),
}
EASY = {("fear", "anger"): 0.1, ("anger", "fear"): 0.6, ("attachment", "grief"): 0.05, ("grief", "attachment"): 0.8, ("joy", "calm"): 0.05, ("calm", "boredom"): 0.1}
READINGS = ("valence", "arousal", "integration", "effective_rank", "counterfactual", "self_attention", "self_causal")


@dataclass(frozen=True)
class Intention:
    kind: str
    target: int
    u: float
    premises: tuple
    why: str

    def __repr__(self) -> str:
        return f"{self.kind}({self.target})"


@dataclass
class Episode:
    day: int
    tag: str  # "loss" | "raid" | "birth" | "bond" | "gift" | "insult" | "sanction" | "omen" | "find" | "talk"
    text: str
    salience: float
    who: tuple
    claim_id: str | None = None
    last_recalled: int = -999

    def decay(self, day: int) -> float:
        return self.salience * math.exp(-(day - self.day) / 240)


@dataclass
class Mind:
    pid: int
    store: tc.Store
    me: tc.Ref
    born_day: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    affect: dict = field(default_factory=lambda: {k: 0.0 for k in READINGS})
    motif: str = "calm"
    last_viability: float = 0.8
    decision_id: str | None = None
    decision: str = "(nothing yet)"
    promoted_day: int = 0
    reason: str = ""
    hostility: dict = field(default_factory=dict)
    thoughts: deque = field(default_factory=lambda: deque(maxlen=40))
    episodes: list = field(default_factory=list)
    relations: dict = field(default_factory=dict)  # other pid -> {"trust","grudge","gratitude","label"}
    tom: dict = field(default_factory=dict)  # other pid -> {"wants","believes","feels","day"}
    said: deque = field(default_factory=lambda: deque(maxlen=24))  # conversation lines
    skills: dict = field(default_factory=dict)  # name -> proficiency learned by teaching
    recalled: str | None = None
    last_person: str | None = None
    last_thing: str | None = None
    said_recently: deque = field(default_factory=lambda: deque(maxlen=8))  # so talk moves on

    def relation(self, other: int) -> dict:
        return self.relations.setdefault(int(other), {"trust": 0.5, "grudge": 0.0, "gratitude": 0.0, "label": "neighbour"})


class Minds:
    """Every living person has a mind. There is no second tier and nothing is thinned.

    The old build kept a focal tier of a few hundred `tc.Store` minds and ran everyone else as rows
    of NumPy, promoting people toward the camera. That was a compromise, and it is gone: `refill`
    now simply makes a mind for anyone alive who lacks one, and a mind dies with its person. The
    population is whatever the machine can carry at that fidelity — set it with `--people`.
    """

    def __init__(self, sim, focal: int = 0) -> None:
        from tensacode.backends.builtin import UtilityChooser

        self.sim = sim
        self.minds: dict[int, Mind] = {}
        self.runtime = tc.Runtime([UtilityChooser()])
        self.think_ms = 0.0
        self.talk_ms = 0.0
        self.thinks = 0
        self.talks = 0
        self.claims_live = 0
        self.watch: int | None = None  # the person the viewer is inspecting
        self.refill()

    @property
    def focal(self) -> int:
        """Kept for the viewer and the measurements: at full fidelity it is the whole population."""
        return len(self.minds)

    # ------------------------------------------------------------ one mind per person

    def refill(self) -> None:
        """Give a mind to anyone alive who does not have one. Nobody is left as a row of numbers."""
        sim, p = self.sim, self.sim.p
        idx = sim.living()
        for pid in idx[p.rich[idx] < 0]:
            self._promote(int(pid), "born into the world" if sim.day > 0 else "here at the founding")

    def _promote(self, pid: int, reason: str) -> None:
        sim, p = self.sim, self.sim.p
        me = tc.Ref(f"person:{pid}")
        m = Mind(pid, tc.Store(), me, promoted_day=sim.day, reason=reason, last_viability=float(p.viability[pid]))
        m.store.declare("motif", functional=True)
        v = sim.village_of(pid)
        claims = [(me, "name", person_name(pid)), (me, "member_of", v.name), (me, "promoted", reason),
                  (me, "adheres", "communal" if p.share[pid] >= 0.5 else "possessive"),
                  (me, "adheres", "martial" if p.martial[pid] >= 0.5 else "peaceable"),
                  (me, "role", ("gatherer", "farmer", "woodcutter", "mason", "guard", "trader", "healer", "leader")[int(p.role[pid])])]
        for k in (p.mother[pid], p.father[pid]):
            if k >= 0:
                claims.append((me, "kin_of", tc.Ref(f"person:{int(k)}")))
        for c in np.flatnonzero((p.mother[: p.n] == pid) | (p.father[: p.n] == pid)):
            claims.append((me, "kin_of", tc.Ref(f"person:{int(c)}")))
        if p.bond[pid] >= 0:
            claims += [(me, "bonded_to", tc.Ref(f"person:{int(p.bond[pid])}")), (me, "kin_of", tc.Ref(f"person:{int(p.bond[pid])}"))]
        edits = tuple(Tell(tc.Claim(s, pr, o), (Evidence(tc.Ref("record:village-register"), self._when(), method="promotion"),)) for s, pr, o in claims)
        commit = m.store.apply(Patch(edits, m.store.revision))
        for cid in commit.added:
            m.born_day[cid] = sim.day
        for k in (p.mother[pid], p.father[pid], p.bond[pid]):
            if k >= 0:
                m.relation(int(k)).update(trust=0.8, label="kin")
        p.rich[pid] = 1
        self.minds[pid] = m
        sim.counters["promotions"] += 1
        m.thoughts.append({"day": sim.day, "phase": PHASES[sim.phase], "text": f"I am {person_name(pid)} of {v.name}."})

    # ------------------------------------------------------------ events in

    def _when(self) -> datetime:
        return EPOCH + timedelta(days=self.sim.day, hours=[2, 8, 14, 20][self.sim.phase])

    def _remember(self, m: Mind, tag: str, text: str, salience: float, who: tuple = (), claim_id: str | None = None) -> None:
        m.episodes.append(Episode(self.sim.day, tag, text, salience, tuple(int(w) for w in who), claim_id))
        del m.episodes[:-60]

    def on_default(self, m: Mind, debtor: int) -> None:
        """A bad debt is remembered, and becomes a belief about the person that can be spoken."""
        self._remember(m, "debt", f"{person_name(debtor)} never paid back what was owed", 0.75, (int(debtor),))
        claim = tc.Claim(tc.Ref(f"person:{person_name(int(debtor))}"), "trustworthy", "False")
        ev = Evidence(tc.Ref(f"person:{int(debtor)}"), self._when(), locator=f"debt due {self.sim.day}",
                      method="was not repaid", confidence=tc.Score(0.8, "witnessed"))
        commit = m.store.apply(Patch((Tell(claim, (ev,)),), m.store.revision))
        for cid in commit.added:
            m.born_day[cid] = self.sim.day

    def on_death(self, pid: int, kin: list, cause: str) -> None:
        self.minds.pop(pid, None)
        for k in kin:
            if k >= 0 and k in self.minds:
                m = self.minds[k]
                m.pending.append((tc.Claim(tc.Ref(f"person:{pid}"), "dead", True), f"event:death:{pid}"))
                self._remember(m, "loss", f"{person_name(pid)} died of {cause}", 1.0, (pid,))
                m.relation(pid)["label"] = "dead"

    def on_bond(self, a: int, b: int) -> None:
        for x, y in ((a, b), (b, a)):
            if x in self.minds:
                m = self.minds[x]
                m.pending += [(tc.Claim(m.me, "bonded_to", tc.Ref(f"person:{y}")), f"event:bond:{a}-{b}"),
                              (tc.Claim(m.me, "kin_of", tc.Ref(f"person:{y}")), f"event:bond:{a}-{b}")]
                m.relation(y).update(trust=0.9, label="partner")
                self._remember(m, "bond", f"{person_name(y)} and I became partners", 0.9, (y,))

    def on_birth(self, kid: int, mother: int, father: int) -> None:
        for x in (mother, father):
            if x in self.minds:
                m = self.minds[x]
                m.pending += [(tc.Claim(tc.Ref(f"person:{kid}"), "born_to", m.me), f"event:birth:{kid}"),
                              (tc.Claim(m.me, "kin_of", tc.Ref(f"person:{kid}")), f"event:birth:{kid}")]
                m.relation(kid).update(trust=0.95, label="child")
                self._remember(m, "birth", f"our child {person_name(kid)} was born", 0.95, (kid,))

    def on_raid(self, raid, success: bool, loot: float, att_dead, def_dead) -> None:
        sim, p = self.sim, self.sim.p
        attacker, target = sim.villages[raid.attacker], sim.villages[raid.target]
        warriors = {int(w) for w in raid.warriors}
        for m in self.minds.values():
            if p.village[m.pid] == raid.target:
                m.pending.append((tc.Claim(tc.Ref(f"village:{attacker.name}"), "raided", target.name), f"event:raid:{sim.day}"))
                g = 0.3 + 0.7 * float(p.gene([m.pid], GAIN)[0])
                m.hostility[attacker.name] = (g * 1.0 + 1.0 * m.hostility.get(attacker.name, 0.2)) / (g + 1.0)
                self._remember(m, "raid", f"raiders from {attacker.name} came in the {'night' if raid.at_night else 'day'}" + (f" and took {loot:.0f} food" if success else " and were driven off"), 1.0)
            elif m.pid in warriors:
                m.pending.append((tc.Claim(m.me, "fought", target.name), f"event:raid:{sim.day}"))
                self._remember(m, "raid", f"I raided {target.name}" + (" and we took food" if success else " and we were beaten back"), 0.8)

    def on_sanction(self, pid: int, village) -> None:
        m = self.minds.get(pid)
        if m is None:
            return
        m.pending.append((tc.Claim(m.me, "fined", village.name), f"event:fine:{self.sim.day}"))
        self._remember(m, "sanction", f"{village.name} fined me for keeping food back", 0.8)

    def on_leader_decision(self, leader: int, village, options, choice, need: float, mart: float) -> None:
        m = self.minds.get(leader)
        if m is None:
            return
        label = "stay home" if isinstance(choice, tc.Unknown) or choice[0] == "stay" else f"raid {self.sim.villages[choice[1]].name}"
        premises = [r.id for r in m.store.claims(m.me, "appraises")][:4] + [r.id for r in m.store.claims(m.me, "adheres")]
        self._decide(m, ("lead", label), premises, f"as leader of {village.name}: need {need:.2f}, martial mood {mart:.2f}", 0.0)
        m.decision = f"(as leader of {village.name}) {label}"
        m.thoughts.append({"day": self.sim.day, "phase": PHASES[self.sim.phase], "text": f"As leader I must decide: {label}."})

    # ------------------------------------------------------------ the phase

    def tick(self, phase: int) -> None:
        import time as _t

        sim = self.sim
        t0 = _t.perf_counter()
        with tc.use(self.runtime):
            for pid in list(self.minds):
                if not sim.p.alive[pid]:
                    self.minds.pop(pid, None)
                    continue
                if sim.p.asleep[pid] and phase == 0 and (pid + sim.day) % 3:
                    continue  # asleep: no thinking most nights (dreams are not modelled)
                if phase in (1, 3) and (pid + sim.day) % max(1, sim.rich_every) == 0:
                    self.think(self.minds[pid], phase)
            if phase == 3:
                self.converse_round()
        self.runtime.trace.spans.clear()
        self.think_ms += (_t.perf_counter() - t0) * 1e3
        self.claims_live = sum(len(m.store._claims) for m in self.minds.values())

    def think(self, m: Mind, phase: int) -> None:
        sim, p = self.sim, self.sim.p
        pid, me = m.pid, m.me
        v = sim.village_of(pid)
        vref = tc.Ref(f"village:{v.name}")
        coupling = 0.2 + 0.8 * float(p.gene([pid], COUPLING)[0])
        gain = 0.3 + 0.7 * float(p.gene([pid], GAIN)[0])
        counts = max(1, int(((p.village[: p.n] == v.id) & p.alive[: p.n]).sum()))
        food_pc = v.food / counts
        e, h = float(p.energy[pid]), float(p.health[pid])
        rain, snow, temp, cloud = sim.weather_at(p.y[pid], p.x[pid])
        light = float(sim.light_at(p.y[pid], p.x[pid]))
        sky = sim.sky
        percept = [
            (me, "hunger", "starving" if e < 0.25 else "hungry" if e < 0.6 else "fed"),
            (me, "health", "failing" if h < 0.3 else "hurt" if h < 0.6 else "well"),
            (vref, "stores", "empty" if food_pc < 2 else "low" if food_pc < 6 else "ample"),
            (me, "season", SEASONS[sim.season]),
            (me, "time", PHASES[phase]),
            (me, "weather", talk_weather(rain, snow, temp, cloud)),
            (me, "burden", "heavy" if p.burden[pid] > 0.45 else "light"),
        ]
        if temp < 4:
            percept.append((me, "cold", True))
        if v.law_share:
            percept.append((vref, "law", "sharing"))
        if p.wealth[pid] > 12:
            percept.append((me, "wealth", "hoard"))
        for b, label in ((GRANARY, "granary"), (SHRINE, "shrine"), (HEARTH, "hearth"), (WELL, "well")):
            if v.buildings.get(b, 0):
                percept.append((me, "sees", label))
        # who is nearby (kin first, then anyone), on the torus
        near = [int(o) for o in sim.neighbours(pid, 5.0)[:5]]
        for other in near:
            percept.append((tc.Ref(f"person:{other}"), "near", me))
        if light < 0.25:  # it is dark where they are: the moons, and anything unusual
            for moon, ph in zip(("Bel", "Ara"), sky.moon_phase):
                percept.append((tc.Ref(f"moon:{moon}"), "shows", ("new", "waxing", "full", "waning")[int((ph % 1) * 4)]))
            if sky.eclipse or sky.conjunction:
                what = sky.eclipse or ("conjunction of " + " and ".join(sky.conjunction))
                m.pending.append((tc.Claim(tc.Ref("sky:overhead"), "omen", what), f"event:sky:{sim.day}"))
                self._remember(m, "omen", f"the sky showed {what}", 0.85)
                sim.counters["omens"] += 1
        frag = Fragment(tc.Ref(f"obs:{sim.day}.{phase}"), tuple((tc.Claim(s, pr, o, scope=PERCEPT), None) for s, pr, o in percept),
                        snapshot_of=PERCEPT, method="perceive", observed_at=self._when())
        events = [Fragment(tc.Ref(src), ((claim, None),), method="witness", observed_at=self._when()) for claim, src in m.pending]
        m.pending.clear()
        t1 = integrate(m.store, frag, *events)
        t2 = think(m.store, self.rules(m, coupling), since=t1, max_rounds=4)
        for r in t1.added + t2.added:
            m.born_day.setdefault(r.id, sim.day)
        self._read_affect(m, t1, t2, coupling, gain, percept)
        self._recall(m)
        self._choose(m, v, food_pc, near)
        self._inner_speech(m, percept, t2)
        self._consolidate(m)
        self.thinks += 1

    # ------------------------------------------------------------ rules

    def rules(self, m: Mind, coupling: float) -> list:
        me = m.me
        crossing = coupling >= 0.35

        def need_food(b, mind):
            if b["h"] != "fed":
                yield tc.Claim(me, "appraises", ("need", "food")), tc.Score(1.0 if b["h"] == "starving" else 0.6, "appraisal")
                if crossing:
                    yield tc.Claim(me, "wants", ("forage", "")), tc.Score(round(coupling, 2), "coupled")

        def grief(b, mind):
            yield tc.Claim(me, "appraises", ("loss", b["x"].id)), tc.Score(0.9, "appraisal")
            if crossing:
                yield tc.Claim(me, "means", ("mortality", b["x"].id)), tc.Score(round(coupling, 2), "coupled")

        def threat(b, mind):
            yield tc.Claim(me, "appraises", ("threat", b["a"].id)), tc.Score(0.8, "appraisal")
            if crossing:
                yield tc.Claim(me, "wants", ("defend", b["a"].id)), tc.Score(round(coupling, 2), "coupled")

        def retaliate(b, mind):
            if crossing:
                yield tc.Claim(me, "wants", ("retaliate", b["a"].id)), tc.Score(round(coupling, 2), "coupled")

        def attachment(b, mind):
            yield tc.Claim(me, "appraises", ("attachment", b["x"].id)), tc.Score(0.7, "appraisal")

        def injustice(b, mind):
            yield tc.Claim(me, "appraises", ("injustice", b["vv"].id)), tc.Score(0.5, "appraisal")

        def meaning(b, mind):
            if crossing:
                yield tc.Claim(me, "wants", ("meaning", "")), tc.Score(round(coupling, 2), "coupled")

        def legacy(b, mind):
            yield tc.Claim(me, "appraises", ("pride", b["c"].id)), tc.Score(0.8, "appraisal")
            if crossing:
                yield tc.Claim(me, "means", ("legacy", b["c"].id)), tc.Score(round(coupling, 2), "coupled")

        def scarcity_raid(b, mind):
            if crossing:
                yield tc.Claim(me, "wants", ("raid", "")), tc.Score(round(coupling, 2), "coupled")

        def shame(b, mind):
            yield tc.Claim(me, "appraises", ("shame", "hoarding under the law")), tc.Score(0.6, "appraisal")

        def chill(b, mind):
            yield tc.Claim(me, "appraises", ("cold", "night")), tc.Score(0.6, "appraisal")
            if crossing:
                yield tc.Claim(me, "wants", ("firewood", "")), tc.Score(round(coupling, 2), "coupled")

        def gloom(b, mind):
            if b["w"] in ("rain", "downpour", "snow"):
                yield tc.Claim(me, "appraises", ("gloom", b["w"])), tc.Score(0.4, "appraisal")

        def wonder(b, mind):
            yield tc.Claim(me, "appraises", ("wonder", b["o"])), tc.Score(0.7, "appraisal")
            if crossing:
                yield tc.Claim(me, "means", ("omen", b["o"])), tc.Score(round(coupling, 2), "coupled")

        def feast(b, mind):
            yield tc.Claim(me, "wants", ("feast", b["moon"].id)), tc.Score(round(coupling, 2), "coupled")

        def mistrust(b, mind):
            yield tc.Claim(me, "appraises", ("mistrust", b["who"].id)), tc.Score(0.5, "appraisal")

        return [
            Rule("need_food", ((me, "hunger", V("h")),), need_food),
            Rule("grief_for_kin", ((V("x"), "dead", True), (me, "kin_of", V("x"))), grief),
            Rule("threat_from_raiders", ((V("a"), "raided", V("vn")), (me, "member_of", V("vn"))), threat),
            Rule("martial_retaliation", ((V("a"), "raided", V("vn")), (me, "member_of", V("vn")), (me, "adheres", "martial")), retaliate),
            Rule("attachment_to_near_partner", ((V("x"), "near", me), (me, "bonded_to", V("x"))), attachment),
            Rule("injustice_hungry_amid_plenty", ((me, "hunger", "hungry"), (V("vv"), "stores", "ample")), injustice),
            Rule("burden_seeks_meaning", ((me, "burden", "heavy"),), meaning),
            Rule("legacy_in_children", ((V("c"), "born_to", me),), legacy),
            Rule("scarcity_licenses_raid", ((V("vv"), "stores", "empty"), (me, "adheres", "martial")), scarcity_raid),
            Rule("shame_of_hoarding_under_law", ((me, "wealth", "hoard"), (V("vv"), "law", "sharing")), shame),
            Rule("cold_night", ((me, "cold", True),), chill),
            Rule("weather_gloom", ((me, "weather", V("w")),), gloom),
            Rule("wonder_at_the_sky", ((tc.Ref("sky:overhead"), "omen", V("o")),), wonder),
            Rule("keep_the_feast", ((V("moon"), "shows", "full"),), feast),
            Rule("mistrust_the_hoarder", ((V("who"), "hoards", "True"),), mistrust),
        ]

    # ------------------------------------------------------------ affect

    def _read_affect(self, m: Mind, t1: Thought, t2: Thought, coupling: float, gain: float, percept: list) -> None:
        p = self.sim.p
        via = float(p.viability[m.pid])
        signs = {"loss": -0.5, "threat": -0.4, "need": -0.25, "injustice": -0.2, "shame": -0.3, "cold": -0.2, "gloom": -0.15,
                 "attachment": 0.3, "pride": 0.4, "wonder": 0.35, "mistrust": -0.15}
        new_app = [r for r in t2.added if r.claim.predicate == "appraises"]
        live = m.store.claims(m.me, "appraises")
        valence = 3.0 * (via - m.last_viability) + sum(signs.get(r.claim.object[0], 0) for r in new_app) + 0.3 * (via - 0.6)
        m.last_viability = via
        arousal = min(1.0, (len(t1.added) + len(t2.added) + 0.5 * (len(t1.retracted) + len(t2.retracted))) / 20 * (0.5 + gain))

        def module(cid: str) -> str:
            rec = m.store._claims.get(cid)
            return MODULE.get(rec.claim.predicate, "other") if rec else "other"

        # integration and counterfactual weight are read from what the mind currently holds derived,
        # not only from this tick's delta (rules do not re-fire while their percepts are unchanged)
        derived = [r for r in m.store._claims.values() if r.evidence and r.evidence[0].derived_from and r.claim.predicate in ("appraises", "wants", "means", "motif")]
        cross = [r for r in derived if len({module(c) for c in r.evidence[0].derived_from}) >= 2]
        integration = coupling * (len(cross) / len(derived)) if derived else 0.0
        weights: dict = {}
        for r in live:
            conf = r.evidence[0].confidence.value if r.evidence and r.evidence[0].confidence else 0.5
            weights[r.claim.object[0]] = weights.get(r.claim.object[0], 0) + conf
        total = sum(weights.values())
        eff = math.exp(-sum(w / total * math.log(w / total) for w in weights.values())) / 7 if total else 0.0
        future = [r for r in derived if r.claim.predicate in ("wants", "means")]
        reading = {"valence": max(-1, min(1, valence)), "arousal": arousal, "integration": integration, "effective_rank": eff,
                   "counterfactual": len(future) / max(1, len(derived)), "self_attention": sum(1 for s, _, _ in percept if s == m.me) / max(1, len(percept)),
                   "self_causal": m.affect.get("self_causal", 0.4)}
        w = 0.25 + 0.6 * gain
        m.affect = {k: round((1 - w) * m.affect.get(k, 0.0) + w * val, 3) for k, val in reading.items()}
        vec = np.array([m.affect[k] for k in READINGS])
        best, best_cost = m.motif, 1e9
        for name, proto in PROTOTYPES.items():
            cost = float(np.linalg.norm(vec - np.array(proto))) + 0.6 * EASY.get((m.motif, name), 0.0 if name == m.motif else 0.25)
            if cost < best_cost:
                best, best_cost = name, cost
        if best != m.motif or not m.store.claims(m.me, "motif"):
            support = tuple(r.id for r in new_app) or tuple(r.id for r in live[:3])
            old = m.store.claims(m.me, "motif")
            edits = tuple(tc.Retract(r.id, "motif changed") for r in old) + (
                Tell(tc.Claim(m.me, "motif", best), (Evidence(tc.Ref("reading:affect-geometry"), self._when(), method="nearest-prototype@1",
                                                              confidence=tc.Score(round(best_cost, 3), "distance"), derived_from=support),)),)
            commit = m.store.apply(Patch(edits, m.store.revision))
            m.store.forget(commit.retracted)
            for cid in commit.added:
                m.born_day[cid] = self.sim.day
        m.motif = best
        p.motif[m.pid] = MOTIFS.index(best)
        p.valence[m.pid] = m.affect["valence"]
        p.arousal[m.pid] = m.affect["arousal"]

    # ------------------------------------------------------------ memory

    def _recall(self, m: Mind) -> None:
        """Bring the most salient relevant memory to mind; it is cited by the decision that follows."""
        if not m.episodes:
            m.recalled = None
            return
        day = self.sim.day
        tags = {r.claim.object[0] for r in m.store.claims(m.me, "appraises")}
        want = {"threat": "raid", "loss": "loss", "need": "find", "shame": "sanction", "wonder": "omen", "attachment": "bond"}
        prefer = {want[t] for t in tags if t in want}
        best = max(m.episodes, key=lambda e: e.decay(day) * (1.6 if e.tag in prefer else 1.0) * (0.35 if day - e.last_recalled < 8 else 1.0))
        best.last_recalled = day
        if best.decay(day) < 0.05:
            m.recalled = None
            return
        m.recalled = best.text
        claim = tc.Claim(m.me, "recalls", (best.tag, best.text))
        commit = m.store.apply(Patch((Tell(claim, (Evidence(tc.Ref(f"memory:{best.day}"), self._when(), method="recall",
                                                            confidence=tc.Score(round(best.decay(day), 2), "salience")),)),), m.store.revision))
        for cid in commit.added:
            m.born_day[cid] = day
            best.claim_id = cid

    def _consolidate(self, m: Mind) -> None:
        """Repeated episodes about a person become a standing belief; the rest fade and are forgotten."""
        day = self.sim.day
        by_person: dict = {}
        for e in m.episodes:
            for who in e.who:
                by_person.setdefault(who, []).append(e)
        for who, eps in by_person.items():
            hurt = [e for e in eps if e.tag in ("insult", "raid", "sanction")]
            good = [e for e in eps if e.tag in ("gift", "bond", "birth", "comfort")]
            rel = m.relation(who)
            if len(hurt) >= 2 and rel["grudge"] < 0.9:
                rel["grudge"] = min(1.0, rel["grudge"] + 0.3)
                rel["label"] = "rival"
                self._tell_self(m, tc.Claim(m.me, "grudge_against", tc.Ref(f"person:{who}")), tuple(e.claim_id for e in hurt if e.claim_id), "consolidation")
            if len(good) >= 2 and rel["gratitude"] < 0.9:
                rel["gratitude"] = min(1.0, rel["gratitude"] + 0.3)
                rel["label"] = "friend" if rel["label"] == "neighbour" else rel["label"]
                self._tell_self(m, tc.Claim(m.me, "grateful_to", tc.Ref(f"person:{who}")), tuple(e.claim_id for e in good if e.claim_id), "consolidation")
        m.episodes = [e for e in m.episodes if e.decay(day) > 0.04][-40:]
        drop = []
        for cid, rec in list(m.store._claims.items()):
            pred = rec.claim.predicate
            age = day - m.born_day.get(cid, day)
            if pred in ("means", "recalls") and age > 8:
                drop.append(cid)
            elif pred in ("appraises", "wants") and age > 240:
                drop.append(cid)  # appraisals fall away with their percepts; this is only a long backstop
            elif pred in ("dead", "raided", "born_to", "fought", "told_me", "sky", "shows") and age > 120:
                drop.append(cid)
        if drop:
            m.store.forget(drop)
            for cid in drop:
                m.born_day.pop(cid, None)

    def _tell_self(self, m: Mind, claim: tc.Claim, premises: tuple, method: str) -> str | None:
        premises = tuple(c for c in premises if c and c in m.store._claims)
        commit = m.store.apply(Patch((Tell(claim, (Evidence(tc.Ref(f"{method}:{self.sim.day}"), self._when(), method=method, derived_from=premises),)),), m.store.revision))
        for cid in commit.added:
            m.born_day[cid] = self.sim.day
        return commit.added[0] if commit.added else None

    # ------------------------------------------------------------ choosing

    def _choose(self, m: Mind, v, food_pc: float, near: list) -> None:
        sim, p = self.sim, self.sim.p
        pid, me = m.pid, m.me
        wants = {r.claim.object[0]: r for r in m.store.claims(me, "wants")}
        apps: dict = {}
        for r in m.store.claims(me, "appraises"):
            apps.setdefault(r.claim.object[0], []).append(r)
        pc = [r.id for r in m.store.claims(scope=PERCEPT)]
        recall_ids = [r.id for r in m.store.claims(me, "recalls")]
        hunger = 1 - float(p.energy[pid])
        burden = float(p.burden[pid])
        age = float(sim.age([pid])[0])
        skill = float(p.skill_forage[pid])
        light = float(sim.light_at(p.y[pid], p.x[pid]))
        ids = lambda *rs: tuple(r.id for group in rs for r in (group if isinstance(group, list) else [group]) if r is not None)  # noqa: E731
        motif_bias = {"fear": {"rest": 0.2, "guard": 0.2}, "anger": {"volunteer_raid": 0.3}, "grief": {"mourn": 0.4, "care_kin": 0.2},
                      "joy": {"socialize": 0.2}, "attachment": {"care_kin": 0.3, "socialize": 0.1}, "flow": {"farm": 0.25, "gather_wood": 0.2},
                      "boredom": {"socialize": 0.2, "quarry": 0.1}, "shame": {"share": 0.4}, "suffering": {"rest": 0.3}, "awe": {"mourn": 0.2}}.get(m.motif, {})
        response = {}
        if burden > 0.35:
            response = {"volunteer_raid": 0.3 * float(p.martial[pid]) * bool(apps.get("threat")), "mourn": 0.25, "quarry": 0.2,
                        "care_kin": 0.25 * (p.bond[pid] >= 0), "farm": 0.15 * skill}
        if light < 0.22:
            opts = [Intention("sleep", -1, 0.9 + 0.3 * (1 - float(p.energy[pid])), tuple(pc[:1]), "sleep through the dark"),
                    Intention("guard", -1, 0.4 * float(p.martial[pid]) + 0.3 * bool(apps.get("threat")), ids(apps.get("threat", [])) or tuple(pc[:1]), "watch while others sleep")]
        else:
            # What is in the store matters as much as what is in your belly. Without this term a
            # well-fed village gathers firewood until the granary is empty and then starves: the
            # mind was myopic, responding only to its own hunger. A villager can see the store.
            counts_v = max(1, int(((p.village[: p.n] == v.id) & p.alive[: p.n]).sum()))
            # a village works toward a buffer, not toward breaking even: 12 days of food per head
            store_short = min(1.0, max(0.0, (12.0 - food_pc) / 12.0))
            wood_pc = v.wood / counts_v
            wood_plenty = min(1.0, wood_pc / 10.0)
            opts = [
                Intention("farm", -1, 1.1 * hunger + 0.9 * store_short + 0.3 * (v.buildings.get(7, 0) > 0) + (0.4 if "forage" in wants else 0),
                          ids(wants.get("forage"), apps.get("need", [])) or tuple(pc[:2]), "work the fields"),
                Intention("forage", -1, 1.0 * hunger + 0.7 * store_short, ids(wants.get("forage"), apps.get("need", [])) or tuple(pc[:2]), "forage for food"),
                Intention("gather_wood", -1, 0.25 + 0.5 * bool(wants.get("firewood")) + 0.3 * (sim.season >= 2) - 0.45 * wood_plenty,
                          ids(wants.get("firewood")) or tuple(pc[:2]), "bring in firewood"),
                Intention("rest", -1, 0.15 + 0.5 * (float(p.health[pid]) < 0.5) + 0.2 * (age > 55), tuple(pc[:2]), "rest and recover"),
                # sitting with the others is an evening thing; in working hours it competes badly
                Intention("socialize", -1, (0.1 + 0.3 * (len(near) > 0) + 0.2 * (p.bond[pid] < 0 and 16 <= age <= 45))
                          * (1.0 if sim.phase == 3 else 0.45), tuple(pc[:1]), "sit with the others"),
                Intention("build", -1, 0.3 + 0.4 * bool(v.plan) * float(p.skill_build[pid]), tuple(pc[:1]), "work on the building"),
            ]
            if apps.get("loss") or "meaning" in wants or "feast" in wants:
                opts.append(Intention("mourn", -1, 0.5 + 0.3 * float(p.ms[pid]) + 0.2 * bool(wants.get("feast")), ids(apps.get("loss", []), wants.get("meaning"), wants.get("feast")) or tuple(recall_ids[:1]), "keep the rite"))
            if burden > 0.3:
                opts.append(Intention("quarry", -1, 0.25 + 0.3 * burden, ids(wants.get("meaning")) or tuple(pc[:1]), "cut stone for the monument"))
            kin = [int(r.claim.object.id.split(":")[1]) for r in m.store.claims(me, "kin_of")]
            kin = [k for k in kin if k < p.n and p.alive[k]]
            if kin and p.wealth[pid] > 2:
                hungriest = min(kin, key=lambda k: float(p.energy[k]))
                opts.append(Intention("care_kin", hungriest, 0.2 + 0.8 * (1 - float(p.energy[hungriest])) + (0.3 if apps.get("attachment") else 0),
                                      ids(apps.get("attachment", []), apps.get("pride", [])) or tuple(pc[:1]), f"feed {person_name(hungriest)}"))
            if p.wealth[pid] > 6:
                opts.append(Intention("share", -1, 0.1 + 0.6 * float(p.share[pid]) + (0.4 if apps.get("shame") else 0), ids(apps.get("shame", []), apps.get("injustice", [])) or tuple(pc[:1]), "put food in the granary"))
                opts.append(Intention("hoard", -1, 0.2 + 0.5 * (1 - float(p.share[pid])), tuple(pc[:1]), "keep my own stores"))
            for key in ("retaliate", "raid"):
                if key in wants and 16 <= age <= 50:
                    target = int(np.argmax([t.food if t.id != v.id else -1 for t in sim.villages]))
                    opts.append(Intention("volunteer_raid", target, 0.4 * float(p.martial[pid]) + 0.3 * (key == "retaliate"),
                                          ids(wants[key], apps.get("threat", [])) + tuple(recall_ids[:1]), f"march on {sim.villages[target].name}"))
        opts = [Intention(o.kind, o.target, round(o.u + motif_bias.get(o.kind, 0) + response.get(o.kind, 0) + 1e-6 * i, 4), o.premises, o.why) for i, o in enumerate(opts)]

        def harm(o, _g):
            return o.kind != "volunteer_raid" or float(p.alpha_p[pid, o.target]) * 0.8 <= 0.42

        def law(o, _g):
            return o.kind != "hoard" or not v.law_share or float(p.energy[pid]) < 0.25

        def able(o, _g):
            return o.kind not in ("forage", "farm", "gather_wood", "quarry", "build", "volunteer_raid") or float(p.health[pid]) >= 0.25

        choice = tc.choose(opts, objective=tc.Objective("expected_valence", "needs, bonds, meaning and standing, biased by motif, memory and burden", lambda o, g: o.u),
                           constraints=(tc.Constraint("harm_forbidden_by_ascription", harm), tc.Constraint("sharing_law", law), tc.Constraint("able_bodied", able)))
        if isinstance(choice, tc.Unknown):
            m.decision = f"no decision ({choice.reason})"
            return
        self._decide(m, (choice.kind, choice.why), choice.premises, choice.why, choice.u)
        m.decision = f"{choice.kind}: {choice.why}"
        self_prem = sum(1 for cid in choice.premises if (rec := m.store._claims.get(cid)) and rec.claim.subject == me)
        m.affect["self_causal"] = round(self_prem / max(1, len(choice.premises)), 3)
        self._act(m, choice, v)

    def _decide(self, m: Mind, what: tuple, premises, why: str, u: float):
        old = m.store.claims(m.me, "decided")
        premises = tuple(c for c in premises if c in m.store._claims)
        edits = tuple(tc.Retract(r.id, "new decision") for r in old) + (
            Tell(tc.Claim(m.me, "decided", what), (Evidence(tc.Ref("choose:expected_valence"), self._when(), method="tc.choose@utility",
                                                            confidence=tc.Score(round(u, 3), "utility"), derived_from=premises),)),)
        commit = m.store.apply(Patch(edits, m.store.revision))
        m.store.forget(commit.retracted)
        m.decision_id = commit.added[0] if commit.added else m.decision_id
        if m.decision_id:
            m.born_day[m.decision_id] = self.sim.day
        return m.decision_id

    def _act(self, m: Mind, o: Intention, v) -> None:
        sim, p = self.sim, self.sim.p
        pid = m.pid
        if p.task[pid] == RAID:
            return
        task = {"farm": FARM, "forage": FORAGE, "gather_wood": WOOD, "rest": REST, "socialize": SOCIAL, "mourn": RITUAL, "quarry": QUARRY,
                "care_kin": CARE, "share": REST, "hoard": FORAGE, "volunteer_raid": REST, "build": BUILD, "sleep": SLEEP, "guard": GUARD}[o.kind]
        p.task[pid] = task
        p.asleep[pid] = o.kind == "sleep"
        if o.kind == "care_kin" and o.target >= 0 and p.alive[o.target]:
            gift = float(p.wealth[pid]) * 0.3
            p.wealth[pid] -= gift
            p.wealth[o.target] += gift
            p.energy[o.target] = min(1.0, p.energy[o.target] + 0.05)
            p.burden[pid] *= 0.9
            if o.target in self.minds:
                other = self.minds[o.target]
                other.relation(pid)["gratitude"] = min(1.0, other.relation(pid)["gratitude"] + 0.3)
                self._remember(other, "gift", f"{person_name(pid)} shared food with me", 0.7, (pid,))
        elif o.kind == "share":
            gift = float(p.wealth[pid]) * 0.5
            p.wealth[pid] -= gift
            v.food += gift
            p.share[pid] = min(1.0, p.share[pid] + 0.02)
        elif o.kind == "hoard":
            p.share[pid] = max(0.0, p.share[pid] - 0.01)
        elif o.kind == "mourn":
            p.burden[pid] *= 0.85
            p.ms[pid] *= 0.9
            if v.buildings.get(SHRINE, 0) and self.sim.rng.random() < 0.02:
                moon = "Bel" if self.sim.rng.random() < 0.5 else "Ara"
                if moon not in v.calendar:
                    v.calendar[moon] = "full"
                    sim.counters["festivals"] += 1
                    sim.log("festival", f"{v.name} began keeping a feast at the full of {moon}, after {person_name(pid)} kept the rite", (pid,))
        elif o.kind == "volunteer_raid":
            p.martial[pid] = min(1.0, p.martial[pid] + 0.04)
        sim._retarget(np.array([pid]))

    # ------------------------------------------------------------ speech

    def converse_round(self) -> None:
        """Evening: whoever stands near someone else talks. Claims move as sentences, lossily."""
        import time as _t

        t0 = _t.perf_counter()
        sim, p = self.sim, self.sim.p
        ids = [pid for pid in self.minds if p.alive[pid] and not p.asleep[pid]]
        used = set()
        for a in ids:
            if a in used:
                continue
            partner = next((int(b) for b in sim.neighbours(a, 6.0) if int(b) in self.minds and int(b) not in used and int(b) != a), None)
            if partner is None:
                continue
            used.update({a, partner})
            self.converse(a, partner)
        self.talk_ms += (_t.perf_counter() - t0) * 1e3

    def _sayable(self, m: Mind) -> list:
        """Claims this person could put into words now, with a weight for how much they want to."""
        sim, p = self.sim, self.sim.p
        v = sim.village_of(m.pid)
        counts = max(1, int(((p.village[: p.n] == v.id) & p.alive[: p.n]).sum()))
        food_pc = v.food / counts
        out = [((f"settlement:{v.name}", "has_amount", "food:" + ("much" if food_pc > 8 else "little" if food_pc > 1 else "none")), 0.5)]
        for r in m.store.claims(m.me, "recalls"):
            tag, text = r.claim.object
            if tag == "loss":
                out.append(((f"person:{text.split(' ')[0]}", "died", "True"), 0.8))
            elif tag == "raid":
                out.append(((f"settlement:{v.name}", "raided", f"settlement:{v.name}"), 0.7))
            elif tag == "omen":
                what = text.replace("the sky showed ", "").split()[0]  # one word the lexicon knows: "solar", "lunar", "conjunction"
                out.append((("sky:sky", "showed", what if what in ("solar", "lunar", "conjunction") else "conjunction"), 0.75))
        for r in m.store.claims(m.me, "grudge_against"):
            out.append(((f"person:{person_name(int(r.claim.object.id.split(':')[1]))}", "trustworthy", "False"), 0.6))
        for r in m.store.claims(m.me, "grateful_to"):
            out.append(((f"person:{person_name(int(r.claim.object.id.split(':')[1]))}", "trustworthy", "True"), 0.5))
        for r in m.store.claims(predicate="keeps_back"):
            out.append(((r.claim.subject.id, "keeps_back", "good:food"), 0.55))
        rain, snow, temp, cloud = sim.weather_at(p.y[m.pid], p.x[m.pid])
        if rain > 0.1 or snow > 0.02 or temp < 2:
            out.append((("weather:" + ("snow" if snow > 0.02 else "rain" if rain > 0.1 else "frost"), "coming", "True"), 0.45))
        if float(p.energy[m.pid]) < 0.45:
            out.append(((f"person:{person_name(m.pid)}", "hungry", "True"), 0.7))
        econ = getattr(sim, "economy", None)
        if econ is not None:
            # dear or cheap is a comparison, not a threshold: what it costs here against elsewhere
            for good in ("wood", "tools"):
                here = econ.price(v.id, good)
                elsewhere = [econ.price(w.id, good) for w in sim.villages if w.id != v.id]
                if not here or not elsewhere:
                    continue
                other = sum(elsewhere) / len(elsewhere)
                if here > other * 1.15:
                    out.append(((f"good:{good}", "expensive", "True"), 0.45))
                elif here < other * 0.87:
                    out.append(((f"good:{good}", "cheap", "True"), 0.4))
            # debt is something others bring up about you, so what is sayable here is a debt the
            # speaker is owed, not one it announces about itself
            for d in econ.debts[-60:]:
                if d[0] == m.pid and not d[4] and d[2] > 1.0 and p.alive[d[1]]:
                    out.append(((f"person:{person_name(int(d[1]))}", "owes", "good:food"), 0.45))
                    break
        return out

    def _speak(self, speaker: int, listener: int, claim: tuple, *, mood: str = "declare", modality=None,
               negated: bool = False, secondhand: str | None = None) -> tuple:
        """Put a claim into words in the speaker's dialect, then hear it in the listener's."""
        sim, p = self.sim, self.sim.p
        ms, ml = self.minds[speaker], self.minds[listener]
        subject, predicate, obj = claim
        sentence = lang.say(subject, predicate, obj, sim.dialects[int(p.lineage[speaker])],
                            mood=mood, modality=modality, negated=negated, secondhand=secondhand)
        names = [person_name(int(i)) for i in sim.neighbours(speaker, 12.0)[:12]] + [v.name for v in sim.villages] + [person_name(speaker), person_name(listener)]
        ctx = {"listener": person_name(listener), "settlement": sim.village_of(listener).name,
               "last_person": ml.last_person, "last_thing": ml.last_thing}
        best = lang.hear(sentence, sim.dialects[int(p.lineage[listener])], names=names, context=ctx,
                         speaker=person_name(speaker), settlements=[v.name for v in sim.villages])
        sim.counters["utterances"] += 1
        if not best.readings:
            sim.counters["not_understood"] += 1
            return sentence, None, 0.0, "did not understand the words"
        if best.unknown and sim.rng.random() < min(0.8, 0.45 * len(best.unknown)):
            # words the listener's dialect does not have: often the sentence is simply lost. This is
            # what makes dialect divergence cost something rather than being decoration.
            sim.counters["not_understood"] += 1
            return sentence, None, 0.0, "unfamiliar words, meaning lost: " + ", ".join(best.unknown)
        if best.readings > 1:
            sim.counters["ambiguous"] += 1
        trust = ml.relation(speaker)["trust"]
        note = ""
        if best.unknown:
            note = "unfamiliar words: " + ", ".join(best.unknown)
        if best.readings > 1:
            note = (note + "; " if note else "") + f"{best.readings} readings"
        heard_any = None
        for c in best.claims:
            if c["subject"] is None or str(c["object"]).startswith("?"):
                sim.counters["misheard"] += 1
                continue
            conf = round(best.score / 3 * (0.4 + 0.6 * trust), 2)
            if c["hearsay"]:
                conf *= 0.7
            claim_obj = tc.Claim(tc.Ref(c["subject"]), c["predicate"], str(c["object"]))
            src = tc.Ref(f"person:{speaker}")
            commit = ml.store.apply(Patch((Tell(claim_obj, (Evidence(src, self._when(), locator=sentence,
                                                                     method="heard" if not c["hearsay"] else f"heard via {c['via']}",
                                                                     confidence=tc.Score(max(0.05, min(1.0, conf)), "hearsay")),)),), ml.store.revision))
            for cid in commit.added:
                ml.born_day[cid] = sim.day
            sim.counters["claims_transmitted"] += 1
            heard_any = c
            if c["subject"].startswith("person:"):
                ml.last_person = c["subject"]
            else:
                ml.last_thing = c["subject"]
        return sentence, heard_any, best.score, note

    def converse(self, a: int, b: int) -> None:
        """Two people talk. Claims move as sentences: generated in one dialect, parsed in the other."""
        sim, p = self.sim, self.sim.p
        ma, mb = self.minds[a], self.minds[b]
        lines: list = []
        opener = talk.GREETINGS[(sim.day + a) % len(talk.GREETINGS)].format(other=person_name(b))
        lines.append((a, "greet", opener, ""))
        for speaker, listener, ms, ml in ((a, b, ma, mb), (b, a, mb, ma)):
            rel = ms.relation(listener)
            act = self._pick_act(ms, ml, rel, speaker, listener)
            if act in ("insult", "threaten"):
                claim = (f"person:{person_name(listener)}", "trustworthy", "False")
                sentence, heard, score, note = self._speak(speaker, listener, claim, negated=act == "threaten")
                lines.append((speaker, act, sentence, note))
                ml.relation(speaker)["grudge"] = min(1.0, ml.relation(speaker)["grudge"] + 0.35)
                self._remember(ml, "insult", f"{person_name(speaker)} spoke against me", 0.7, (speaker,))
                p.arousal[listener] = min(1.0, float(p.arousal[listener]) + 0.2)
                continue
            if act == "comfort":
                lines.append((speaker, act, f"The loss is heavy, {person_name(listener)}. You are not alone in it.", ""))
                p.ms[listener] = max(0.0, float(p.ms[listener]) - 0.15)
                p.burden[listener] = max(0.0, float(p.burden[listener]) - 0.05)
                ml.relation(speaker)["gratitude"] = min(1.0, ml.relation(speaker)["gratitude"] + 0.2)
                self._remember(ml, "comfort", f"{person_name(speaker)} sat with me in my grief", 0.6, (speaker,))
                continue
            if act == "court":
                sentence, _, _, note = self._speak(speaker, listener, (f"person:{person_name(listener)}", "help", "speaker"), mood="command", modality="should")
                lines.append((speaker, act, sentence, note))
                ml.relation(speaker)["trust"] = min(1.0, ml.relation(speaker)["trust"] + 0.1)
                continue
            if act == "teach":
                sentence, _, _, note = self._speak(speaker, listener, ("skill:farming", "work", "field"), mood="command")
                lines.append((speaker, act, sentence, note))
                p.skill_forage[listener] = min(1.0, float(p.skill_forage[listener]) + 0.03)
                ml.skills["farming"] = round(min(1.0, ml.skills.get("farming", 0.0) + 0.1), 2)
                self._remember(ml, "teach", f"{person_name(speaker)} showed me how to work a field", 0.5, (speaker,))
                continue
            if act == "ask":  # a real question: the other answers from what they believe
                sentence, _, _, note = self._speak(speaker, listener, (f"settlement:{sim.village_of(listener).name}", "has_amount", "food:much"), mood="ask")
                lines.append((speaker, "ask", sentence, note))
                reply = next((c for c, _w in self._sayable(ml) if c[1] == "has_amount"), None)
                if reply:
                    rs, rheard, _, rnote = self._speak(listener, speaker, reply)
                    lines.append((listener, "answer", rs, rnote))
                continue
            if act == "request":
                sentence, _, _, note = self._speak(speaker, listener, (f"person:{person_name(listener)}", "gives", "good:food"), mood="command", modality="should")
                lines.append((speaker, act, sentence, note))
                if float(p.wealth[listener]) > 4 and ml.relation(speaker)["trust"] > 0.4:
                    give = float(p.wealth[listener]) * 0.25
                    p.wealth[listener] -= give
                    p.wealth[speaker] += give
                    lines.append((listener, "offer", "Take this much, and remember it.", ""))
                    ms.relation(listener)["gratitude"] = min(1.0, ms.relation(listener)["gratitude"] + 0.3)
                    self._remember(ms, "gift", f"{person_name(listener)} gave me food when I asked", 0.7, (listener,))
                    econ = getattr(sim, "economy", None)
                    if econ is not None:
                        econ.note_debt(listener, speaker, give)  # a gift asked for in hard times is owed back
                else:
                    lines.append((listener, "refuse", "I have little enough myself.", ""))
                    ms.relation(listener)["trust"] = max(0.0, ms.relation(listener)["trust"] - 0.05)
                continue
            # tell / gossip: pick something worth saying, mark it as hearsay if that is how it came
            sayable = self._sayable(ms)
            if not sayable:
                continue
            # what you bring up is drawn by salience, not always the single most salient thing —
            # otherwise every conversation in the village is about the same omen for weeks
            weights = np.array([cw[1] * (1.4 if act == "gossip" and cw[0][1] in ("trustworthy", "keeps_back", "owes") else 1.0)
                                * (0.35 if cw[0] in ms.said_recently else 1.0) for cw in sayable], dtype=float)
            weights = np.maximum(weights, 1e-6) ** 2.5
            claim = sayable[int(sim.rng.choice(len(sayable), p=weights / weights.sum()))][0]
            ms.said_recently.append(claim)
            secondhand = None
            for r in ms.store.claims(subject=tc.Ref(claim[0]), predicate=claim[1]):
                ev = r.evidence[-1] if r.evidence else None
                if ev and ev.method and ev.method.startswith("heard"):
                    secondhand = _label(ev.source)
                    break
            lie = (act == "tell" and claim[1] == "has_amount" and float(p.wealth[speaker]) > 10 and float(p.share[speaker]) < 0.35)
            if lie:
                good, _, amount = str(claim[2]).partition(":")
                claim = (claim[0], claim[1], f"{good}:{'none' if amount == 'much' else 'much'}")
            sentence, heard, score, note = self._speak(speaker, listener, claim, secondhand=secondhand)
            lines.append((speaker, "lie" if lie else act, sentence, note))
            if heard is None:
                lines.append((listener, "confused", "I don't take your meaning.", ""))
                continue
            ml.tom[speaker] = {"believes": f"{heard['subject'].split(':')[-1]} {heard['predicate']} {heard['object']}",
                               "feels": ms.motif, "wants": ms.decision.split(":")[0], "day": sim.day}
            self._remember(ml, "talk", f"“{sentence}” (from {person_name(speaker)})", 0.35 + 0.2 * score, (speaker,))
        lines.append((a, "farewell", talk.FAREWELLS[(sim.day + b) % len(talk.FAREWELLS)], ""))
        for who, act, text, note in lines:
            for mind in (self.minds[a], self.minds[b]):
                mind.said.append({"day": sim.day, "act": act, "speaker": person_name(who), "text": text, "note": note})
        sim.transcript.append({"day": sim.day, "village": sim.village_of(a).name, "between": [person_name(a), person_name(b)],
                               "ids": [int(a), int(b)],
                               "lines": [{"who": person_name(w), "id": int(w), "act": act, "text": t, "note": note} for w, act, t, note in lines]})
        del sim.transcript[:-200]
        sim.counters["conversations"] += 1
        self.talks += 1

    def _pick_act(self, speaker: Mind, listener: Mind, rel: dict, a: int, b: int) -> str:
        p = self.sim.p
        if rel["grudge"] > 0.5 and p.martial[a] > 0.5:
            return "threaten" if rel["grudge"] > 0.8 else "insult"
        if float(p.ms[b]) > 0.5:
            return "comfort"
        if p.bond[a] < 0 and p.bond[b] < 0 and p.female[a] != p.female[b] and 16 <= float(self.sim.age([a])[0]) <= 45:
            return "court"
        if float(p.energy[a]) < 0.4:
            return "request"
        if float(p.skill_forage[a]) > 0.65 and float(p.skill_forage[b]) < 0.45:
            return "teach"
        if float(p.energy[a]) < 0.7 and self.sim.rng.random() < 0.3:
            return "ask"
        if speaker.store.claims(speaker.me, "grudge_against") or speaker.store.claims(predicate="hoards"):
            return "gossip"
        return "tell"

    # ------------------------------------------------------------ language of thought -> inner speech

    def _inner_speech(self, m: Mind, percept: list, t2: Thought) -> None:
        sim = self.sim
        bits = []
        want = {r.claim.object[0] for r in m.store.claims(m.me, "wants")}
        apps = [r.claim.object for r in m.store.claims(m.me, "appraises")]
        seen = dict((pr, o) for s, pr, o in percept if s == m.me)
        if "hunger" in seen and seen["hunger"] != "fed":
            bits.append({"starving": "I am starving.", "hungry": "I am hungry."}[seen["hunger"]])
        if seen.get("weather") in ("rain", "downpour", "snow", "frost"):
            bits.append({"rain": "The rain does not stop.", "downpour": "This rain will rot the stores.", "snow": "Snow again.", "frost": "The frost bites."}[seen["weather"]])
        for kind, obj in apps[:3]:
            if kind == "loss":
                bits.append("I keep thinking of the dead.")
            elif kind == "threat":
                bits.append(f"{str(obj).split(':')[-1]} may come again.")
            elif kind == "attachment":
                bits.append(f"{person_name(int(str(obj).split(':')[-1]))} is close by; that steadies me.")
            elif kind == "injustice":
                bits.append("The granary is full and I am hungry.")
            elif kind == "shame":
                bits.append("They know what I keep back.")
            elif kind == "wonder":
                bits.append(f"The sky showed {obj}. It must mean something.")
            elif kind == "cold":
                bits.append("The night will be cold without wood.")
        if m.recalled:
            bits.append(f"I remember: {m.recalled}.")
        if "feast" in want:
            bits.append("The moon is full; we should keep the feast.")
        bits.append(f"I feel {m.motif}.")
        bits.append("So: " + m.decision + ".")
        m.thoughts.append({"day": sim.day, "phase": PHASES[sim.phase], "text": " ".join(bits)})

    # ------------------------------------------------------------ inspector

    def inspect(self, pid: int) -> dict:
        m = self.minds.get(pid)
        if m is None:
            return {}
        day = self.sim.day
        beliefs = []
        for rec in sorted(m.store._claims.values(), key=lambda r: -m.born_day.get(r.id, 0))[:44]:
            ev = rec.evidence[-1] if rec.evidence else None
            obj = rec.claim.object
            obj = obj.id if isinstance(obj, tc.Ref) else obj
            beliefs.append({"claim": f"{_label(rec.claim.subject)} {rec.claim.predicate} {_label(obj)}",
                            "source": _label(ev.source) if ev else "", "method": ev.method if ev else "",
                            "heard": ev.locator if ev and ev.method == "heard" else None,
                            "confidence": round(ev.confidence.value, 2) if ev and ev.confidence else None,
                            "day": m.born_day.get(rec.id), "derived": bool(ev and ev.derived_from)})
        why = explain(m.store, m.decision_id, depth=4) if m.decision_id and m.decision_id in m.store._claims else []
        return {
            "focal": True, "affect": m.affect, "motif": m.motif, "decision": m.decision, "explain": [_pretty(line) for line in why][:30],
            "beliefs": beliefs, "claims": len(m.store._claims), "promoted": m.reason, "hostility": {k: round(v, 2) for k, v in m.hostility.items()},
            "thoughts": list(m.thoughts)[-12:],
            "memories": [{"day": e.day, "tag": e.tag, "text": e.text, "salience": round(e.decay(day), 2)} for e in sorted(m.episodes, key=lambda e: -e.decay(day))[:12]],
            "relations": [{"id": k, "name": person_name(k), "label": r["label"], "trust": round(r["trust"], 2), "grudge": round(r["grudge"], 2), "gratitude": round(r["gratitude"], 2)}
                          for k, r in sorted(m.relations.items(), key=lambda kv: -(kv[1]["trust"] + kv[1]["grudge"] + kv[1]["gratitude"]))[:10]],
            "theory_of_mind": [{"id": k, "name": person_name(k), **{kk: vv for kk, vv in val.items() if kk != "day"}, "as_of_day": val["day"]} for k, val in list(m.tom.items())[-6:]],
            "conversation": list(m.said)[-12:],
            "skills_learned": m.skills,
        }


def talk_weather(rain: float, snow: float, temp: float, cloud: float) -> str:
    from .weather import weather_words

    return weather_words(rain, snow, temp, cloud)


def _label(x) -> str:
    s = str(getattr(x, "id", x))
    if s.startswith("person:"):
        try:
            return person_name(int(s.split(":")[1]))
        except ValueError:
            return s.split(":")[-1]
    return s


def _pretty(line: str) -> str:
    import re

    return re.sub(r"person:(\d+)", lambda mm: person_name(int(mm[1])), line)
