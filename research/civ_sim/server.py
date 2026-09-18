"""Watch the civilization live.

    PYTHONPATH=src:. python -m research.civ_sim.server [--seed 1] [--port 8780] [--no-open]

The simulation runs in a background thread. The page fetches the world once, then small
frames: people (position, appearance, task, motif), only the tiles that changed, the coarse
weather fields, the sky, and new events. Everything is drawn client-side.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import socketserver
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import sky as skymod
from .sim import MOTIFS, PHASE_HOURS, PHASES, RAID, ROLES, TASKS, Simulation, person_name
from .weather import weather_words
from .world import BUILDING_NAMES, DAYS_PER_YEAR, SEASONS, TERRAIN_NAMES

PAGE = Path(__file__).parent / "viewer.html"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


class Runner:
    def __init__(self, seed: int, **kw) -> None:
        self.seed = seed
        self.lock = threading.Lock()
        self.speed = 6.0
        self.paused = False
        self.sim = Simulation(seed=seed, **kw)
        self.frame_no = 0
        self.tick_ms = 0.0
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self) -> None:
        while True:
            if self.paused:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            with self.lock:
                self.sim.step()
                self.frame_no += 1
            self.tick_ms = 0.8 * self.tick_ms + 0.2 * (time.perf_counter() - t0) * 1e3
            if self.speed > 0:
                time.sleep(max(0.0, 1.0 / self.speed - (time.perf_counter() - t0)))

    # ------------------------------------------------------------ payloads

    def world(self) -> dict:
        with self.lock:
            w = self.sim.world
            return {
                "size": w.size, "seed": self.seed,
                "elevation": b64((w.elevation * 255).astype(np.uint8)),
                "terrain": b64(w.terrain),
                "fertility": b64((w.fertility * 255).astype(np.uint8)),
                "crop": b64(self.crop()),
                "building": b64(w.building),
                "progress": b64((w.build_progress * 255).astype(np.uint8)),
                "owner": b64(w.build_owner),
                "ground": b64(np.stack([(w.trample * 255).astype(np.uint8), (w.tilled * 255).astype(np.uint8), (w.burn * 255).astype(np.uint8)], -1)),
                "terrain_names": TERRAIN_NAMES, "building_names": BUILDING_NAMES, "motifs": MOTIFS, "tasks": TASKS, "roles": ROLES,
                "villages": [{"id": v.id, "name": v.name, "y": v.cy, "x": v.cx} for v in self.sim.villages],
                "moons": [m["name"] for m in skymod.MOONS], "planets": [p["name"] for p in skymod.PLANETS],
                "weather_grid": self.sim.weather.n,
            }

    def crop(self) -> np.ndarray:
        """How ripe each tile is: what it holds over what it can hold. Fields visibly ripen and are cut."""
        w = self.sim.world
        cap = np.maximum(w.effective_food_cap(), 1e-9)
        return (np.clip(w.food / cap, 0, 1) * 255).astype(np.uint8)

    def frame(self, since_event: int, since_series: int, since_talk: int, full_ground: bool) -> dict:
        with self.lock:
            s, p, w = self.sim, self.sim.p, self.sim.world
            idx = s.living()
            age = s.age(idx)
            flags = ((p.rich[idx] >= 0).astype(np.uint8) | ((p.task[idx] == RAID).astype(np.uint8) << 1)
                     | (p.asleep[idx].astype(np.uint8) << 2) | (p.female[idx].astype(np.uint8) << 3)
                     | (np.isin(idx, [v.leader for v in s.villages]).astype(np.uint8) << 4) | ((p.bond[idx] >= 0).astype(np.uint8) << 5))
            rec = np.zeros(len(idx), dtype=[("id", "<u4"), ("x", "<u2"), ("y", "<u2"), ("v", "u1"), ("m", "u1"), ("f", "u1"), ("task", "u1"),
                                            ("role", "u1"), ("age", "u1"), ("skin", "u1"), ("hair", "u1"), ("face", "u1"), ("build", "u1"), ("health", "u1")])
            rec["id"], rec["x"], rec["y"] = idx, (p.x[idx] * 16).astype(np.uint16), (p.y[idx] * 16).astype(np.uint16)
            rec["v"], rec["m"], rec["f"] = p.village[idx], p.motif[idx], flags
            rec["task"], rec["role"] = p.task[idx], p.role[idx]
            rec["age"] = np.clip(age, 0, 100).astype(np.uint8)
            for name, locus in (("skin", 8), ("hair", 9), ("face", 10), ("build", 11)):
                rec[name] = p.genome[idx, locus]
            rec["health"] = (np.clip(p.health[idx], 0, 1) * 255).astype(np.uint8)
            counts = np.bincount(p.village[idx], minlength=len(s.villages))
            sky = s.sky
            wx = s.weather
            dirty = sorted(s.dirty_tiles)
            s.dirty_tiles = set()
            out = {
                "day": s.day, "year": s.year, "season": SEASONS[s.season], "phase": PHASES[s.phase], "hour": PHASE_HOURS[s.phase],
                "frame": self.frame_no, "paused": self.paused, "speed": self.speed, "tick_ms": round(self.tick_ms, 1),
                "agents": b64(rec), "agent_bytes": rec.dtype.itemsize,
                "sky": {"sun": sky.sun_longitude, "moons": [{"name": n, "phase": round(ph, 3), "lon": round(lo, 3)} for n, ph, lo in zip([m["name"] for m in skymod.MOONS], sky.moon_phase, sky.moon_longitude)],
                        "planets": [{"name": pl["name"], "lon": round(lo, 3)} for pl, lo in zip(skymod.PLANETS, sky.planet_longitude)],
                        "eclipse": sky.eclipse, "conjunction": list(sky.conjunction), "moonlight": round(sky.moonlight(), 3), "text": sky.describe()},
                "light": b64((skymod.light_field(sky, w.size) * 255).astype(np.uint8)),
                "weather": {"cloud": b64((np.clip(wx.cloud, 0, 1) * 255).astype(np.uint8)), "rain": b64((np.clip(wx.rain * 4, 0, 1) * 255).astype(np.uint8)),
                            "snow": b64((np.clip(wx.snow * 4, 0, 1) * 255).astype(np.uint8)), "events": list(wx.events),
                            "text": weather_words(float(wx.rain.mean()), float(wx.snow.mean()), float(s.base_temp().mean() + wx.anomaly.mean()), float(wx.cloud.mean())),
                            "temperature_c": round(float(s.base_temp().mean() + wx.anomaly.mean()), 1)},
                "counters": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.counters.items()},
                "villages": [{"name": v.name, "pop": int(counts[v.id]), "food": round(v.food), "wood": round(v.wood), "stone": round(v.stone), "law": v.law_share,
                              "herd": round(v.herd, 1), "y": v.cy, "x": v.cx,
                              "leader": person_name(v.leader) if v.leader >= 0 else None, "leader_id": int(v.leader), "monuments": v.monuments,
                              "buildings": {BUILDING_NAMES[b]: c for b, c in sorted(v.buildings.items())}, "calendar": v.calendar,
                              "building_now": BUILDING_NAMES[v.plan[0][0]] if v.plan else None} for v in s.villages],
                "events": s.events[since_event:][-80:], "event_count": len(s.events),
                "series": s.series[since_series:], "series_count": len(s.series),
                "talk": s.transcript[since_talk:][-6:], "talk_count": len(s.transcript),
                "raids": [{"from": r.attacker, "to": r.target, "n": int(len(r.warriors))} for r in s.raids],
                "settlements": s.settlements.report(), "settlement_day": s.settlements.day,
                "settlement_events": s.settlements.events[-12:],
                "economy": self.economy(),
                "focal": len(s.minds.minds) if s.minds else 0, "claims": s.minds.claims_live if s.minds else 0,
                "talks": s.minds.talks if s.minds else 0,
                "timing": {k: (round(s.timing[k] / max(1, s.timing["ticks"]), 2) if k != "ticks" else v) for k, v in s.timing.items()},
                "dirty": [[int(y), int(x), int(w.building[y, x]), int(w.build_progress[y, x] * 255), int(w.build_owner[y, x]),
                           int(w.trample[y, x] * 255), int(w.tilled[y, x] * 255), int(w.burn[y, x] * 255)] for y, x in dirty[:4000]],
                "model_calls": 0,
            }
            out["crop"] = b64(self.crop())
            if full_ground:
                out["ground"] = b64(np.stack([(w.trample * 255).astype(np.uint8), (w.tilled * 255).astype(np.uint8), (w.burn * 255).astype(np.uint8)], -1))
                out["progress"] = b64((w.build_progress * 255).astype(np.uint8))
                out["building"] = b64(w.building)
            return out

    def economy(self) -> dict:
        """What the economy is doing right now: prices per settlement, what settles trades, flows."""
        from .economy import GOODS

        s, e = self.sim, self.sim.economy
        if e is None:
            return {}
        idx = s.living()
        last = e.stats[-1] if e.stats else {}
        total = float(e.settled.sum()) or 1.0
        return {
            "money": e.money, "goods": list(GOODS),
            "settles": {GOODS[g]: round(float(e.settled[g] / total), 3) for g in range(4)},
            "accept": {GOODS[g]: round(float(e.accept[idx, g].mean()), 3) for g in range(4)} if len(idx) else {},
            "price": last.get("price", {}), "volume": last.get("volume", {}), "held": last.get("held", {}),
            "gini": last.get("gini"), "worth": last.get("median_worth"), "deprived": last.get("deprived"),
            "tools_pc": last.get("tools_pc"), "labour": last.get("labour", {}), "routes": last.get("routes", {}),
            "debt": last.get("debt"), "defaults": e.counters["defaults"], "loans": e.counters["loans"],
            "trades": e.counters["trades"], "attempts": e.counters["attempts"],
            "for_use": e.counters["for_use"], "to_pass_on": e.counters["to_pass_on"],
            "failed_no_medium": e.counters["failed_no_medium"], "caravans": e.counters["caravans"],
            "tax": round(e.counters["tax"], 1),
        }

    def agent(self, pid: int) -> dict:
        with self.lock:
            s, p = self.sim, self.sim.p
            if pid < 0 or pid >= p.n:
                return {"error": "no such person"}
            if s.minds is not None:
                s.minds.watch = pid  # watching someone keeps them in the focal tier
            i = [pid]
            kin = lambda arr: [{"id": int(k), "name": person_name(int(k)), "alive": bool(p.alive[k])} for k in arr if k >= 0]  # noqa: E731
            kids = np.flatnonzero((p.mother[: p.n] == pid) | (p.father[: p.n] == pid))
            rain, snow, temp, cloud = s.weather_at(p.y[pid], p.x[pid])
            local_hour = (PHASE_HOURS[s.phase] + (float(p.x[pid]) / s.world.size) * 24) % 24
            out = {
                "id": pid, "name": person_name(pid), "village": s.village_of(pid).name, "alive": bool(p.alive[pid]),
                "age": round(float(s.age(i)[0]), 1), "sex": "female" if p.female[pid] else "male", "task": TASKS[int(p.task[pid])],
                "role": ROLES[int(p.role[pid])], "asleep": bool(p.asleep[pid]),
                "needs": {"energy": round(float(p.energy[pid]), 2), "health": round(float(p.health[pid]), 2), "warmth": round(float(p.warmth[pid]), 2),
                          "wealth": round(float(p.wealth[pid]), 1), "mortality_salience": round(float(p.ms[pid]), 2), "burden": round(float(p.burden[pid]), 2),
                          "sick": round(float(p.sick[pid]), 2)},
                "affect": {"valence": round(float(p.valence[pid]), 2), "arousal": round(float(p.arousal[pid]), 2)},
                "motif": MOTIFS[int(p.motif[pid])],
                "axes": {"gain": round(0.3 + 0.7 * float(p.gene(i, 0)[0]), 2), "coupling": round(0.2 + 0.8 * float(p.gene(i, 1)[0]), 2),
                         "ascription": {v.name: round(float(p.alpha_p[pid, v.id]), 2) for v in s.villages}},
                "ideology": {"communal": round(float(p.share[pid]), 2), "martial": round(float(p.martial[pid]), 2)},
                "skills": {"foraging": round(float(p.skill_forage[pid]), 2), "building": round(float(p.skill_build[pid]), 2), "fighting": round(float(p.skill_fight[pid]), 2)},
                "appearance": {"skin": int(p.genome[pid, 8]), "hair": int(p.genome[pid, 9]), "face": int(p.genome[pid, 10]), "build": int(p.genome[pid, 11]),
                               "eyes": int(p.genome[pid, 12]), "nose": int(p.genome[pid, 13]), "brow": int(p.genome[pid, 14]), "mouth": int(p.genome[pid, 15])},
                "partner": kin([int(p.bond[pid])]), "parents": kin([int(p.mother[pid]), int(p.father[pid])]), "children": kin(kids[:12]),
                "raids": int(p.raids[pid]), "life": [{"day": d, "text": t} for d, t in s.life.get(pid, [])][-30:],
                "leader_of": next((v.name for v in s.villages if v.leader == pid), None),
                "where": {"y": round(float(p.y[pid]), 1), "x": round(float(p.x[pid]), 1), "local_hour": round(local_hour, 1),
                          "weather": weather_words(rain, snow, temp, cloud), "temperature_c": round(temp, 1)},
            }
            e = s.economy
            if e is not None:
                from .economy import GOODS, TOOLS, WOOD

                place = s.settlements.of(pid)
                owed = sum(d[2] for d in e.debts if d[1] == pid and not d[4])
                due = sum(d[2] for d in e.debts if d[0] == pid and not d[4])
                out["holdings"] = {"food": round(float(p.wealth[pid]), 2)} | {
                    GOODS[g]: round(float(e.stock[pid, g]), 2) for g in range(1, 4)}
                out["means"] = {"net_worth": round(float(e.net_worth(np.array([pid]))[0]), 2),
                                "owes": round(owed, 2), "owed_to_them": round(due, 2),
                                "tool_bonus": round(float(e.tool_bonus(np.array([pid]))[0]), 3)}
                out["settlement"] = ({"name": place.name, "kind": place.kind, "pop": place.pop,
                                      "specialization": place.specialization, "gini": place.gini,
                                      "cohesion": place.cohesion, "institutions": place.institutions}
                                     if place is not None else None)
            if s.minds is not None and pid in s.minds.minds:
                out.update(s.minds.inspect(pid))
                out["affect"] = s.minds.minds[pid].affect
            else:
                out["focal"] = False
            return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=72, help="tiles across the torus")
    ap.add_argument("--villages", type=int, default=3, help="founding bands")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--people", type=int, default=240, help="how many people, every one of them a full mind")
    ap.add_argument("--focal", type=int, default=0, help="ignored: every person is a mind now")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    runner = Runner(args.seed, people=args.people, size=args.size, villages=args.villages)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/world":
                    return self._send(200, json.dumps(runner.world()).encode(), "application/json")
                if u.path == "/frame":
                    body = runner.frame(int(q.get("events", 0)), int(q.get("series", 0)), int(q.get("talk", 0)), q.get("ground") == "1")
                    return self._send(200, json.dumps(body).encode(), "application/json")
                if u.path == "/agent":
                    return self._send(200, json.dumps(runner.agent(int(q.get("id", -1)))).encode(), "application/json")
            except Exception as exc:  # noqa: BLE001 - a viewer request must never kill the server
                return self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if "paused" in body:
                runner.paused = bool(body["paused"])
            if "speed" in body:
                runner.speed = float(body["speed"])
            if "watch" in body and runner.sim.minds is not None:
                runner.sim.minds.watch = int(body["watch"]) if body["watch"] is not None else None
            self._send(200, json.dumps({"paused": runner.paused, "speed": runner.speed}).encode(), "application/json")

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server(("127.0.0.1", args.port), Handler)
    print(f"civilization viewer: http://127.0.0.1:{args.port}/  (seed {args.seed}, {args.people} people, every one a full mind, {args.size}x{args.size} torus; no model calls)", flush=True)
    if not args.no_open:
        webbrowser.open(f"http://127.0.0.1:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
