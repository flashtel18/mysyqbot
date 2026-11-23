"""
world.py - the SIMULATED campus day (replaces a real robot + real sensors).

What the ROBOT knows:
  * the floor map and the timetable (which lectures end when)
  * what its CAMERA sees: about 9 cells (~14 m) around it, blocked by walls.
    It cannot see around corners or inside rooms.
What the robot does NOT know: where students actually are. So it
  * keeps a Bayes BELIEF for every hall where a lecture just ended
    (timetable prior, updated by what the camera sees or doesn't see)
  * plans which halls to check with HILL CLIMBING (+ A* travel times)
  * otherwise just PATROLS the corridors
  * whenever the camera spots students (far or near), it goes to them (A*)
  * meets them: HMM guesses the mood, the MDP picks what to say
  * answers questions (Naive Bayes)

The simulator itself knows the hidden TRUTH (who comes out, when they leave,
the real moods) - that's what makes the evaluation honest.

Rules of the world on a break:
  * only ~40% of the students come out into the hall; the rest stay inside
  * if the group "hangs out" they stay ~4 min, otherwise they leave within 1 min
  * curious students near the robot walk up to it; some ask questions
"""
import math
import random

import bayes
import mood_hmm
from astar import astar, bfs
from campus import (Campus, Timetable, SECONDS_PER_CELL, STUDENT_SECONDS_PER_CELL,
                    hhmm_to_sec, sec_to_hhmm)
from robot import MysyqBrain, STUDENT_QUESTIONS
from tour_planner import BREAK_SEC, INTERACT_SEC, LATE_DELAY, OUT_FRACTION, TAU

REACH = 2.6              # robot talks to everyone within this many cells
VISION_R = 9             # camera range in cells (~14 m)
P_SEE_NEAR, P_SEE_FAR = 0.95, 0.6   # chance to notice a visible student (close / far)
SEEN_MEMORY = 3.0        # robot remembers a sighting for 3 s
ZONE_UPDATE_EVERY = 30.0  # repeated looks at the same hall are not independent evidence:
                          # update when it comes into view, when the observation changes,
                          # or at most every 30 s
CURIOUS_RADIUS = 7       # students this close may notice the robot...
CURIOUS_PROB = 0.5       # ...and walk up to it by themselves
QUESTION_PROB = 0.6      # chance that someone asks a question during a visit
TRANSPARENT = set(".~S+X")    # the camera sees along halls, doorways and the stairs - not through walls
STANDABLE = set(".~S+X")      # cells where a person can be seen

STRATEGIES = {
    "mysyq": "MysyqBot: Bayes belief + camera updates + hill-climbing tour + go to anyone it sees",
    "no_camera_bayes": "same, but the belief never uses the camera (timetable prior only)",
    "nearest": "checks the halls of ending lectures nearest-first (no probabilities)",
    "wander": "random patrol, goes to students only when it happens to see them",
}

GROUP_COLORS = {
    "AIN-2601": "#e4572e", "AIN-2602": "#f3a712", "AIN-2603": "#29335c",
    "PAI-2601": "#669bbc", "PAI-2602": "#8e5572", "PAI-2603": "#2a9d8f",
}
OUT_STATES = ("walking_out", "hall", "approach")
ALGOS = ["vision", "bayes", "astar", "hill", "hmm", "mdp", "nb"]


# ----------------------------------------------------------------- truth
def make_truth(timetable, rng):
    """Hidden ground truth for every lecture of the day."""
    truth = {}
    for group in timetable.groups:
        lecs = timetable.of_group(group)
        dl = timetable.next_deadline(group)
        init = mood_hmm.initial_distribution(dl["days"] if dl else None)
        moods, acts = mood_hmm.sample_day(len(lecs), init, rng)
        for lec, mood, act in zip(lecs, moods, acts):
            truth[lec.id] = {"mood": mood, "activity": act}
    for lec in timetable.lectures:
        prior, _ = bayes.prior_present(lec, timetable)
        # reality is close to the robot's prior, but not identical
        p_true = min(0.95, max(0.05, prior + rng.uniform(-0.1, 0.1)))
        present = rng.random() < p_true          # does the group hang out in the hall?
        late = rng.randint(120, 240) if lec.late_teacher else 0
        truth[lec.id].update(present=present, actual_end=lec.end + late)
    return truth


def sample_leave_time(truth_lec, rng):
    """When does one student who came out leave the hall?"""
    if truth_lec["present"]:
        return truth_lec["actual_end"] + rng.expovariate(1 / TAU)
    return truth_lec["actual_end"] + rng.uniform(0, 60)


def sample_break(lec, truth_lec, rng):
    """Which students come out (~40%), and when they leave the hall."""
    return [sample_leave_time(truth_lec, rng) for _ in range(lec.size)
            if rng.random() < OUT_FRACTION]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------- simulation
class Simulation:
    def __init__(self, seed=7, start="10:15", student_questions=True, strategy="mysyq", explain=True):
        assert strategy in STRATEGIES, strategy
        self.seed, self.strategy, self.explain_on = seed, strategy, explain
        self.rng = random.Random(seed)
        self.campus = Campus()
        self.tt = Timetable()
        self.lec_by_id = {l.id: l for l in self.tt.lectures}
        self.brain = MysyqBrain(self.campus, self.tt, random.Random(seed + 1))
        self.truth = make_truth(self.tt, self.rng)
        self.t = float(hhmm_to_sec(start))
        self.student_questions = student_questions

        self.students = []
        self._next_sid = 0
        self.spawned, self.released = set(), set()
        self.handled_breaks = {e for e in self.tt.end_times if e < self.t}

        d = self.campus.dock
        self.robot = {"pos": (float(d[0]), float(d[1])), "cell": d, "path": []}
        self.mode, self.goal, self.goal_label = "patrol", None, ""
        self.tour, self.planned_path, self.guide_path = [], [], []
        self.guide = None                 # {"target":..., "phase": "go"|"wait", "until":...}
        self.interaction = None
        self.interacting = False
        self.belief = {}                  # room -> {lec, prior, p, why, ...}
        self.break_E, self.break_end = None, None
        self.met_ids, self.seen = set(), {}
        self.greeted = set()              # lectures whose group was already greeted this break
        self.visible = frozenset()
        self._vis_cache, self._sense_acc, self._need_replan = {}, 0.0, False
        self._recent_wp = []
        self.bubble, self.bubble_until = None, 0
        self.near_group, self.near_until = None, 0
        self.log_lines = []
        self.chat = []                    # every question + answer (shown in the 3D view)
        self.student_bubble = None        # {"sid", "text", "until"} - the student who is asking
        self.stats = {"breaks": 0, "visits": 0, "empty": 0, "reached": 0,
                      "came_by_themselves": 0, "questions": 0, "spotted": 0}
        self.visits = []          # (lecture, belief P, group students met, all met)
        self.bayes_records = []   # (prior, final belief, truth, n_updates) per hall per break
        self.last_plan = None
        self.visits_at_plan = 0
        self.astar_viz = {"id": 0, "cells": []}
        self.explain = {"phase": "patrol", "thought": "Starting up...", "active": ["vision"],
                        "event": 0, "event_kind": "", "bayes": {"zones": [], "last": None},
                        "astar": None, "hill": None, "hmm": None, "mdp": None, "nb": None,
                        "vision": {"seen": 0, "radius": VISION_R}}
        self.log(f"MysyqBot online at {sec_to_hhmm(self.t)}. Camera range {VISION_R} cells, "
                 f"halls only. Strategy: {STRATEGIES[strategy]}")

    # ------------------------------------------------------------ utils
    def log(self, *lines):
        self.log_lines.extend(lines)

    def say(self, text, seconds):
        self.bubble, self.bubble_until = text, self.t + seconds

    def think(self, phase, thought, active, event=None, **sections):
        if not self.explain_on:
            return
        self.explain.update(phase=phase, thought=thought, active=active, **sections)
        if event:
            self.explain["event"] += 1
            self.explain["event_kind"] = event

    def is_idle(self):
        return self.mode == "patrol" and not self.guide and not self.tour

    def mode_label(self):
        return {"patrol": "patrolling the halls", "tour": "checking halls (break plan)",
                "approach": "going to students it sees", "interact": "talking to students",
                "guide": "guiding a student"}.get(self.mode, self.mode)

    def next_break(self):
        upcoming = [e for e in self.tt.end_times if e not in self.handled_breaks]
        return min(upcoming) if upcoming else None

    def readings_today(self, group, until_t):
        return [self.truth[l.id]["activity"] for l in self.tt.of_group(group) if l.end <= until_t]

    def hall_count(self, room):
        lec_ids = {l.id for l in self.tt.lectures if l.room == room}
        return sum(1 for s in self.students if s["lec"] in lec_ids and s["state"] in OUT_STATES)

    def since_end(self, lec):
        """Seconds since the class is EXPECTED to end (robot's view: late teachers +3 min)."""
        return self.t - (lec.end + (LATE_DELAY if lec.late_teacher else 0))

    # --------------------------------------------------------- stepping
    def step(self, dt):
        while dt > 1e-9:
            h = min(1.0, dt)
            self._tick(h)
            dt -= h

    def skip_to_next_break(self, before=20):
        nb = self.next_break()
        if nb is not None and self.is_idle() and nb - before > self.t:
            self.t = nb - before
            self.log(f"... fast-forward to {sec_to_hhmm(self.t)}")

    def _tick(self, h):
        self.t += h
        self._update_students(h)
        for e in self.tt.end_times:
            if e not in self.handled_breaks and self.t >= e:
                self.handled_breaks.add(e)
                if self.t - e < 300:
                    self._start_break(e)
        if self.break_end and self.t > self.break_end + 120:
            self._close_break()
        self._sense_acc += h
        if self._sense_acc >= 1.0:
            self._sense_acc = 0.0
            self._sense()
            self._decide()
        self._move(h)
        if self.bubble and self.t > self.bubble_until:
            self.bubble = None

    # ---------------------------------------------------------- students
    def _new_student(self, lec, cell):
        self._next_sid += 1
        return {"id": self._next_sid, "lec": lec.id, "group": lec.group, "state": "seated",
                "cell": cell, "path": [], "leave_t": None, "noticed": False,
                "pos": (cell[0] + self.rng.uniform(-.35, .35), cell[1] + self.rng.uniform(-.35, .35))}

    def _update_students(self, h):
        for lec in self.tt.lectures:
            tr = self.truth[lec.id]
            if lec.id not in self.spawned and lec.start <= self.t < tr["actual_end"]:
                self.spawned.add(lec.id)
                cells = self.campus.rooms[lec.room]["cells"]
                for _ in range(lec.size):
                    self.students.append(self._new_student(lec, self.rng.choice(cells)))
            if lec.id in self.spawned and lec.id not in self.released and self.t >= tr["actual_end"]:
                self.released.add(lec.id)
                group = [s for s in self.students if s["lec"] == lec.id]
                times = sample_break(lec, tr, self.rng)   # ~40% come out, with leave times
                self.rng.shuffle(group)
                for i, s in enumerate(group):
                    if i < len(times):                       # comes out to the hall
                        s["state"] = "walking_out"
                        s["leave_t"] = times[i]
                        target = self.rng.choice(self.campus.rooms[lec.room]["gather"])
                        s["path"] = list(self.brain.route(s["cell"], target, halls_only=False)[0])
                    else:                                    # stays in the room
                        s["state"] = "inroom"
                        s["leave_t"] = lec.end + 600

        keep = []
        for s in self.students:
            if self.t > self.truth[s["lec"]]["actual_end"] + 1500:   # long gone
                continue
            st = s["state"]
            if st == "inroom" and self.t >= s["leave_t"]:
                continue                                  # went to the next class
            if st in OUT_STATES and self.t >= s["leave_t"]:
                s["state"] = st = "leaving"
                x = self.rng.random()          # where do they go? stairs, toilets or the sofas
                if x < 0.45:
                    target = self.rng.choice(self.campus.exits)
                elif x < 0.75:
                    wc = self.rng.choice(self.campus.toilets)
                    target = self.rng.choice(self.campus.rooms[wc]["cells"])
                else:
                    target = self.rng.choice(self.campus.lounge or self.campus.exits)
                s["path"] = list(self.brain.route(s["cell"], target, halls_only=False)[0])
            if st in ("walking_out", "approach", "leaving"):
                if self._advance(s, h, STUDENT_SECONDS_PER_CELL, weighted=False):
                    if st == "leaving":
                        continue                          # left the floor -> disappears
                    s["state"] = "hall"
            keep.append(s)
        self.students = keep

    # ------------------------------------------------------------ camera
    def visible_from(self, cell):
        """Cells the camera can see from `cell` (line of sight, cached)."""
        if cell in self._vis_cache:
            return self._vis_cache[cell]
        r0, c0 = cell
        R = VISION_R
        g = self.campus.grid
        out = set()
        for r in range(max(0, r0 - R), min(self.campus.H, r0 + R + 1)):
            for c in range(max(0, c0 - R), min(self.campus.W, c0 + R + 1)):
                if (r - r0) ** 2 + (c - c0) ** 2 > R * R or g[r][c] not in TRANSPARENT:
                    continue
                n = max(abs(r - r0), abs(c - c0)) * 2
                ok = True
                for k in range(1, n):
                    rr = round(r0 + (r - r0) * k / n)
                    cc = round(c0 + (c - c0) * k / n)
                    if g[rr][cc] not in TRANSPARENT:
                        ok = False
                        break
                if ok:
                    out.add((r, c))
        self._vis_cache[cell] = frozenset(out)
        return self._vis_cache[cell]

    def _sense(self):
        """Camera: notice visible students; Bayes-update the belief of halls in view."""
        rc = self.robot["cell"]
        rp = self.robot["pos"]
        self.visible = vis = self.visible_from(rc)
        for s in self.students:
            if s["state"] in OUT_STATES + ("leaving",) and s["cell"] in vis:
                d = dist(s["pos"], rp)
                p = P_SEE_NEAR - (P_SEE_NEAR - P_SEE_FAR) * min(1.0, d / VISION_R)
                if self.rng.random() < p:
                    self.seen[s["id"]] = self.t
        self.seen = {k: v for k, v in self.seen.items() if self.t - v <= SEEN_MEMORY}
        if self.explain_on:
            self.explain["vision"] = {"seen": len(self.seen), "radius": VISION_R}

        if self.strategy != "mysyq":
            return
        seen_lecs = {s["lec"] for s in self.students if s["id"] in self.seen}
        for room, b in self.belief.items():
            since = self.since_end(b["lec"])
            zone = self.campus.rooms[room]["gather"]
            frac = sum(1 for c in zone if c in vis) / max(1, len(zone))
            in_view = frac >= 0.5
            was_in_view, b["in_view"] = b.get("in_view", False), in_view
            if since < 20 or since > BREAK_SEC or not in_view:
                continue                       # first 20 s: students are still coming out
            seen = b["lec"].id in seen_lecs
            new_look = not was_in_view or seen != b.get("last_obs")
            if not new_look and self.t - b["last_update"] < ZONE_UPDATE_EVERY:
                continue
            if seen and b["p"] > 0.97:
                continue                       # already sure, nothing to learn
            b["last_obs"] = seen
            res = bayes.camera_update(b["p"], seen, since)
            before = b["p"]
            b["p"], b["last_update"] = res["posterior"], self.t
            b["updates"] += 1
            if dist(rp, self.campus.spot(room)) <= 3.5 and not seen:
                b["checked"] = True
            if abs(b["p"] - before) > 0.12:
                self._need_replan = True
            self.log(f"[Bayes] camera looks at the {room} hall ({since:.0f}s after class): "
                     f"{res['obs']} -> P(group hangs out) {before:.2f} -> {b['p']:.2f}")
            self.think(self.explain["phase"],
                       f"I looked at the hall of {room}: {res['obs']}. "
                       f"Bayes: P(group hangs out) {before:.2f} -> {b['p']:.2f}.",
                       ["vision", "bayes"],
                       bayes={"zones": self._belief_view(),
                              "last": {"room": room, "group": b["lec"].group, "obs": res["obs"],
                                       "since": round(since), "prior": round(before, 3),
                                       "l_h": round(res["P(obs|H)"], 3), "l_n": round(res["P(obs|not H)"], 3),
                                       "evidence": round(res["evidence"], 3), "post": round(b["p"], 3)}})

    def _belief_view(self):
        out = []
        for room, b in self.belief.items():
            since = self.since_end(b["lec"])
            share = bayes.expected_share(b["p"], since) if since >= 0 else 1.0
            out.append({"room": room, "group": b["lec"].group, "prior": round(b["prior"], 3),
                        "p": round(b["p"], 3), "why": b["why"], "checked": b["checked"],
                        "expect": round(b["lec"].size * OUT_FRACTION * share, 1)})
        return sorted(out, key=lambda z: -z["p"])

    # ------------------------------------------------------------ breaks
    def _start_break(self, e):
        self._close_break()
        ending = self.tt.ending_at(e)
        self.break_E, self.break_end = e, e + BREAK_SEC
        self.met_ids = set()
        self.greeted = set()
        self.belief = {}
        self.stats["breaks"] += 1
        self.log("", f"=== BREAK {sec_to_hhmm(e)}: {len(ending)} lectures end "
                     f"({', '.join(l.room for l in ending)}) ===")
        if self.strategy in ("mysyq", "no_camera_bayes"):
            self.log("[Bayes] priors from the timetable (the robot cannot see the students yet):")
            for lec in ending:
                prior, why = bayes.prior_present(lec, self.tt)
                self.belief[lec.room] = {"lec": lec, "prior": prior, "p": prior, "why": why,
                                         "last_update": -1e9, "updates": 0, "checked": False}
                self.log(f"  {lec.room} {lec.group}: P(group hangs out) = {prior:.2f} ({why})")
            self._replan("break start", event=True)
        elif self.strategy == "nearest":
            self.tour = self._nearest_order([l.room for l in ending])
            self.log(f"[nearest] tour: {' > '.join(self.tour)}")
        self.say("Break! Let's find some students. Meow!", 8)

    def _close_break(self):
        for room, b in self.belief.items():
            self.bayes_records.append((b["prior"], b["p"], self.truth[b["lec"].id]["present"], b["updates"]))
        self.belief, self.tour, self.planned_path = {}, [], []
        self.break_E = self.break_end = None

    def _nearest_order(self, rooms):
        tour, here, t, left = [], self.robot["cell"], 0.0, list(rooms)
        while left:
            k = min(left, key=lambda r: self.brain.travel_sec(here, self.campus.spot(r)))
            t += self.brain.travel_sec(here, self.campus.spot(k)) + INTERACT_SEC
            if t > BREAK_SEC:
                break
            tour.append(k)
            left.remove(k)
            here = self.campus.spot(k)
        return tour

    def _replan(self, why, event=False):
        self._need_replan = False
        if not self.break_end or self.strategy not in ("mysyq", "no_camera_bayes"):
            return
        budget = self.break_end - self.t
        zones = {room: self.brain.make_zone(b["lec"], b["p"], self.t)
                 for room, b in self.belief.items() if not b["checked"] and b["p"] > 0.03}
        if budget < 45 or not zones:
            self.tour, self.planned_path = [], []
            return
        res = self.brain.plan_tour(self.t, self.robot["cell"], zones, budget, why)
        self.tour = list(res["plan"]["tour"])
        self.planned_path = res["path"]
        self.last_plan = {"tour": [self.belief[r]["lec"] for r in self.tour], "log": res["log"]}
        self.visits_at_plan = len(self.visits)
        self.log(*res["log"])
        tour_s = " -> ".join(self.tour) or "nothing worth checking"
        self.think("plan", f"Plan ({why}): Bayes says where students probably are, A* gives the "
                           f"travel times, hill climbing picks the order: {tour_s}.",
                   ["bayes", "astar", "hill"], event="plan" if event else None,
                   hill=res["explain"], bayes={"zones": self._belief_view(),
                                               "last": self.explain["bayes"].get("last")})

    # ------------------------------------------------------------ decide
    def _decide(self):
        if self.interaction or self.guide:
            return
        rp = self.robot["pos"]
        cands = [s for s in self.students if s["id"] in self.seen and s["state"] in OUT_STATES
                 and s["id"] not in self.met_ids and s["lec"] not in self.greeted]
        if cands:
            target = min(cands, key=lambda s: dist(s["pos"], rp))
            d = dist(target["pos"], rp)
            if d <= REACH - 0.4:
                self._start_interaction(self.lec_by_id[target["lec"]])
                return
            goal = target["cell"] if self.campus.is_hall(target["cell"]) else \
                min(self.campus.near_hall_cells(target["cell"], 2) or [target["cell"]],
                    key=lambda c: dist(c, target["pos"]))
            if self.mode != "approach" or self.goal is None or dist(goal, self.goal) > 1.5:
                if self.mode != "approach":
                    self.stats["spotted"] += 1
                    n = len({s["id"] for s in cands})
                    self.log(f"{sec_to_hhmm(self.t, True)} [camera] I see {n} student(s) "
                             f"{d:.0f} cells away ({target['group']}) -> going to them")
                    self.say(f"I see students over there! Meow!", 6)
                self.mode = "approach"
                self._set_goal(goal, f"students of {target['group']}")
                self.think("approach", f"My camera sees {len(cands)} student(s) {d:.0f} cells away. "
                                       f"A* path to them: {len(self.robot['path'])} steps.",
                           ["vision", "astar"])
            return
        if self.mode == "approach":
            self.mode, self.goal = "patrol", None
            self._replan("lost sight of students")
        if self._need_replan:
            self._replan("belief changed")
        if self.tour:
            room = self.tour[0]
            spot = self.campus.spot(room)
            if self.robot["cell"] == spot and not self.robot["path"]:
                self.tour.pop(0)
                self.stats["empty"] += 1
                if room in self.belief:
                    self.belief[room]["checked"] = True
                self.log(f"{sec_to_hhmm(self.t, True)} at the {room} hall: nobody here -> next")
                self.say(f"Nobody near {room}... next hall!", 5)
                self.mode, self.goal = "patrol", None
                self._replan("hall was empty")
                return
            if self.mode != "tour" or self.goal != spot:
                self.mode = "tour"
                b = self.belief.get(room)
                self._set_goal(spot, f"hall of {room}")
                self.think("tour", f"Going to check the hall of {room}"
                                   + (f" - Bayes: {b['p']:.0%} chance the group hangs out there." if b else "."),
                           ["bayes", "astar"])
            return
        # patrol
        if self.mode != "patrol":
            self.mode, self.goal = "patrol", None
        if self.goal is None or (self.robot["cell"] == self.goal and not self.robot["path"]):
            self._pick_waypoint()

    def _pick_waypoint(self):
        wps = [r["spot"] for r in self.campus.rooms.values()] + [self.campus.exit_spot, self.campus.dock]
        weights = []
        for w in wps:
            wt = 1.0
            if w == self.robot["cell"] or w in self._recent_wp:
                wt = 0.0
            weights.append(wt)
        if sum(weights) == 0:
            weights = [1.0] * len(wps)
        goal = self.rng.choices(wps, weights)[0]
        self._recent_wp = (self._recent_wp + [goal])[-4:]
        self._set_goal(goal, "patrol waypoint")
        self.think("patrol", "Nobody in sight. I can't see through walls, so I patrol the halls "
                             "and keep my camera open.", ["vision", "astar"])

    def _set_goal(self, cell, label):
        self.goal, self.goal_label = cell, label
        path, cost, expanded = self.brain.route(self.robot["cell"], cell)
        self.robot["path"] = list(path or [])
        if self.explain_on and path:
            trace = []
            astar(self.campus, self.robot["cell"], cell, True, trace)
            _, bcost, bexp = bfs(self.campus, self.robot["cell"], cell, True)
            self.astar_viz = {"id": self.astar_viz["id"] + 1, "cells": trace}
            self.explain["astar"] = {"goal": label, "cost": cost, "seconds": round(cost * SECONDS_PER_CELL),
                                     "cells": len(path), "expanded": expanded,
                                     "bfs_expanded": bexp, "bfs_cost": bcost}

    # ------------------------------------------------------------ moving
    def _advance(self, ent, dt, sec_per_cell, weighted):
        """Move entity along ent['path']; returns True when path finished."""
        budget = dt
        while budget > 1e-9 and ent["path"]:
            nxt = ent["path"][0]
            pr, pc = ent["pos"]
            d = abs(nxt[0] - pr) + abs(nxt[1] - pc)
            if d < 1e-9:
                ent["path"].pop(0)
                ent["cell"] = nxt
                continue
            spc = sec_per_cell * (self.campus.cost(nxt) if weighted else 1)
            need = d * spc
            if budget >= need:
                ent["pos"] = (float(nxt[0]), float(nxt[1]))
                ent["cell"] = nxt
                ent["path"].pop(0)
                budget -= need
            else:
                f = (budget / spc) / d
                ent["pos"] = (pr + (nxt[0] - pr) * f, pc + (nxt[1] - pc) * f)
                budget = 0
        return not ent["path"]

    def _move(self, h):
        if self.interaction:
            self._update_interaction()
            return
        if self.guide:
            g = self.guide
            if g["phase"] == "go":
                if self._advance(self.robot, h, SECONDS_PER_CELL, weighted=True):
                    g["phase"], g["until"] = "wait", self.t + 12
                    self.say("Here it is! Good luck. Meow!", 12)
            elif self.t >= g["until"]:
                self.guide, self.guide_path = None, []
                self.mode, self.goal = "patrol", None
            return
        self._advance(self.robot, h, SECONDS_PER_CELL, weighted=True)

    # ------------------------------------------------------- interaction
    def _start_interaction(self, lec):
        tr = self.truth[lec.id]
        self.mode = "interact"
        self.robot["path"] = []
        b = self.belief.get(lec.room)
        p = b["p"] if b else float("nan")
        self.interaction = {"lec": lec, "start": self.t, "until": self.t + INTERACT_SEC, "met": set(),
                            "met_group": set(), "came": 0, "p": p, "ask_t": None}
        self.interacting = True
        if self.student_questions and self.rng.random() < QUESTION_PROB:
            self.interaction["ask_t"] = max(self.t, tr["actual_end"]) + self.rng.uniform(40, 80)
        self.log(f"{sec_to_hhmm(self.t, True)} meeting {lec.group} in the hall near {lec.room}")
        res = self.brain.decide_action(lec, self.readings_today(lec.group, lec.end))
        self.log(*res["log"], f"  says: \"{res['message']}\"")
        self.say(res["message"], 40)
        self.near_group, self.near_until = lec.group, self.t + 900
        self.think("interact", f"Talking to {lec.group}. HMM: the group is probably "
                               f"{max(res['belief'], key=res['belief'].get)} "
                               f"({max(res['belief'].values()):.0%}). MDP: best action = {res['action']}.",
                   ["hmm", "mdp"], event="decide",
                   hmm={"group": lec.group, "readings": self.readings_today(lec.group, lec.end),
                        "belief": {k: round(v, 3) for k, v in res["belief"].items()}, "viterbi": res["viterbi"]},
                   mdp={"belief": {k: round(v, 3) for k, v in res["belief"].items()},
                        "scores": {k: round(v, 2) for k, v in res["scores"].items()},
                        "action": res["action"], "policy": mdp_policy(self.brain.Q)})

    def _update_interaction(self):
        it = self.interaction
        rpos = self.robot["pos"]
        for s in self.students:
            if s["state"] not in OUT_STATES:
                continue
            d = dist(s["pos"], rpos)
            if d <= REACH:
                it["met"].add(s["id"])
                if s["lec"] == it["lec"].id:
                    it["met_group"].add(s["id"])
            elif s["state"] == "hall" and not s["noticed"] and d <= CURIOUS_RADIUS:
                s["noticed"] = True
                if self.rng.random() < CURIOUS_PROB:     # walks up to the robot by itself
                    spot = self.rng.choice(self.campus.near_hall_cells(self.robot["cell"], 2))
                    s["path"] = list(self.brain.route(s["cell"], spot, halls_only=True)[0])
                    s["state"] = "approach"
                    it["came"] += 1
        if it["ask_t"] and self.t >= it["ask_t"] and it["met"]:
            it["ask_t"] = None
            q = self.rng.choice(STUDENT_QUESTIONS).format(group=it["lec"].group)
            self.stats["questions"] += 1
            near = [s for s in self.students if s["id"] in it["met"] and s["state"] in OUT_STATES]
            asker = self.rng.choice(near) if near else None
            self.ask(q, who=f"{asker['group'] if asker else it['lec'].group} student",
                     sid=asker["id"] if asker else None)
        if self.t >= it["until"]:
            n_all, n_grp = len(it["met"]), len(it["met_group"])
            new = it["met"] - self.met_ids
            self.met_ids |= it["met"]
            self.greeted.add(it["lec"].id)
            self.stats["visits"] += 1
            self.stats["reached"] += len(new)
            self.stats["came_by_themselves"] += it["came"]
            self.visits.append((it["lec"], it["p"], n_grp, n_all))
            self.log(f"{sec_to_hhmm(self.t, True)} done: met {n_all} students in the hall "
                     f"({n_grp} from {it['lec'].group}, {it['came']} walked up by themselves)")
            b = self.belief.get(it["lec"].room)
            if b:
                b["checked"] = True
            if it["lec"].room in self.tour:
                self.tour.remove(it["lec"].room)
            self.interaction, self.interacting = None, False
            self.mode, self.goal = "patrol", None
            self._replan("after meeting a group")

    # --------------------------------------------------------- questions
    def ask(self, text, who="You", sid=None):
        group = self.near_group if self.t < self.near_until else None
        res = self.brain.answer(text, self.t, self.robot["cell"], group)
        self.log("", f"{sec_to_hhmm(self.t, True)} {who.upper()} ASKS: {text}", *res["log"],
                 f"  answer: {res['reply']}")
        self.chat.append({"n": len(self.chat) + 1, "clock": sec_to_hhmm(self.t, True), "who": who,
                          "sid": sid, "text": text, "intent": res["intent"],
                          "conf": round(res["probs"][0][1], 2), "reply": res["reply"]})
        if sid is not None:
            self.student_bubble = {"sid": sid, "text": text, "until": self.t + 14}
        self.say(res["reply"], 30)
        active = ["nb", "astar"] if res["path"] else ["nb"]
        self.think("answer", f"Question: \"{text}\". Naive Bayes -> {res['intent']} "
                             f"({res['probs'][0][1]:.0%})." + (" A* draws the route." if res["path"] else ""),
                   active, event="answer",
                   nb={"text": text, "words": res["words"], "intent": res["intent"], "reply": res["reply"],
                       "probs": [[c, round(p, 3)] for c, p in res["probs"]]})
        if res["path"]:
            self.guide_path = res["path"]
            if self.explain_on:
                trace = []
                astar(self.campus, self.robot["cell"], res["target"], True, trace)
                self.astar_viz = {"id": self.astar_viz["id"] + 1, "cells": trace}
                self.explain["astar"] = {"goal": "the place the student asked for",
                                         "cost": len(res["path"]) - 1,
                                         "seconds": round((len(res["path"]) - 1) * SECONDS_PER_CELL),
                                         "cells": len(res["path"]), "expanded": len(trace),
                                         "bfs_expanded": bfs(self.campus, self.robot["cell"], res["target"], True)[2],
                                         "bfs_cost": None}
            if self.mode == "patrol" and not self.tour and not self.interaction:
                self.mode = "guide"
                self.robot["path"] = list(res["path"])
                self.guide = {"phase": "go", "until": None}
        return res


def mdp_policy(Q):
    return {s: max(Q[s], key=Q[s].get) for s in Q}


if __name__ == "__main__":
    sim = Simulation(seed=7, start="10:15")
    sim.step(60 * 15)
    print("\n".join(sim.log_lines))
    print(sim.stats)
