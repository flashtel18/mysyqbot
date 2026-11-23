"""
tour_planner.py - ALGORITHM 3: Hill climbing with random restarts
                  (course topic: Week 3, local search).

Problem: several lectures end at the same time, but the break is only
10 minutes. Which rooms should the robot visit, and in what ORDER?

A "tour" = ordered list of rooms, e.g. ["204", "201"].

Score of a tour (expected number of students the robot cheers up):

    score = SUM over rooms i in the tour of
              P(present_i)          <- from Bayes (bayes.py)
            * group_size_i          <- from the timetable
            * OUT_FRACTION          <- only ~40% of students come out to the hall
            * priority_i            <- exam soon? (deadlines.csv)
    arrival_i is computed with A* travel times (astar.py)
    + each visit takes INTERACT_SEC seconds
    + if the teacher usually runs late, the robot waits at the door
      (expected LATE_DELAY seconds) before the students come out
    Tours that do not finish inside the break get a big penalty.

Hill climbing: start from a tour, look at all NEIGHBOUR tours
   (add a room / remove a room / swap two rooms / replace a room),
   move to the best neighbour if it is better, stop when no neighbour is better
   (local optimum). Random restarts reduce the chance of a bad local optimum.

`exhaustive_best` tries every possible tour - used only to CHECK that hill
climbing finds the optimum (fine here because there are at most 5 rooms).
"""
import itertools
import math
import random

BREAK_SEC = 600          # 10-minute break
INTERACT_SEC = 120       # time spent with one group
TAU = 240                # students leave with mean time 4 min after class
PENALTY_PER_SEC = 5      # penalty for every second over the break
LATE_DELAY = 180         # expected extra minutes of a "late" teacher (3 min)
QUICK_LEAVE = 60         # a group that does not hang out is gone within 1 min
OUT_FRACTION = 0.40      # share of a group that comes out to the hall on a break


def priority_from_deadline(days):
    if days is None:
        return 1.0
    if days <= 3:
        return 1.5
    if days <= 7:
        return 1.2
    return 1.0


def evaluate_tour(tour, cands, travel, start, budget=BREAK_SEC):
    """tour: list of candidate keys. cands: key -> dict(p, size, priority).
    travel(a, b) -> seconds, where a/b are 'START' or candidate keys.
    Returns (score, timeline, feasible)."""
    t = 0.0
    here = "START"
    score = 0.0
    timeline = []
    for key in tour:
        t += travel(here, key)
        c = cands[key]
        arrive = t
        delay = c.get("delay", 0)   # seconds from now until the class ends (<0: ended already)
        if t < delay:               # teacher finishes late -> wait at the door
            t = delay
        since_end = t - delay
        linger = math.exp(-since_end / TAU)              # group hangs out
        quick = max(0.0, 1 - since_end / QUICK_LEAVE)    # group leaves fast
        expected_share = c["p"] * linger + (1 - c["p"]) * quick
        value = c["size"] * OUT_FRACTION * c["priority"] * expected_share
        score += value
        timeline.append({"key": key, "arrive": arrive, "start": t, "linger": linger, "value": value})
        t += INTERACT_SEC
        here = key
    overtime = max(0.0, t - budget)
    feasible = overtime == 0
    return score - PENALTY_PER_SEC * overtime, timeline, feasible


def neighbours(tour, keys):
    unused = [k for k in keys if k not in tour]
    out = []
    for i in range(len(tour)):                      # remove one room
        out.append(("remove " + tour[i], tour[:i] + tour[i + 1:]))
    for u in unused:                                # add one room anywhere
        for j in range(len(tour) + 1):
            out.append((f"add {u}@{j}", tour[:j] + [u] + tour[j:]))
    for i in range(len(tour)):                      # swap two rooms
        for j in range(i + 1, len(tour)):
            t = tour[:]
            t[i], t[j] = t[j], t[i]
            out.append((f"swap {tour[i]}<->{tour[j]}", t))
    for i in range(len(tour)):                      # replace a room
        for u in unused:
            t = tour[:]
            t[i] = u
            out.append((f"replace {tour[i]}->{u}", t))
    return out


def hill_climb(start_tour, cands, travel, start, budget=BREAK_SEC):
    keys = list(cands)
    current = list(start_tour)
    cur_score = evaluate_tour(current, cands, travel, start, budget)[0]
    steps = [("start", current[:], cur_score)]
    while True:
        best_move, best_tour, best_score = None, None, cur_score
        for move, t in neighbours(current, keys):
            s = evaluate_tour(t, cands, travel, start, budget)[0]
            if s > best_score + 1e-9:
                best_move, best_tour, best_score = move, t, s
        if best_tour is None:        # no better neighbour -> local optimum
            return current, cur_score, steps
        current, cur_score = best_tour, best_score
        steps.append((best_move, current[:], cur_score))


def plan_tour(cands, travel, start="START", restarts=8, rng=None, budget=BREAK_SEC):
    """Hill climbing with random restarts. Returns dict with the best tour,
    its timeline, and the trace of the first climb (for explanation)."""
    rng = rng or random.Random(0)
    keys = list(cands)
    starts = [[]]  # restart 0: empty tour (robot visits nobody yet)
    for _ in range(restarts - 1):
        k = rng.randint(1, len(keys)) if keys else 0
        starts.append(rng.sample(keys, k))
    best = None
    traces = []
    for st in starts:
        tour, score, steps = hill_climb(st, cands, travel, start, budget)
        traces.append(steps)
        if best is None or score > best[1] + 1e-9:
            best = (tour, score)
    tour, score = best
    _, timeline, feasible = evaluate_tour(tour, cands, travel, start, budget)
    return {"tour": tour, "score": score, "timeline": timeline,
            "feasible": feasible, "trace": traces[0], "restarts": len(starts),
            "all_scores": [t[-1][2] for t in traces]}


def exhaustive_best(cands, travel, start="START"):
    keys = list(cands)
    best = ([], evaluate_tour([], cands, travel, start)[0])
    for k in range(1, len(keys) + 1):
        for perm in itertools.permutations(keys, k):
            s = evaluate_tour(list(perm), cands, travel, start)[0]
            if s > best[1] + 1e-9:
                best = (list(perm), s)
    return best
