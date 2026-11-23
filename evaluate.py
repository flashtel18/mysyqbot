"""
evaluate.py - RESULTS: how good is MysyqBot, compared with simpler robots?

PART 1 runs the FULL simulation (camera, walking, students coming out and
leaving) for every break of many simulated days, once per strategy:
  MysyqBot         Bayes belief + camera updates + hill-climbing tour + go to anyone it sees
  No camera Bayes  same, but the belief ignores the camera (timetable prior only)
  Nearest halls    checks the halls of ending lectures nearest-first (no probabilities)
  Random wander    random patrol; goes to students only when it happens to see them
All robots have the same camera and go to any students they see.

PART 2 checks every algorithm on its own against a baseline.

Run:   python evaluate.py            (60 days, ~2-3 min, writes results/evaluation.md)
       python evaluate.py --days 20  (quicker)
       python evaluate.py --chart    (also results/evaluation.png, needs matplotlib)
"""
import argparse
import itertools
import os
import random
import statistics as st

import bayes
import mdp
import mood_hmm
from astar import astar, bfs
from campus import Campus, Timetable, SECONDS_PER_CELL, sec_to_hhmm
from intent_nb import leave_one_out_accuracy
from tour_planner import evaluate_tour, exhaustive_best, plan_tour, priority_from_deadline
from world import Simulation, make_truth

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STRATS = [("mysyq", "MysyqBot"), ("no_camera_bayes", "No camera Bayes"),
          ("nearest", "Nearest halls"), ("wander", "Random wander")]


def run_strategies(days):
    tt = Timetable()
    res = {k: {"met": [], "empty": 0, "visits": 0, "spotted": 0} for k, _ in STRATS}
    bayes_rec = []
    for day in range(days):
        for e in tt.end_times:
            if not tt.ending_at(e):
                continue
            for key, _ in STRATS:
                # same seed -> same hidden truth for every strategy
                sim = Simulation(seed=1000 + day, start=sec_to_hhmm(e - 60), student_questions=False,
                                 strategy=key, explain=False)
                sim.step(60 + 600 + 60)
                r = res[key]
                r["met"].append(sim.stats["reached"])
                r["empty"] += sim.stats["empty"]
                r["visits"] += sim.stats["visits"] + sim.stats["empty"]
                r["spotted"] += sim.stats["spotted"]
                if key == "mysyq":
                    sim._close_break()
                    bayes_rec += sim.bayes_records
        print(f"  day {day + 1}/{days} done", end="\r")
    print()
    return res, bayes_rec


def algorithm_checks(days):
    campus, tt = Campus(), Timetable()
    out = {}
    # A* vs BFS on all pairs of meeting spots (halls only)
    pts = [campus.dock, campus.exit_spot] + [r["spot"] for r in campus.rooms.values()]
    ac = bc = ae = be = n = 0
    for p in pts:
        for q in pts:
            if p != q:
                _, c1, e1 = astar(campus, p, q, halls_only=True)
                _, c2, e2 = bfs(campus, p, q, halls_only=True)
                ac += c1; bc += c2; ae += e1; be += e2; n += 1
    out["astar"] = (ac / n, bc / n, ae / n, be / n)

    # hill climbing vs exhaustive search on the break-start plans
    dist = {}

    def rc(a, b):
        if (a, b) not in dist:
            dist[(a, b)] = astar(campus, a, b, halls_only=True)[1] * SECONDS_PER_CELL
        return dist[(a, b)]
    opt = tot = 0
    gaps = []
    for day in range(days):
        rng = random.Random(day)
        for e in tt.end_times:
            ending = tt.ending_at(e)
            cands = {}
            for lec in ending:
                prior, _ = bayes.prior_present(lec, tt)
                dl = tt.next_deadline(lec.group)
                cands[lec.room] = {"lecture": lec, "p": min(1, max(0, prior + rng.uniform(-.3, .3))),
                                   "size": lec.size, "delay": 180 if lec.late_teacher else 0,
                                   "priority": priority_from_deadline(dl["days"] if dl else None)}
            doors = {"START": campus.dock, **{k: campus.spot(k) for k in cands}}

            def travel(a, b, d=doors):
                return rc(d[a], d[b])
            plan = plan_tour(cands, travel, rng=random.Random(day))
            best = exhaustive_best(cands, travel)[1]
            tot += 1
            opt += abs(plan["score"] - best) < 1e-6
            gaps.append(best - plan["score"])
    out["hill"] = (opt / tot, st.mean(gaps), tot)

    # HMM + MDP decision quality against the hidden true mood
    _, Q, _, _ = mdp.value_iteration()
    dec = {"HMM belief": 0.0, "latest reading only": 0.0, "always Motivate": 0.0, "true mood": 0.0}
    hits = base = m = 0
    for day in range(days * 3):
        truth = make_truth(tt, random.Random(1000 + day))
        for group in tt.groups:
            lecs = tt.of_group(group)
            dl = tt.next_deadline(group)
            init = mood_hmm.initial_distribution(dl["days"] if dl else None)
            obs = [truth[l.id]["activity"] for l in lecs]
            for i, b in enumerate(mood_hmm.forward(obs, init)):
                true = truth[lecs[i].id]["mood"]
                guess = max(mood_hmm.STATES, key=lambda s: mood_hmm.EMISSION[s][obs[i]])
                one_hot = lambda mood: {s: float(s == mood) for s in mood_hmm.STATES}
                hits += max(b, key=b.get) == true
                base += guess == true
                dec["HMM belief"] += Q[true][mdp.choose_action(b, Q)[0]]
                dec["latest reading only"] += Q[true][mdp.choose_action(one_hot(guess), Q)[0]]
                dec["always Motivate"] += Q[true]["Motivate"]
                dec["true mood"] += Q[true][mdp.choose_action(one_hot(true), Q)[0]]
                m += 1
    out["hmm"] = (hits / m, base / m, {k: v / m for k, v in dec.items()})
    out["nb"] = leave_one_out_accuracy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args()

    print("PART 1: full simulation of every break ...")
    res, brec = run_strategies(args.days)
    print("PART 2: algorithm checks ...")
    chk = algorithm_checks(args.days)

    n_breaks = len(res["mysyq"]["met"])
    lines = [f"# MysyqBot evaluation ({args.days} simulated days, {n_breaks} breaks, full simulation)\n",
             "## Main goal: students met in the halls per break\n",
             "All robots have the same camera (9 cells, no seeing through walls) and walk to any "
             "students they see. Only ~40% of each group come out of the room.\n",
             "| Strategy | Avg students met / break | Trips to an empty hall | Groups spotted by camera |",
             "|---|---|---|---|"]
    means = {}
    for key, name in STRATS:
        r = res[key]
        means[key] = st.mean(r["met"])
        sem = st.stdev(r["met"]) / len(r["met"]) ** 0.5
        empty = "— (no planned stops)" if key == "wander" else f"{r['empty'] / max(1, r['visits']):.0%}"
        lines.append(f"| {name} | {means[key]:.1f} ± {sem:.1f} | {empty} | {r['spotted'] / n_breaks:.1f} |")
    lines += ["",
              f"MysyqBot meets {means['mysyq'] / means['wander'] - 1:+.0%} more students than random "
              f"wandering and {means['mysyq'] / means['nearest'] - 1:+.0%} more than nearest-first.\n"]

    # Bayes: final belief vs prior, for halls the camera actually looked at
    looked = [r for r in brec if r[3] > 0]
    def brier(idx, rows):
        return st.mean((r[idx] - r[2]) ** 2 for r in rows) if rows else float("nan")
    def acc(idx, rows):
        return st.mean(((r[idx] > 0.5) == r[2]) for r in rows) if rows else float("nan")
    (ac, bc, ae, be) = chk["astar"]
    hopt, hgap, htot = chk["hill"]
    hacc, hbase, dec = chk["hmm"]
    lines += ["## Per-algorithm checks\n",
              "| Algorithm | Metric | Ours | Baseline |", "|---|---|---|---|",
              f"| Bayes (camera) | accuracy of P(group hangs out) > 0.5, halls the camera saw ({len(looked)}) "
              f"| {acc(1, looked):.1%} | timetable prior only: {acc(0, looked):.1%} |",
              f"| Bayes (camera) | Brier score (lower = better) | {brier(1, looked):.3f} | prior only: {brier(0, looked):.3f} |",
              f"| A* | avg nodes expanded (halls only) | {ae:.1f} | BFS: {be:.1f} |",
              f"| A* | avg path cost | {ac:.1f} | BFS: {bc:.1f} |",
              f"| Hill climbing | found the true optimum ({htot} plans) | {hopt:.1%} | exhaustive search |",
              f"| Hill climbing | avg score gap to optimum | {hgap:.3f} | 0 |",
              f"| HMM | mood accuracy (forward filtering) | {hacc:.1%} | latest reading only: {hbase:.1%} |",
              f"| HMM + MDP | avg value of chosen action | {dec['HMM belief']:.2f} | latest reading: "
              f"{dec['latest reading only']:.2f}, always Motivate: {dec['always Motivate']:.2f}, "
              f"knowing true mood: {dec['true mood']:.2f} |",
              f"| Naive Bayes | intent accuracy (leave-one-out) | {chk['nb']:.1%} | majority class: ~20% |"]
    report = "\n".join(lines)
    print(report)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "evaluation.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    if args.chart:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [n for _, n in STRATS][::-1]
        vals = [means[k] for k, _ in STRATS][::-1]
        colors = ["#f28c28" if n == "MysyqBot" else "#b8bcc6" for n in names]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        bars = ax.barh(names, vals, color=colors, height=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center",
                    fontsize=10, color="#3d405b")
        ax.set_xlabel("Average students met in the halls per break", color="#3d405b")
        ax.set_title(f"Same camera, different brains ({args.days} simulated days)", loc="left",
                     fontsize=12, color="#3d405b")
        ax.grid(axis="x", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#3d405b", length=0)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "evaluation.png"), dpi=130)
        print("chart saved to results/evaluation.png")


if __name__ == "__main__":
    main()
