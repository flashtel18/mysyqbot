"""
tests.py - small test cases for every algorithm.
Run:  python tests.py      (or: python -m pytest tests.py)
"""
import random

import bayes
import mdp
import mood_hmm
from astar import astar, bfs
from campus import Campus, Timetable
from intent_nb import NaiveBayesIntent
from tour_planner import evaluate_tour, exhaustive_best, plan_tour

campus = Campus()
tt = Timetable()


def test_astar_optimal_and_avoids_busy_lobby():
    s, g = campus.dock, campus.spot("204")
    path, cost, _ = astar(campus, s, g, halls_only=True)
    assert path[0] == s and path[-1] == g
    assert all(campus.passable(c) for c in path)
    _, bfs_cost, _ = bfs(campus, s, g, halls_only=True)
    assert cost <= bfs_cost                    # A* never worse than BFS in cost
    assert not any(campus.char(c) == "~" for c in path)  # walks around the slow lobby


def test_robot_never_enters_rooms():
    for name in campus.rooms:
        path, _, _ = astar(campus, campus.dock, campus.spot(name), halls_only=True)
        assert path is not None, name
        assert all(campus.is_hall(c) for c in path), name
        assert campus.char(campus.spot(name)) != "+"   # waits OUTSIDE the door


def test_about_40_percent_come_out():
    from world import make_truth, sample_break
    rng = random.Random(5)
    total = out = 0
    for _ in range(200):
        truth = make_truth(tt, rng)
        for lec in tt.lectures:
            total += lec.size
            out += len(sample_break(lec, truth[lec.id], rng))
    assert 0.37 < out / total < 0.43


def test_astar_unreachable():
    path, cost, _ = astar(campus, campus.dock, (0, 0))   # (0,0) is a wall
    assert path is None and cost == float("inf")


def test_bayes_rule_by_hand():
    r = bayes.posterior(0.65, 0.85, 0.20)
    # 0.85*0.65 / (0.85*0.65 + 0.20*0.35) = 0.5525 / 0.6225
    assert abs(r["posterior"] - 0.5525 / 0.6225) < 1e-9


def test_bayes_prior_rules():
    late = next(l for l in tt.lectures if l.late_teacher)
    assert bayes.prior_present(late, tt)[0] == bayes.PRIOR_LATE_TEACHER
    last = next(l for l in tt.lectures if tt.is_last_of_day(l) and not l.late_teacher)
    assert bayes.prior_present(last, tt)[0] == bayes.PRIOR_LAST_CLASS


def _toy_tour_problem():
    cands = {"A": {"p": 0.9, "size": 20, "priority": 1.0},
             "B": {"p": 0.1, "size": 20, "priority": 1.0},
             "C": {"p": 0.8, "size": 25, "priority": 1.5}}
    times = {("START", "A"): 30, ("START", "B"): 20, ("START", "C"): 90,
             ("A", "B"): 40, ("A", "C"): 60, ("B", "C"): 50}

    def travel(a, b):
        return times.get((a, b)) or times.get((b, a))
    return cands, travel


def test_hill_climbing_matches_exhaustive():
    cands, travel = _toy_tour_problem()
    plan = plan_tour(cands, travel, rng=random.Random(1))
    best_tour, best_score = exhaustive_best(cands, travel)
    assert abs(plan["score"] - best_score) < 1e-9
    assert plan["feasible"]
    assert plan["tour"][0] == "B"        # 20 s away: even quick leavers are still in the hall
    assert "C" in plan["tour"]           # exam-priority room is not skipped


def test_tour_penalty_when_too_long():
    cands, travel = _toy_tour_problem()
    score, _, feasible = evaluate_tour(["A", "B", "C", "A", "B"], cands, travel, "START")
    assert not feasible and score < 0


def test_hmm_forward_is_distribution_and_reacts():
    init = mood_hmm.initial_distribution(None)
    beliefs = mood_hmm.forward(["Restless", "Restless", "Restless"], init)
    for b in beliefs:
        assert abs(sum(b.values()) - 1) < 1e-9
    assert max(beliefs[-1], key=beliefs[-1].get) == "Stressed"
    assert mood_hmm.viterbi(["Active", "Active"], init) == ["Energetic", "Energetic"]


def test_mdp_converges_and_policy_makes_sense():
    V, Q, iters, _ = mdp.value_iteration()
    assert iters < 1000
    pol = mdp.policy(Q)
    assert pol == {"Energetic": "MiniQuiz", "Tired": "Joke", "Stressed": "Motivate"}
    # Bellman equation holds at convergence
    for s in mdp.STATES:
        assert abs(V[s] - max(mdp.q_value(s, a, V) for a in mdp.ACTIONS)) < 1e-4


def test_naive_bayes_intents():
    nb = NaiveBayesIntent().fit()
    assert nb.predict("where is room 203") == "location"
    assert nb.predict("когда следующая пара") == "schedule"
    assert nb.predict("explain value iteration") == "course_help"
    assert nb.predict("i am so tired") == "motivation"
    probs, _ = nb.predict_proba("hello")
    assert abs(sum(p for _, p in probs) - 1) < 1e-9


def test_full_break_simulation_runs():
    from world import Simulation
    sim = Simulation(seed=3, start="10:19")
    visited = set()
    for _ in range(12 * 60):
        sim.step(1)
        visited.add(sim.robot["cell"])
    assert sim.stats["breaks"] == 1 and sim.stats["visits"] >= 1
    assert sim.stats["reached"] > 0
    assert all(campus.is_hall(c) for c in visited)   # robot stayed in the halls


def test_camera_cannot_see_through_walls_or_into_rooms():
    from world import Simulation, VISION_R
    sim = Simulation(seed=1, start="10:00")
    vis = sim.visible_from(campus.dock)
    assert vis and all(campus.char(c) in ".~S+VX" for c in vis)
    room_cells = {c for r in campus.rooms.values() for c in r["cells"]}
    assert not (vis & room_cells)                          # never sees inside a classroom
    assert all((c[0] - campus.dock[0]) ** 2 + (c[1] - campus.dock[1]) ** 2 <= VISION_R ** 2 for c in vis)


def test_camera_bayes_update():
    import bayes
    seen = bayes.camera_update(0.5, True, 120)["posterior"]
    not_seen = bayes.camera_update(0.5, False, 120)["posterior"]
    assert seen > 0.5 > not_seen
    # right after class everyone is in the hall: seeing them tells almost nothing
    assert abs(bayes.camera_update(0.5, True, 0)["posterior"] - 0.5) < 0.05


def test_mysyqbot_beats_random_wandering():
    from world import Simulation
    met = {"mysyq": 0, "wander": 0}
    for seed in range(6):
        for strat in met:
            sim = Simulation(seed=100 + seed, start="14:39", student_questions=False,
                             strategy=strat, explain=False)
            sim.step(700)
            met[strat] += sim.stats["reached"]
    assert met["mysyq"] > met["wander"]


def test_every_room_reachable_and_3d_payload_is_json():
    import json
    import server3d
    for name in campus.rooms:
        assert astar(campus, campus.dock, campus.spot(name), halls_only=True)[0] is not None, name
    app = server3d.App(seed=1, start="10:18")
    app.sim.step(120)
    json.dumps(app.map_payload())
    st = app.state_payload(0)
    json.dumps(st)
    assert st["students"] and st["log_total"] > 0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
