"""
mdp.py - ALGORITHM 5: Markov Decision Process + value iteration
         (course topic: Week 5, sequential decision making).

Question: "I reached the group. What should I DO?"

States  : group mood  Energetic / Tired / Stressed   (estimated by the HMM)
Actions : Motivate, Joke, MiniQuiz, LeaveAlone
R(s,a)  : immediate reward (how good the action is for that mood)
T(s'|s,a): how the action changes the mood
gamma   : 0.8 - the robot meets the same students again later, so it should
          care about their FUTURE mood, not only the immediate reaction.

Value iteration (Bellman update, repeated until convergence):
    Q(s,a) = R(s,a) + gamma * SUM_s' T(s'|s,a) * V(s')
    V(s)   = max_a Q(s,a)

The robot does NOT know the true mood, only the HMM belief b(s), so it picks
    a* = argmax_a  SUM_s b(s) * Q(s,a)        (the "QMDP" approximation)
All numbers are ASSUMPTIONS for the toy model.
"""
STATES = ["Energetic", "Tired", "Stressed"]
ACTIONS = ["Motivate", "Joke", "MiniQuiz", "LeaveAlone"]
GAMMA = 0.8

REWARD = {
    "Energetic": {"Motivate": 2, "Joke": 3, "MiniQuiz": 8, "LeaveAlone": 0},
    "Tired":     {"Motivate": 4, "Joke": 6, "MiniQuiz": -3, "LeaveAlone": 1},
    "Stressed":  {"Motivate": 7, "Joke": -2, "MiniQuiz": -6, "LeaveAlone": 2},
}

# TRANSITION[action][state] = {next_state: prob}
TRANSITION = {
    "Motivate": {
        "Energetic": {"Energetic": 0.8, "Tired": 0.1, "Stressed": 0.1},
        "Tired":     {"Energetic": 0.4, "Tired": 0.5, "Stressed": 0.1},
        "Stressed":  {"Energetic": 0.3, "Tired": 0.3, "Stressed": 0.4},
    },
    "Joke": {
        "Energetic": {"Energetic": 0.9, "Tired": 0.1, "Stressed": 0.0},
        "Tired":     {"Energetic": 0.6, "Tired": 0.4, "Stressed": 0.0},
        "Stressed":  {"Energetic": 0.1, "Tired": 0.2, "Stressed": 0.7},
    },
    "MiniQuiz": {
        "Energetic": {"Energetic": 0.6, "Tired": 0.2, "Stressed": 0.2},
        "Tired":     {"Energetic": 0.1, "Tired": 0.6, "Stressed": 0.3},
        "Stressed":  {"Energetic": 0.0, "Tired": 0.2, "Stressed": 0.8},
    },
    "LeaveAlone": {
        "Energetic": {"Energetic": 0.7, "Tired": 0.3, "Stressed": 0.0},
        "Tired":     {"Energetic": 0.1, "Tired": 0.8, "Stressed": 0.1},
        "Stressed":  {"Energetic": 0.1, "Tired": 0.3, "Stressed": 0.6},
    },
}


def q_value(s, a, V, gamma=GAMMA):
    return REWARD[s][a] + gamma * sum(p * V[s2] for s2, p in TRANSITION[a][s].items())


def value_iteration(gamma=GAMMA, eps=1e-6, max_iter=1000):
    """Returns (V, Q, iterations, history_of_V)."""
    V = {s: 0.0 for s in STATES}
    history = [dict(V)]
    for it in range(1, max_iter + 1):
        newV = {s: max(q_value(s, a, V, gamma) for a in ACTIONS) for s in STATES}
        delta = max(abs(newV[s] - V[s]) for s in STATES)
        V = newV
        history.append(dict(V))
        if delta < eps:
            break
    Q = {s: {a: q_value(s, a, V, gamma) for a in ACTIONS} for s in STATES}
    return V, Q, it, history


def policy(Q):
    return {s: max(Q[s], key=Q[s].get) for s in STATES}


def choose_action(belief, Q):
    """QMDP: expected Q under the belief. Returns (best_action, scores)."""
    scores = {a: sum(belief[s] * Q[s][a] for s in STATES) for a in ACTIONS}
    return max(scores, key=scores.get), scores


if __name__ == "__main__":
    V, Q, it, _ = value_iteration()
    print(f"converged in {it} iterations")
    for s in STATES:
        print(f"{s:10s} V={V[s]:6.2f}  " + "  ".join(f"{a}={Q[s][a]:6.2f}" for a in ACTIONS))
    print("policy:", policy(Q))
