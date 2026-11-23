"""
mood_hmm.py - ALGORITHM 4: Hidden Markov Model (course topic: Week 7,
              sequential inference: forward filtering + Viterbi).

Question: "How does this group FEEL right now?"
The mood is HIDDEN - the robot can't read minds. What it CAN observe is the
room's activity sensor after each of the group's classes today:
    Active   (lots of talking / moving)
    Calm     (quiet, slow)
    Restless (fidgeting, complaining, packing bags early)

Hidden states : Energetic, Tired, Stressed
Transition    : how mood changes from one class to the next
Emission      : P(sensor reading | mood)
Initial       : depends on how close the group's next exam is (deadlines.csv)

Forward algorithm  -> belief P(mood_now | all readings today)   (used by MDP)
Viterbi algorithm  -> the single most likely mood sequence      (for display)
All probabilities are ASSUMPTIONS for the toy model.
"""
import random

STATES = ["Energetic", "Tired", "Stressed"]
OBS = ["Active", "Calm", "Restless"]

TRANSITION = {  # P(next | current)
    "Energetic": {"Energetic": 0.60, "Tired": 0.30, "Stressed": 0.10},
    "Tired":     {"Energetic": 0.15, "Tired": 0.70, "Stressed": 0.15},
    "Stressed":  {"Energetic": 0.10, "Tired": 0.20, "Stressed": 0.70},
}

EMISSION = {  # P(observation | state)
    "Energetic": {"Active": 0.70, "Calm": 0.20, "Restless": 0.10},
    "Tired":     {"Active": 0.15, "Calm": 0.70, "Restless": 0.15},
    "Stressed":  {"Active": 0.10, "Calm": 0.25, "Restless": 0.65},
}


def initial_distribution(days_to_exam):
    """Mood at the start of the day. Exams soon -> more stress."""
    if days_to_exam is not None and days_to_exam <= 3:
        return {"Energetic": 0.30, "Tired": 0.20, "Stressed": 0.50}
    if days_to_exam is not None and days_to_exam <= 7:
        return {"Energetic": 0.45, "Tired": 0.25, "Stressed": 0.30}
    return {"Energetic": 0.60, "Tired": 0.30, "Stressed": 0.10}


def _normalize(d):
    z = sum(d.values())
    return {k: v / z for k, v in d.items()}


def forward(observations, init):
    """Forward filtering. Returns list of beliefs, one per observation."""
    beliefs = []
    belief = None
    for i, o in enumerate(observations):
        if i == 0:
            pred = dict(init)
        else:  # predict: sum over previous states
            pred = {s: sum(belief[p] * TRANSITION[p][s] for p in STATES) for s in STATES}
        belief = _normalize({s: pred[s] * EMISSION[s][o] for s in STATES})  # update
        beliefs.append(belief)
    return beliefs


def viterbi(observations, init):
    """Most likely hidden sequence."""
    if not observations:
        return []
    V = [{s: init[s] * EMISSION[s][observations[0]] for s in STATES}]
    back = [{}]
    for o in observations[1:]:
        col, ptr = {}, {}
        for s in STATES:
            prev, val = max(((p, V[-1][p] * TRANSITION[p][s]) for p in STATES),
                            key=lambda x: x[1])
            col[s] = val * EMISSION[s][o]
            ptr[s] = prev
        V.append(col)
        back.append(ptr)
    last = max(V[-1], key=V[-1].get)
    path = [last]
    for ptr in reversed(back[1:]):
        path.append(ptr[path[-1]])
    return path[::-1]


def _draw(dist, rng):
    x, acc = rng.random(), 0.0
    for k, v in dist.items():
        acc += v
        if x < acc:
            return k
    return k


def sample_day(n, init, rng=None):
    """Used by the SIMULATOR to create the 'true' hidden moods + readings."""
    rng = rng or random.Random()
    states, obs = [], []
    for i in range(n):
        s = _draw(init if i == 0 else TRANSITION[states[-1]], rng)
        states.append(s)
        obs.append(_draw(EMISSION[s], rng))
    return states, obs


if __name__ == "__main__":
    init = initial_distribution(2)
    obs = ["Active", "Calm", "Restless"]
    for o, b in zip(obs, forward(obs, init)):
        print(o, {k: round(v, 3) for k, v in b.items()})
    print("Viterbi:", viterbi(obs, init))
