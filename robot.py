"""
robot.py - MysyqBot's BRAIN. This is where the algorithms work together:

  the robot only knows the timetable + what its CAMERA sees (world.py)
     |
     |-- Bayes ........ belief P(group hangs out in the hall) for every room that
     |                  just ended: timetable prior, updated by camera observations
     |-- A* ........... travel time between robot and every hall (halls only)
     |-- Hill climbing  which halls to check, in which order, inside the break
     |-- (camera sees students -> A* straight to them)
     v
  robot meets a group
     |-- HMM .......... belief about the group's mood (from today's readings)
     |-- MDP .......... best action for that belief -> message to students
     v
  student asks a question
     |-- Naive Bayes .. intent of the question
     |-- A* / timetable / FAQ .. the answer (A* also draws the route)

Every method returns `log` lines so the console demo and the GUI can show
input -> key steps -> output for each algorithm.
"""
import random
import re

import bayes
import mood_hmm
import mdp
import tour_planner
from astar import astar, describe_path
from campus import SECONDS_PER_CELL, ROOM_TITLES, sec_to_hhmm
from intent_nb import NaiveBayesIntent

ROOM_WORDS = {   # words -> place on floor 2 (checked in this order)
    "toilet": "WC", "wc": "WC", "restroom": "WC", "туалет": "WC", "уборн": "WC",
    "robotics": "Lab", "lab": "Lab", "лаборат": "Lab",
    "stairs": "EXIT", "elevator": "EXIT", "lift": "EXIT", "exit": "EXIT",
    "лифт": "EXIT", "лестниц": "EXIT", "выход": "EXIT",
    "cafeteria": "OTHER_FLOOR", "canteen": "OTHER_FLOOR", "eat": "OTHER_FLOOR",
    "library": "OTHER_FLOOR", "столов": "OTHER_FLOOR", "кафе": "OTHER_FLOOR",
    "библиотек": "OTHER_FLOOR",
}

AI_FAQ = {
    "a star": "A* picks the node with the lowest f = g + h: real cost so far plus a heuristic guess. "
              "If h never overestimates, A* finds the cheapest path. I use it to walk around campus!",
    "heuristic": "A heuristic is a quick estimate of the distance to the goal. Mine is Manhattan "
                 "distance: |row difference| + |column difference|.",
    "bfs": "BFS explores level by level, so it finds the path with the FEWEST steps. "
           "It ignores costs - that's why I prefer A* (the busy lobby is slow).",
    "dfs": "DFS goes deep first using a stack. Little memory, but no shortest-path guarantee.",
    "bayes": "Bayes rule: posterior = likelihood x prior / evidence. I use it to guess in which "
             "halls students hang out, and update it with what my camera sees.",
    "hidden markov": "An HMM has hidden states (your mood) that emit observations (room activity). "
                     "Forward filtering gives P(mood now | all readings).",
    "hmm": "An HMM has hidden states (your mood) that emit observations (room activity). "
           "Forward filtering gives P(mood now | all readings).",
    "viterbi": "Viterbi finds the single most likely sequence of hidden states, using dynamic "
               "programming over max instead of sum.",
    "mdp": "An MDP = states, actions, transition probabilities, rewards. Value iteration repeats "
           "the Bellman update V(s) = max_a [R + gamma * sum T V(s')] until it converges.",
    "markov decision": "An MDP = states, actions, transition probabilities, rewards. Value iteration "
                       "repeats the Bellman update until V stops changing.",
    "value iteration": "Value iteration applies the Bellman update to every state again and again "
                       "until the values converge, then picks the best action per state.",
    "bellman": "Bellman: V(s) = max_a [ R(s,a) + gamma * sum_s' T(s'|s,a) V(s') ].",
    "hill climbing": "Hill climbing: look at neighbour solutions, move to the best one, stop at a "
                     "local optimum. I add random restarts to escape bad hills.",
    "simulated annealing": "Like hill climbing, but sometimes accepts worse moves (with probability "
                           "exp(-delta/T)) and cools T down to escape local optima.",
    "local search": "Local search keeps one current solution and improves it step by step "
                    "(hill climbing, simulated annealing).",
    "naive bayes": "Naive Bayes assumes words are independent given the class: "
                   "P(class|words) ~ P(class) x product P(word|class). I use it to understand you!",
    "minimax": "Minimax: MAX picks the move with the best value assuming MIN answers perfectly. "
               "Alpha-beta pruning skips branches that can't change the result.",
    "alpha beta": "Alpha-beta pruning cuts branches of the minimax tree that cannot affect the "
                  "final decision - same answer, fewer nodes.",
    "perceptron": "A perceptron computes w*x + b and outputs 1 if it's > 0. It learns by nudging "
                  "w toward misclassified examples.",
    "q learning": "Q-learning updates Q(s,a) += alpha [r + gamma max Q(s',a') - Q(s,a)] from "
                  "experience, without knowing the transition model.",
    "bayesian network": "A Bayesian network is a graph of variables with conditional probability "
                        "tables - a compact way to write a big joint distribution.",
    "эвристик": "Эвристика - быстрая оценка расстояния до цели. Я использую манхэттенское расстояние.",
    "байес": "Правило Байеса: апостериорная = правдоподобие x априорная / нормировка.",
}


# Questions that simulated students ask on their own when they meet the robot
STUDENT_QUESTIONS = [
    "where is room 219", "where is the toilet", "где женский туалет", "how do i get to 203",
    "where can i eat", "where are the stairs", "как пройти в 224",
    "where is 225", "где 207",
    "when is my next class {group}", "when is the endterm {group}",
    "какая следующая пара {group}", "what is a heuristic?", "explain bayes rule",
    "how does value iteration work", "what is hill climbing", "i am so tired",
    "i'm stressed about the endterm", "hello cat", "are you a real cat",
]


class MysyqBrain:
    def __init__(self, campus, timetable, rng=None):
        self.campus = campus
        self.tt = timetable
        self.rng = rng or random.Random(0)
        self.V, self.Q, self.vi_iters, _ = mdp.value_iteration()
        self.nb = NaiveBayesIntent().fit()
        self._dist = {}

    # ------------------------------------------------------------------ A*
    def route(self, a, b, halls_only=True):
        """A* route. The robot (default) stays in the halls; students may use doors."""
        key = (a, b, halls_only)
        if key not in self._dist:
            self._dist[key] = astar(self.campus, a, b, halls_only)
        return self._dist[key]

    def travel_sec(self, a, b):
        return self.route(a, b)[1] * SECONDS_PER_CELL

    # ------------------------------------------------- break planning
    def make_zone(self, lec, p, now):
        """Planner input for one hall zone (the hall outside lec.room)."""
        dl = self.tt.next_deadline(lec.group)
        late = tour_planner.LATE_DELAY if lec.late_teacher else 0
        return {"lecture": lec, "p": p, "size": lec.size,
                "priority": tour_planner.priority_from_deadline(dl["days"] if dl else None),
                "delay": lec.end + late - now}          # >0: class still running

    def plan_tour(self, now, robot_cell, zones, budget, why="break"):
        """Hill climbing over the hall zones the robot BELIEVES may have students.
        zones = {room: make_zone(...)} with p from the Bayes belief.
        Returns the plan + log lines + data for the explanation panel."""
        log = [f"[Hill climbing] plan ({why}) at {sec_to_hhmm(now, True)}, "
               f"{budget:.0f}s left in the break"]
        doors = {"START": robot_cell, **{k: self.campus.spot(k) for k in zones}}

        def travel(a, b):
            return self.travel_sec(doors[a], doors[b])

        log.append("[A*] travel times in the halls: " +
                   ", ".join(f"{k} {travel('START', k):.0f}s" for k in zones))
        plan = tour_planner.plan_tour(zones, travel, restarts=8, rng=self.rng, budget=budget)
        for move, tour, score in plan["trace"]:
            log.append(f"  {move:22s} -> {'>'.join(tour) or '(empty)'}  score {score:.1f}")
        log.append(f"  BEST tour: {' > '.join(plan['tour']) or '(none - just patrol)'}  "
                   f"(score {plan['score']:.1f} = expected students x exam priority)")
        skipped = [k for k in zones if k not in plan["tour"]]
        if skipped:
            log.append(f"  skipped: {', '.join(skipped)} (not worth the time)")
        path, cur = [], robot_cell
        for k in plan["tour"]:
            p = self.route(cur, doors[k])[0]
            path += p if not path else p[1:]
            cur = doors[k]
        explain = {"trace": [[m, t, round(sc, 2)] for m, t, sc in plan["trace"]],
                   "best": plan["tour"], "score": round(plan["score"], 2),
                   "restarts": plan["restarts"], "all_scores": [round(x, 2) for x in plan["all_scores"]],
                   "budget": round(budget), "skipped": skipped,
                   "travel": {k: round(travel("START", k)) for k in zones},
                   "zones": {k: {"p": round(z["p"], 3), "size": z["size"], "priority": z["priority"]}
                             for k, z in zones.items()}}
        return {"plan": plan, "path": path, "log": log, "explain": explain}

    # ------------------------------------------------- at the room
    def decide_action(self, lecture, activity_readings):
        """activity_readings: today's readings of this group (oldest first)."""
        log = [f"[HMM] group {lecture.group}, readings today: {activity_readings}"]
        dl = self.tt.next_deadline(lecture.group)
        init = mood_hmm.initial_distribution(dl["days"] if dl else None)
        beliefs = mood_hmm.forward(activity_readings, init)
        belief = beliefs[-1]
        path = mood_hmm.viterbi(activity_readings, init)
        log.append("  belief now: " + ", ".join(f"{s} {belief[s]:.2f}" for s in mood_hmm.STATES))
        log.append(f"  Viterbi mood path: {' > '.join(path)}")

        action, scores = mdp.choose_action(belief, self.Q)
        log.append("[MDP] expected Q under belief: " +
                   ", ".join(f"{a} {v:.1f}" for a, v in scores.items()))
        log.append(f"  -> action: {action}")
        return {"belief": belief, "viterbi": path, "action": action,
                "scores": scores, "log": log, "message": self.message(action, lecture)}

    def message(self, action, lec):
        dl = self.tt.next_deadline(lec.group)
        nxt = self.tt.next_of_group(lec.group, lec.end)
        exam = (f"{dl['course']} {dl['kind']} in {dl['days']} day{'s' * (dl['days'] != 1)}"
                if dl else "no exams soon")
        nxt_s = f"{nxt.course} at {sec_to_hhmm(nxt.start)} in {nxt.room}" if nxt else "nothing - go home!"
        options = {
            "Motivate": [
                f"Meow! {exam}. You've got this - one step at a time, like A*.",
                f"{lec.group}, I believe in you! {exam}. Tea, then revise 25 min.",
            ],
            "Joke": [
                "Why did the robot cat fail the exam? It only studied the cache. Meow!",
                "My heuristic says: you are 0 steps away from a snack break.",
                "I tried BFS to find motivation. Expanded 1000 nodes, found coffee.",
            ],
            "MiniQuiz": [
                "Quick quiz! Is Manhattan distance admissible on a 4-way grid? (Yes!)",
                "Quick quiz! In Bayes rule, what do we divide by? (The evidence!)",
                "Quick quiz! What does gamma do in an MDP? (Discounts the future!)",
            ],
            "LeaveAlone": ["(MysyqBot quietly waves its tail and lets you rest.)"],
        }
        return self.rng.choice(options[action]) + f"  Next: {nxt_s}"

    # ------------------------------------------------- questions
    def answer(self, text, t, robot_cell, near_group=None):
        probs, words = self.nb.predict_proba(text)
        intent = probs[0][0]
        log = [f"[Naive Bayes] '{text}' words={words}",
               "  " + ", ".join(f"{c} {p:.2f}" for c, p in probs[:3]),
               f"  -> intent: {intent}"]
        reply, target, path = None, None, None
        low = text.lower()

        if intent == "location":
            target_room, prefix = None, ""
            m = re.search(r"\b(\d{3})\b", low)
            if m and m.group(1) in self.campus.rooms:
                target_room = m.group(1)
            elif m:
                target_room, prefix = "EXIT", f"Room {m.group(1)} is not on floor 2. "
            else:
                for w, room in ROOM_WORDS.items():
                    if w in low:
                        target_room = room
                        break
                if target_room == "OTHER_FLOOR":
                    target_room, prefix = "EXIT", "That's on another floor. "
            gender = None
            if target_room == "WC":                  # nearest toilet of the right kind
                kinds = self.campus.toilets
                if any(w in low for w in ("women", "female", "girl", "lady", "ladies", "женск", "девоч", "девушк")):
                    kinds, gender = [k for k in kinds if "-F" in k], "F"
                elif any(w in low for w in ("men", "male", "boy", "мужск", "мальчик", "парн")):
                    kinds, gender = [k for k in kinds if "-M" in k], "M"
                target_room = min(kinds, key=lambda k: self.travel_sec(robot_cell, self.campus.spot(k)))
            if target_room and target_room != "EXIT" and target_room not in self.campus.rooms:
                reply = (f"I don't have the {target_room} on my floor-2 map yet. "
                         f"Ask at the info screen, meow!")
            elif target_room:
                target = self.campus.exit_spot if target_room == "EXIT" else self.campus.spot(target_room)
                path, cost, expanded = self.route(robot_cell, target)   # halls only
                if target_room.startswith("WC") and gender:
                    name = "the men's toilet" if gender == "M" else "the women's toilet"
                elif target_room.startswith("WC"):
                    name = "the nearest toilets (men's and women's doors are side by side)"
                else:
                    name = {"EXIT": "the central stairs"}.get(target_room, f"room {target_room}")
                reply = (f"{prefix}Follow me to {name}! About {cost * SECONDS_PER_CELL} s. "
                         f"Route: {describe_path(path)}.")
                log.append(f"[A*] to {name}: cost {cost}, {len(path)} cells, expanded {expanded}")
            else:
                reply = "Which room? Try: 'where is 204' or 'where is the WC'."

        elif intent == "schedule":
            m = re.search(r"\b([a-z]{2,4}-\d{4})\b", low)
            group = m.group(1).upper() if m else near_group
            if not group:
                reply = "Tell me your group, e.g. 'next class AIN-2601'."
            elif any(w in low for w in ("exam", "endterm", "midterm", "deadline", "экзамен", "quiz")):
                dl = self.tt.next_deadline(group)
                reply = (f"{group}: {dl['course']} {dl['kind']} on {dl['date']:%d %b} "
                         f"({dl['days']} days). You can do it!" if dl else f"{group}: no deadlines found.")
            else:
                now = next((l for l in self.tt.of_group(group) if l.start <= t < l.end), None)
                nxt = self.tt.next_of_group(group, t)
                if now:
                    reply = (f"{group}: you should be in {now.course} in room {now.room} right now "
                             f"(started {sec_to_hhmm(now.start)})! Run, I'll show you the way.")
                elif nxt:
                    reply = (f"{group}: next is {nxt.course} ({nxt.kind}) at {sec_to_hhmm(nxt.start)} "
                             f"in room {nxt.room}.")
                else:
                    reply = f"{group}: no more classes today. Rest!"
            log.append(f"[Timetable] group={group}")

        elif intent == "course_help":
            norm = low.replace("*", " star").replace("-", " ")
            for key, ans in AI_FAQ.items():
                if key in norm:
                    reply = ans
                    break
            reply = reply or "Good question! I only know AI Fundamentals topics - ask your TA at office hours."

        elif intent == "motivation":
            reply = self.rng.choice([
                "You are doing better than you think. 25 minutes of study, 5 minutes of rest. Meow!",
                "Even A* expands one node at a time. Small steps!",
                "Drink water, take a breath. The endterm is a hill - and you can climb it.",
            ])
        else:
            reply = self.rng.choice([
                "Meow! I'm MysyqBot, QAIRU's robot cat. I run on A* and tea.",
                "Purr... Hello! Ask me where a room is or what A* means.",
                "Meow meow. I can't eat fish, but I can show you the stairs to the cafeteria.",
            ])
        return {"intent": intent, "probs": probs, "reply": reply, "words": words,
                "target": target, "path": path, "log": log, "text": text}
