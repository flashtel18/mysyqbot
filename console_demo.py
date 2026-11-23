"""
console_demo.py - step-by-step demo in the terminal (no extra libraries).

Shows one break of the simulated day as a story:
  INPUT (timetable) -> every algorithm's steps as they happen -> RESULT -> WHY

Run:   python console_demo.py                  (break at 14:40, seed 7)
       python console_demo.py --break 10:20    (another break)
       python console_demo.py --ask            (then chat with MysyqBot)
"""
import argparse
import sys

import mdp
from campus import hhmm_to_sec, sec_to_hhmm
from world import VISION_R, Simulation

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WHY = {
    "Camera": f"The robot cannot see through walls ({VISION_R} cells range), so it must GUESS where "
              "students are and go and look.",
    "Bayes": "The timetable gives a prior per hall; every look of the camera ('seen' / 'not seen') "
             "updates it with Bayes rule. Right after class 'seen' proves little (everyone is out).",
    "A*": "Fastest route through the halls only; the Manhattan heuristic is admissible, so the route is "
          "optimal and A* expands far fewer cells than BFS.",
    "Hill climbing": "The break is only 10 minutes: choose which halls and in which order, re-planned "
                     "whenever the belief changes or a meeting ends.",
    "HMM": "Mood is hidden; the HMM combines all of today's activity readings instead of trusting one.",
    "MDP": "Value iteration plans for the group's future mood too; the HMM belief picks the action.",
    "Naive Bayes": "Students type in their own words (EN/RU); Naive Bayes turns words into an intent "
                   "so the robot knows which tool to use.",
}


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--break", dest="brk", default="14:40")
    ap.add_argument("--ask", action="store_true", help="interactive questions at the end")
    args = ap.parse_args()

    e = hhmm_to_sec(args.brk)
    sim = Simulation(seed=args.seed, start=sec_to_hhmm(e - 60))
    ending = sim.tt.ending_at(e)
    if not ending:
        print(f"No lectures end at {args.brk}. Breaks: {[sec_to_hhmm(t) for t in sim.tt.end_times]}")
        return

    header(f"MysyqBot console demo | seed {args.seed} | break at {args.brk}")
    print("INPUT 1 - timetable rows ending now (the robot KNOWS this):")
    for lec in ending:
        dl = sim.tt.next_deadline(lec.group)
        print(f"  room {lec.room}: {lec.group:8s} {lec.course:28s} {lec.size} students, "
              f"late teacher={lec.late_teacher}, next deadline: "
              f"{dl['course'] + ' ' + dl['kind'] + ' in ' + str(dl['days']) + ' d' if dl else '-'}")
    print(f"INPUT 2 - its camera: {VISION_R} cells, line of sight, cannot see inside rooms.")
    print("HIDDEN (only the simulator knows): which groups hang out, who comes out (~40%), moods.")

    header("THE BREAK, STEP BY STEP  ([Bayes] [A*] [Hill climbing] [camera] [HMM] [MDP] [Naive Bayes])")
    n0 = len(sim.log_lines)
    sim.step(60 + 600 + 30)
    for ln in sim.log_lines[n0:]:
        if ln.strip().startswith(("add ", "remove ", "swap ", "replace ", "start ")) and "->" in ln:
            print("   " + ln)        # hill-climbing moves
        else:
            print(ln)

    header("RESULT of this break (compared with the simulator's hidden truth)")
    for lec in ending:
        tr = sim.truth[lec.id]
        met = [v for v in sim.visits if v[0].id == lec.id]
        print(f"  {lec.room} {lec.group:8s} truly hangs out: {str(tr['present']):5s} | "
              + (f"met {sum(v[2] for v in met)} of the group" if met else "not met"))
    st = sim.stats
    print(f"  students met in the halls: {st['reached']} | groups spotted by camera: {st['spotted']} | "
          f"walked up by themselves: {st['came_by_themselves']} | empty halls checked: {st['empty']}")

    header("MDP policy (value iteration)")
    V, Q, iters, _ = mdp.value_iteration()
    print(f"converged in {iters} iterations (gamma={mdp.GAMMA}):")
    for s in mdp.STATES:
        print(f"   {s:10s} V={V[s]:6.2f} | " + " ".join(f"{a}={Q[s][a]:6.2f}" for a in mdp.ACTIONS))
    print(f"   optimal policy: {mdp.policy(Q)}")

    header("Naive Bayes - student questions")
    for q in ["where is room 216?", "когда следующая пара AIN-2601", "what is a heuristic?",
              "где 207?", "where can i eat", "я устал", "hello cat!"]:
        res = sim.ask(q)
        print(f"Q: {q}")
        for ln in res["log"]:
            print(ln)
        print(f"A: {res['reply']}\n")

    header("WHY each algorithm is needed")
    for k, v in WHY.items():
        print(f"  {k:14s} {v}")

    if args.ask:
        header("Ask MysyqBot anything (empty line to quit)")
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                break
            if not q:
                break
            res = sim.ask(q)
            print("   " + "\n   ".join(res["log"]))
            print(f"MysyqBot> {res['reply']}")


if __name__ == "__main__":
    main()
