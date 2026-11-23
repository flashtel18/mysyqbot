"""
bayes.py - ALGORITHM 2: Bayesian inference (course topic: Week 6).

The robot CANNOT see through walls. It only knows the timetable and what its
camera shows. For every room whose lecture just ended it keeps a belief

    H = "this group hangs out in the hall outside their door this break"

  PRIOR  P(H)  - from the TIMETABLE (rules below, labelled assumptions)
  EVIDENCE     - camera: when the hall of that room comes into view, the robot
                 either sees some of the group ("seen") or nobody ("not seen")
  POSTERIOR    - Bayes rule:

      P(H | obs) =            P(obs | H) * P(H)
                   ---------------------------------------------
                   P(obs | H) * P(H)  +  P(obs | not H) * P(not H)

  The likelihoods depend on how long ago the class ended (t seconds):
      P(seen | H)     = P_DETECT * linger(t)            linger(t) = exp(-t/240)
      P(seen | not H) = P_DETECT * quick(t) + FALSE_ALARM   quick(t) = max(0, 1 - t/60)
  (a group that hangs out stays ~4 min; a group that doesn't is gone within 1 min)

All numbers are ASSUMPTIONS for the toy model.
"""
import math

from tour_planner import QUICK_LEAVE, TAU

# camera model (assumption)
P_DETECT = 0.90      # chance to notice students that are in view
FALSE_ALARM = 0.03   # chance to "see" the group when it is not there (other people)

# Prior rules from the timetable (assumptions, first matching rule wins)
PRIOR_LATE_TEACHER = 0.85   # teacher usually runs late -> students come out later, still around
PRIOR_LAST_CLASS = 0.25     # last class of the day -> students go home fast
PRIOR_LONG_GAP = 0.40       # lunch / long gap -> students leave for food
PRIOR_SHORT_BREAK = 0.65    # normal 10-min break -> students hang around


def prior_present(lecture, timetable):
    """Returns (prior, reason) using only what the timetable says."""
    if lecture.late_teacher:
        return PRIOR_LATE_TEACHER, "teacher usually finishes late"
    if timetable.is_last_of_day(lecture):
        return PRIOR_LAST_CLASS, "last class of the day for this group"
    gap = timetable.gap_after(lecture)
    if gap is not None and gap >= 60:
        return PRIOR_LONG_GAP, f"long gap ({gap} min) - lunch time"
    return PRIOR_SHORT_BREAK, "short break, next class soon"


def linger(t):
    return math.exp(-max(0.0, t) / TAU)


def quick(t):
    return max(0.0, 1 - max(0.0, t) / QUICK_LEAVE)


def posterior(prior, like_h, like_not_h):
    """Bayes rule for one binary hypothesis. Returns every intermediate number."""
    joint_h = like_h * prior
    joint_n = like_not_h * (1 - prior)
    evidence = joint_h + joint_n
    return {"prior": prior, "P(obs|H)": like_h, "P(obs|not H)": like_not_h,
            "evidence": evidence, "posterior": joint_h / evidence if evidence else prior}


def camera_update(prior, seen, since_end):
    """Update P(H) after looking at the hall (seen = saw some of the group)."""
    seen_h = P_DETECT * linger(since_end)
    seen_n = min(1.0, P_DETECT * quick(since_end) + FALSE_ALARM)
    if seen:
        res = posterior(prior, seen_h, seen_n)
    else:
        res = posterior(prior, 1 - seen_h, 1 - seen_n)
    res.update(obs="seen" if seen else "not seen", since_end=since_end)
    return res


def expected_share(p, since_end):
    """Expected share of the out-coming students still in the hall now."""
    return p * linger(since_end) + (1 - p) * quick(since_end)


if __name__ == "__main__":
    for prior in (0.25, 0.65, 0.85):
        for seen in (True, False):
            for t in (30, 120):
                r = camera_update(prior, seen, t)
                print(f"prior={prior:.2f} {r['obs']:8s} t={t:3d}s -> P(H)={r['posterior']:.3f}")
