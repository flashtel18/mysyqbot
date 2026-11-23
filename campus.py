"""
campus.py - the "world knowledge" of MysyqBot.

Contains:
  * the floor map of one QAIRU floor (toy map, edit the ASCII art to change it)
  * rooms and their doors
  * the timetable (data/timetable.csv) and deadlines (data/deadlines.csv)

Everything here is plain data + small helpers. No AI yet - the AI lives in the
algorithm files (astar.py, bayes.py, tour_planner.py, mood_hmm.py, mdp.py,
intent_nb.py).
"""
import csv
import os
from dataclasses import dataclass
from datetime import date

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The simulated "today" (a Monday). Deadlines are counted from this date.
TODAY = date(2026, 11, 16)

# ---------------------------------------------------------------------------
# Floor map of QAIRU, FLOOR 2 - traced from a photo of the floor plan on the
# lobby info screen (tools/qairu_floor2.png) by tools/gen_map.py.
# One cell = 20 px of the photo (about 1.6 m).  Legend:
#   ' ' outside the building  #  wall          .  hall / corridor
#   ~   lounge with sofas next to the stairs (crowded, slow: cost 3)
#   +   door                  S  robot dock (next to the info screen)
#   X   central stairs
# The dark blocks on the plan are solid walls; about 40% of them are toilets
# (WC-M / WC-F on the left, WC-M2 / WC-F2 on the right).
#   letters = inside of a room (see ROOM_LETTERS)
# The ROBOT drives only in the halls (. ~ S). Students also use doors, rooms, X.
# To change the map, edit tools/gen_map.py and paste its output here.
# ---------------------------------------------------------------------------
FLOOR_MAP = [
    "                                              #############           ",
    "                                  ############eeeeeeeeeee#######      ",
    "                            #######ddddddd#eeeeeeeeeeeeee#ffffff##    ",
    "                       #####cccccc#dd##################ee#ffffffff#   ",
    "                  #####bb#.#cccccc#d#.##################e#fffffffff#  ",
    "             #####bbbbbbb+.#cc#####+#.##################+#fffffffff#  ",
    "         #####aaa#########.#+#........#################..#fffffffff#  ",
    "     ####a###############.............#################..#fffffffff#  ",
    "  #####t#+.##############................................#fffffffff#  ",
    "  ##tttt+..##############................................#fffffffff#  ",
    " ###tttt#..##############..........#################.....#fffffffff#  ",
    " ###tttt#..##############..........#################.....+fffffffff#  ",
    " #uu#####..##############..........#################.....#fffffffff#  ",
    "#uuu#vvv+..##############..........#################.....#fffff#fff#  ",
    "#uu######..##############...XXXX...#CCCCCCCC#DDDDDD#..###########ff#  ",
    "###w#xxx#..#AAAAAA#BBBBB#...XXXX...#CCCCCCCC#DDDDDD#..#gggggggg#####  ",
    "#www#xxx+..#AAAAAA#BBBBB#...XXXX...#CCCCCCCC#DDDDDD#..+ggggggggggg#   ",
    "#www#xxx#..#AAAAAA#BBBBB#...XXXX...####+#######+####..#ggggggggggg#   ",
    "#########..###+######+###...XXXX......................#ggggggggggg#   ",
    " #yy#zzz+...................XXXX~~~...................####ggggggg#    ",
    " #yy#####.###...............XXXX~~~###############....+hhh########    ",
    " ###rrrrr#rr+..##########...XXXX~~~#jjjjjjjjjjjjj#....#hhhhhhhhh#     ",
    " #rrrrrrrrrr#..#ssssssss#...XXXX~~~#jjjjjjjjjjjjj#....######hhhh#     ",
    " #rrrrrrrr###..#ssssssss#...XXXX~~~#jjjjjjjjjjjjj+....#iiiii#####     ",
    " ##rrr####qq#..#ssssssss#...XXXX~~~#jjjjjjjjjjjjj#....#iiiiiiii#      ",
    " #####qqqqqq+..+ssssssss#...XXXX~~~#jjjjjjjjjjjjj#....+iiiiiiii#      ",
    "  #qqqqqqqqq#..#ssssssss#...XXXX~~~##jjjjjjjjjjjj#....#iiiiiiii#      ",
    "  #qqqqqqq###..#sss######...XXXX~~~..#############....#iiiiiiii#      ",
    "  #qqq####pp#..####..................................#iiiiiiii#       ",
    "  ####pppppp#....................S...................#iiiiiiii#       ",
    "  #ppppppppp+.......................#+####...........#iiiiiiii#       ",
    "  #ppppppppp#......+#####+##########ll#kkk#+######...#i#iiiiii#       ",
    "  #pppppp###..#####nnn#mmmmmmmmmmmm#ll#kkkkkkkkkkk#######iiii#        ",
    "  ##p######o+#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkkkk###ii#         ",
    "   ####oo#ooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkkkkkkk##          ",
    "   #o#ooooooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkkkkkkkk#          ",
    "   #ooooooooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkkkkkkk#           ",
    "   #ooooooooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkkkkkk#            ",
    "    #oooooooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkkkkk#             ",
    "     ##oooooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkkkk#              ",
    "       #ooooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkkkk##               ",
    "        ##ooo#nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkkkkkk##                 ",
    "          ####nnnnnnnn#mmmmmmmmmmmm#ll#kkkkkkkk####                   ",
    "              #################################                       ",
    "                                                                      ",
]

ROOM_LETTERS = {
    "a": "214", "b": "215", "c": "216", "d": "217", "e": "218", "f": "219",
    "g": "220", "h": "221", "i": "222", "j": "223", "k": "224", "l": "225",
    "m": "226", "n": "200", "o": "201", "p": "202", "q": "203", "r": "207",
    "s": "204", "t": "213", "u": "212", "v": "209", "w": "211", "x": "210",
    "y": "208", "z": "206",
    "A": "WC-M", "B": "WC-F", "C": "WC-M2", "D": "WC-F2",
}
ROOM_TITLES = {"WC-M": "WC ♂", "WC-F": "WC ♀", "WC-M2": "WC ♂", "WC-F2": "WC ♀"}


def is_toilet(name):
    return name.startswith("WC")

HALL_CHARS = set(".~S")    # the robot only drives on these cells

# Cost of ENTERING a cell (= how many "robot steps" it takes).
BUSY_COST = 3
SECONDS_PER_CELL = 2.5        # robot cat speed: 1.6 m cells at ~0.65 m/s
STUDENT_SECONDS_PER_CELL = 1.3  # students walk faster (~1.2 m/s)


def hhmm_to_sec(s):
    h, m = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60


def sec_to_hhmm(t, seconds=False):
    t = int(t)
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if seconds else f"{h:02d}:{m:02d}"


class Campus:
    """Grid world. Cells are (row, col) tuples."""

    def __init__(self, floor_map=FLOOR_MAP):
        widths = {len(r) for r in floor_map}
        assert len(widths) == 1, f"map rows have different widths: {widths}"
        self.grid = floor_map
        self.H, self.W = len(floor_map), len(floor_map[0])
        self.rooms = {}
        self.dock = None
        self.exits, self.lounge = [], []

        for r, row in enumerate(floor_map):
            for c, ch in enumerate(row):
                if ch in ROOM_LETTERS:
                    name = ROOM_LETTERS[ch]
                    self.rooms.setdefault(name, {"letter": ch, "cells": [], "door": None})
                    self.rooms[name]["cells"].append((r, c))
                elif ch == "S":
                    self.dock = (r, c)
                elif ch == "X":
                    self.exits.append((r, c))
                elif ch == "~":
                    self.lounge.append((r, c))

        # A door belongs to the room whose cell touches it.
        for r, row in enumerate(floor_map):
            for c, ch in enumerate(row):
                if ch == "+":
                    for nr, nc in self._around((r, c)):
                        letter = floor_map[nr][nc]
                        if letter in ROOM_LETTERS:
                            self.rooms[ROOM_LETTERS[letter]]["door"] = (r, c)

        # rooms without a corridor door on the plan (208, 211, 212) are offices:
        # shown on the map, but not used for lectures or by the robot
        self.offices = {n: r for n, r in self.rooms.items() if r["door"] is None}
        self.toilets = sorted(n for n, r in self.rooms.items() if is_toilet(n) and r["door"])
        self.rooms = {n: r for n, r in self.rooms.items() if r["door"] is not None}
        for name, room in list(self.rooms.items()) + list(self.offices.items()):
            rs = [p[0] for p in room["cells"]]
            cs = [p[1] for p in room["cells"]]
            room["center"] = ((min(rs) + max(rs)) / 2, (min(cs) + max(cs)) / 2)
            if room["door"] is None:
                continue
            # meeting spot = the hall cell touching the door (robot never enters rooms)
            room["spot"] = next(n for n in self._around(room["door"]) if self.is_hall(n))
            # where students stand and chat when they come out on a break
            room["gather"] = self.near_hall_cells(room["spot"], 3)

    def _around(self, cell):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W:
                yield nr, nc

    def char(self, cell):
        return self.grid[cell[0]][cell[1]]

    def passable(self, cell):
        return self.char(cell) not in "# V"

    @property
    def entrance(self):
        """Top of the stairs (students leave the floor here)."""
        return self.exits[0]

    @property
    def exit_spot(self):
        """Hall cell next to the stairs: where the robot can guide people."""
        return next(n for n in self._around(self.entrance) if self.is_hall(n))

    def cost(self, cell):
        """Cost of stepping INTO this cell."""
        return BUSY_COST if self.char(cell) == "~" else 1

    def is_hall(self, cell):
        """Corridor / hall cells: the only cells the robot is allowed to use."""
        return self.char(cell) in HALL_CHARS

    def neighbors(self, cell, halls_only=False):
        for n in self._around(cell):
            if self.passable(n) and (not halls_only or self.is_hall(n)):
                yield n

    def near_hall_cells(self, cell, radius):
        r0, c0 = cell
        return [(r, c) for r in range(self.H) for c in range(self.W)
                if abs(r - r0) + abs(c - c0) <= radius and self.is_hall((r, c))
                and (r, c) != self.dock]

    def spot(self, room_name):
        """Hall cell right outside the room's door: where the robot waits."""
        return self.rooms[room_name]["spot"]

    def door(self, room_name):
        return self.rooms[room_name]["door"]

    def room_of_cell(self, cell):
        return ROOM_LETTERS.get(self.char(cell))


# ---------------------------------------------------------------------------
# Timetable
# ---------------------------------------------------------------------------
@dataclass
class Lecture:
    id: int
    start: int          # seconds since midnight
    end: int
    room: str
    group: str
    course: str
    kind: str           # lecture / practice
    size: int           # number of students
    late_teacher: bool  # teacher usually finishes a few minutes late

    def label(self):
        return f"{self.course} ({self.kind}) | {self.group} | room {self.room}"


class Timetable:
    def __init__(self, timetable_csv=None, deadlines_csv=None, today=TODAY):
        timetable_csv = timetable_csv or os.path.join(DATA_DIR, "timetable.csv")
        deadlines_csv = deadlines_csv or os.path.join(DATA_DIR, "deadlines.csv")
        self.today = today
        self.lectures = []
        with open(timetable_csv, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                self.lectures.append(Lecture(
                    id=i,
                    start=hhmm_to_sec(row["start"]),
                    end=hhmm_to_sec(row["end"]),
                    room=row["room"].strip(),
                    group=row["group"].strip(),
                    course=row["course"].strip(),
                    kind=row["type"].strip(),
                    size=int(row["group_size"]),
                    late_teacher=row["late_teacher"].strip() == "1",
                ))
        self.lectures.sort(key=lambda l: (l.start, l.room))

        self.deadlines = []
        with open(deadlines_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                y, m, d = map(int, row["date"].split("-"))
                self.deadlines.append({
                    "group": row["group"].strip(), "course": row["course"].strip(),
                    "kind": row["kind"].strip(), "date": date(y, m, d),
                })

    # --- simple queries ---------------------------------------------------
    @property
    def groups(self):
        return sorted({l.group for l in self.lectures})

    @property
    def end_times(self):
        return sorted({l.end for l in self.lectures})

    def ending_at(self, t):
        return [l for l in self.lectures if l.end == t]

    def of_group(self, group):
        return [l for l in self.lectures if l.group == group]

    def next_of_group(self, group, after_t):
        later = [l for l in self.of_group(group) if l.start >= after_t]
        return later[0] if later else None

    def is_last_of_day(self, lec):
        return self.next_of_group(lec.group, lec.end) is None

    def gap_after(self, lec):
        """Minutes until this group's next class (None if last class)."""
        nxt = self.next_of_group(lec.group, lec.end)
        return None if nxt is None else (nxt.start - lec.end) // 60

    def next_deadline(self, group):
        upcoming = [d for d in self.deadlines
                    if d["group"] == group and d["date"] >= self.today]
        if not upcoming:
            return None
        d = min(upcoming, key=lambda x: x["date"])
        return {**d, "days": (d["date"] - self.today).days}


if __name__ == "__main__":
    campus = Campus()
    tt = Timetable()
    print(f"Map {campus.H}x{campus.W}, dock={campus.dock}, stairs={campus.entrance}")
    for name, room in campus.rooms.items():
        print(f"  {name:10s} door={room['door']} cells={len(room['cells'])}")
    print(f"{len(tt.lectures)} lectures, groups: {tt.groups}")
    for t in tt.end_times:
        print(sec_to_hhmm(t), [(l.room, l.group) for l in tt.ending_at(t)])
