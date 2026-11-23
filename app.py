"""
app.py - visual simulation of MysyqBot (matplotlib window).

Run:   python app.py                  (needs: pip install matplotlib)
       python app.py --seed 3         (different random day)
       python app.py --gif demo.gif   (record the first break to a GIF, no window)

Window controls:
  * type a question in the "Ask MysyqBot" box and press Enter
      e.g.  where is 216 | where is the WC | when is my next class AIN-2601 | what is A*?
  * Pause / Slower / Faster / Next break buttons
The right panel shows every algorithm's input -> steps -> output live.
"""
import argparse
import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Button, TextBox

from campus import ROOM_LETTERS, ROOM_TITLES, sec_to_hhmm
from world import GROUP_COLORS, REACH, Simulation

CELL_COLORS = {
    "#": "#3d405b", ".": "#f4f1de", "~": "#f2cc8f", "+": "#b08968",
    "S": "#81b29a", "X": "#7aa6c2", " ": "#ffffff", "V": "#b9c4d6",
}
ROOM_COLORS = {"WC-M": "#bfe6e0", "WC-F": "#bfe6e0", "WC-M2": "#bfe6e0", "WC-F2": "#bfe6e0"}
DEFAULT_ROOM = "#dfe7f1"
LOG_LINES = 46
LOG_WIDTH = 74


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


class App:
    def __init__(self, sim, interactive=True):
        self.sim = sim
        self.speed = 3.0          # simulated seconds per frame
        self.paused = False
        self.interactive = interactive
        self.fig = plt.figure(figsize=(16, 7.6), facecolor="white")
        try:
            self.fig.canvas.manager.set_window_title("MysyqBot - QAIRU campus robot simulation")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.01, 0.20, 0.60, 0.70])
        self.ax_log = self.fig.add_axes([0.625, 0.03, 0.365, 0.94])
        self._draw_map()
        self._draw_log_panel()
        if interactive:
            self._make_widgets()

    # ------------------------------------------------------------ drawing
    def _draw_map(self):
        c = self.sim.campus
        img = []
        for r in range(c.H):
            row = []
            for col in range(c.W):
                ch = c.grid[r][col]
                if ch in ROOM_LETTERS:
                    row.append(hex_rgb(ROOM_COLORS.get(ROOM_LETTERS[ch], DEFAULT_ROOM)))
                else:
                    row.append(hex_rgb(CELL_COLORS[ch]))
            img.append(row)
        ax = self.ax
        ax.imshow(img, extent=(-0.5, c.W - 0.5, c.H - 0.5, -0.5), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        self.room_text, self.room_wide = {}, {}
        for name, room in c.rooms.items():
            r, col = room["center"]
            cols = [p[1] for p in room["cells"]]
            rows = [p[0] for p in room["cells"]]
            self.room_wide[name] = max(cols) - min(cols) >= 8
            tall = max(rows) - min(rows) >= 2
            ax.text(col, r - (0.45 if tall else 0.25), ROOM_TITLES.get(name, name), ha="center",
                    va="center", fontsize=9, weight="bold", color="#3d405b", alpha=0.75, zorder=2)
            self.room_text[name] = ax.text(col, r + (0.6 if tall else 0.45), "", ha="center",
                                           va="center", fontsize=6.3, color="#3d405b", alpha=0.85, zorder=2)

        def centre(cells):
            return (sum(p[0] for p in cells) / len(cells), sum(p[1] for p in cells) / len(cells))
        # label every atrium (connected group of V cells), the stairs and the lounge
        atria, seen = [], set()
        for r in range(c.H):
            for col in range(c.W):
                if c.grid[r][col] == "V" and (r, col) not in seen:
                    comp, stack = [], [(r, col)]
                    seen.add((r, col))
                    while stack:
                        p = stack.pop()
                        comp.append(p)
                        for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
                            if 0 <= q[0] < c.H and 0 <= q[1] < c.W and q not in seen \
                                    and c.grid[q[0]][q[1]] == "V":
                                seen.add(q)
                                stack.append(q)
                    atria.append(comp)
        notes = [(centre(a), "atrium") for a in atria if len(a) > 20]
        notes.append((centre(c.exits), "stairs"))
        if c.lounge:
            notes.append((centre(c.lounge), "sofas"))
        for name, room in c.offices.items():
            r, col = room["center"]
            ax.text(col, r, name, ha="center", va="center", fontsize=7, color="#8d99ae")
        for (r, col), txt in notes:
            ax.text(col, r, txt, ha="center", va="center", fontsize=6.5, color="#1d3557",
                    alpha=0.8, rotation=90 if txt in ("stairs", "sofas") else 0)
        ax.text(c.dock[1], c.dock[0] + 1.0, "dock + info screen", ha="center", fontsize=6.5,
                color="#2d6a4f")
        ax.text(c.W / 2, -0.9, "QAIRU - Floor 2 (from the lobby screen plan)", ha="center", fontsize=8,
                color="#8d99ae", style="italic")

        self.plan_line, = ax.plot([], [], "--", color="#f28c28", lw=2.2, alpha=0.8, zorder=3)
        self.guide_line, = ax.plot([], [], ":", color="#2d6a4f", lw=3, alpha=0.9, zorder=3)
        self.students_sc = ax.scatter([], [], s=30, zorder=4, edgecolors="white", linewidths=0.4)

        # the area where the robot can talk to students
        self.reach = Circle((0, 0), REACH, color="#f28c28", alpha=0.13, zorder=3, visible=False)
        ax.add_patch(self.reach)

        # the robot cat
        self.cat_body = Circle((0, 0), 0.42, color="#f28c28", zorder=6, ec="#7f4f24", lw=1.2)
        self.cat_ears = [Polygon([[0, 0], [0, 0], [0, 0]], closed=True, color="#f28c28",
                                 ec="#7f4f24", lw=1.0, zorder=5) for _ in range(2)]
        ax.add_patch(self.cat_body)
        for e in self.cat_ears:
            ax.add_patch(e)
        self.cat_face = ax.text(0, 0, "^.^", ha="center", va="center", fontsize=7,
                                weight="bold", color="#3d2c1d", zorder=7)
        self.bubble = ax.text(0, 0, "", fontsize=9, zorder=8, va="bottom", wrap=False,
                              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#f28c28", lw=1.5))
        ax.set_xlim(-0.5, c.W - 0.5)          # keep the map size fixed
        ax.set_ylim(c.H - 0.5, -1.5)
        ax.set_autoscale_on(False)
        self.title = self.fig.text(0.01, 0.94, "", fontsize=15, weight="bold", color="#3d405b")
        self.status = self.fig.text(0.01, 0.165, "", fontsize=10, color="#3d405b")

        handles = [Line2D([], [], marker="o", ls="", color=col, label=g, markersize=7)
                   for g, col in GROUP_COLORS.items()]
        handles += [Line2D([], [], marker="o", ls="", color="#f28c28", alpha=0.3, markersize=12,
                           label="talk zone"),
                    Line2D([], [], ls="--", color="#f28c28", lw=2, label="tour (A*)"),
                    Line2D([], [], ls=":", color="#2d6a4f", lw=3, label="guide (A*)")]
        self.fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.095),
                        ncol=9, fontsize=8, frameon=False)

    def _draw_log_panel(self):
        ax = self.ax_log
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#1e1e2e")
        ax.text(0.02, 0.99, "MysyqBot brain  (input -> steps -> output)", transform=ax.transAxes,
                va="top", fontsize=10, weight="bold", color="#f2cc8f", family="monospace")
        self.log_text = ax.text(0.02, 0.965, "", transform=ax.transAxes, va="top",
                                fontsize=7.2, color="#e0e0e0", family="monospace")

    def _make_widgets(self):
        ax_box = self.fig.add_axes([0.075, 0.03, 0.21, 0.05])
        self.box = TextBox(ax_box, "Ask MysyqBot ", initial="")
        self.box.on_submit(self._on_ask)
        specs = [("Ask", self._ask_btn), ("Pause", self._pause), ("Slower", self._slower),
                 ("Faster", self._faster), ("Next break", self._next_break)]
        self.buttons = []
        x = 0.295
        for label, fn in specs:
            w = 0.07 if label == "Next break" else 0.046
            b = Button(self.fig.add_axes([x, 0.03, w, 0.05]), label,
                       color="#f4f1de", hovercolor="#f2cc8f")
            b.on_clicked(fn)
            self.buttons.append(b)
            x += w + 0.008

    # ----------------------------------------------------------- widgets
    def _on_ask(self, text):
        text = text.strip()
        if not text:
            return
        self.sim.ask(text)
        self.box.set_val("")

    def _ask_btn(self, _):
        self._on_ask(self.box.text)

    def _pause(self, _):
        self.paused = not self.paused
        self.buttons[1].label.set_text("Play" if self.paused else "Pause")

    def _slower(self, _):
        self.speed = max(0.5, self.speed / 2)

    def _faster(self, _):
        self.speed = min(48, self.speed * 2)

    def _next_break(self, _):
        self.sim.skip_to_next_break()

    # ------------------------------------------------------------ update
    def _set_cat(self, r, c):
        self.cat_body.center = (c, r)
        self.cat_ears[0].set_xy([[c - 0.42, r - 0.12], [c - 0.32, r - 0.72], [c - 0.05, r - 0.38]])
        self.cat_ears[1].set_xy([[c + 0.42, r - 0.12], [c + 0.32, r - 0.72], [c + 0.05, r - 0.38]])
        self.cat_face.set_position((c, r + 0.05))

    def update(self, _frame=None):
        sim = self.sim
        if not self.paused:
            nb = sim.next_break()
            fast = sim.is_idle() and nb is not None and nb - sim.t > 45
            sim.step(self.speed * (12 if fast else 1))

        # students
        if sim.students:
            self.students_sc.set_offsets([(s["pos"][1], s["pos"][0]) for s in sim.students])
            self.students_sc.set_color([GROUP_COLORS.get(s["group"], "gray") for s in sim.students])
            self.students_sc.set_edgecolors("white")
        else:
            self.students_sc.set_offsets([[float("nan"), float("nan")]])

        # who is in which room
        for name, txt in self.room_text.items():
            lec = next((l for l in sim.tt.lectures if l.room == name and l.start <= sim.t
                        < sim.truth[l.id]["actual_end"]), None)
            if lec:
                txt.set_text(f"{lec.group} - {lec.course}" if self.room_wide[name] else lec.group)
            else:
                n_in = sum(1 for s in sim.students if s["state"] == "inroom"
                           and sim.campus.room_of_cell(s["cell"]) == name)
                n_out = sim.hall_count(name)
                label = (f"break: {n_in} in, {n_out} out" if self.room_wide[name]
                         else f"{n_in} in/{n_out} out")
                txt.set_text(label if n_in or n_out else "")

        # robot + paths
        r, c = sim.robot["pos"]
        self._set_cat(r, c)
        self.reach.center = (c, r)
        self.reach.set_visible(sim.interacting)
        pp = sim.planned_path if sim.tour else []
        self.plan_line.set_data([p[1] for p in pp], [p[0] for p in pp])
        gp = sim.guide_path
        self.guide_line.set_data([p[1] for p in gp], [p[0] for p in gp])

        if sim.bubble:
            wrapped = "\n".join(textwrap.wrap(sim.bubble, 40))
            right = c > sim.campus.W * 0.55
            self.bubble.set_text(wrapped)
            self.bubble.set_position((c - 0.6 if right else c + 0.6, r - 0.7))
            self.bubble.set_ha("right" if right else "left")
            self.bubble.set_va("bottom" if r > 4 else "top")
            if r <= 4:
                self.bubble.set_position((c - 0.6 if right else c + 0.6, r + 0.8))
            self.bubble.set_visible(True)
        else:
            self.bubble.set_visible(False)

        # texts
        nb = sim.next_break()
        mode = sim.mode_label()
        if sim.is_idle() and nb:
            mode += f", next break {sec_to_hhmm(nb)}"
        self.title.set_text(f"MysyqBot @ QAIRU floor 2  |  Mon 16 Nov  {sec_to_hhmm(sim.t, True)}  |  {mode}")
        st = sim.stats
        self.status.set_text(
            f"breaks: {st['breaks']}   halls visited: {st['visits']}   empty: {st['empty']}   "
            f"students met: {st['reached']}   walked up: {st['came_by_themselves']}   "
            f"questions: {st['questions']}   speed: x{self.speed * 25:.0f}"
            + ("   [PAUSED]" if self.paused else ""))

        lines = []
        for ln in sim.log_lines[-LOG_LINES * 2:]:
            lines += textwrap.wrap(ln, LOG_WIDTH, subsequent_indent="      ") or [""]
        self.log_text.set_text("\n".join(lines[-LOG_LINES:]))
        return []


def main():
    ap = argparse.ArgumentParser(description="MysyqBot campus robot simulation")
    ap.add_argument("--seed", type=int, default=7, help="random seed for the day")
    ap.add_argument("--start", default="10:15", help="start time HH:MM")
    ap.add_argument("--gif", help="record the first break to this GIF file instead of a window")
    args = ap.parse_args()

    if args.gif:
        matplotlib.use("Agg")
        from matplotlib.animation import FuncAnimation, PillowWriter
        sim = Simulation(seed=args.seed, start=args.start)
        app = App(sim, interactive=False)
        app.speed = 4
        anim = FuncAnimation(app.fig, app.update, frames=190, interval=60, blit=False)
        anim.save(args.gif, writer=PillowWriter(fps=15), dpi=55)
        print("saved", args.gif)
        return

    from matplotlib.animation import FuncAnimation
    sim = Simulation(seed=args.seed, start=args.start)
    app = App(sim)
    app._anim = FuncAnimation(app.fig, app.update, interval=40, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
