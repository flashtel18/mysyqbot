"""
server3d.py - 3D view of MysyqBot on QAIRU floor 2.

Python runs the whole simulation and ALL the AI (Bayes, A*, hill climbing,
HMM, MDP, Naive Bayes) exactly like app.py / console_demo.py.
The browser page (web/index.html + web/viewer.js) only DRAWS the state in 3D
with plain WebGL - no internet and no extra libraries needed.

Run:   python server3d.py               -> opens http://localhost:8000
       python server3d.py --seed 3 --port 8080 --no-browser
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from campus import ROOM_TITLES, SECONDS_PER_CELL, is_toilet, sec_to_hhmm
from world import GROUP_COLORS, REACH, VISION_R, Simulation

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATE_CODES = {"seated": 0, "inroom": 1, "walking_out": 2, "hall": 3, "approach": 4, "leaving": 5}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".png": "image/png", ".ico": "image/x-icon"}


class App:
    def __init__(self, seed, start):
        self.lock = threading.Lock()
        self.seed, self.start = seed, start
        self.sim = Simulation(seed=seed, start=start)
        self.groups = list(GROUP_COLORS)

    def reset(self, seed=None):
        self.seed = self.seed if seed is None else int(seed)
        self.sim = Simulation(seed=self.seed, start=self.start)

    # ------------------------------------------------------------ payloads
    def map_payload(self):
        c = self.sim.campus

        def room_info(name, room, office):
            rows = [p[0] for p in room["cells"]]
            cols = [p[1] for p in room["cells"]]
            return {"center": room["center"], "door": room["door"], "letter": room["letter"],
                    "office": office, "size": len(room["cells"]),
                    "title": ROOM_TITLES.get(name, name), "toilet": is_toilet(name),
                    "bbox": [min(rows), max(rows), min(cols), max(cols)],
                    "spot": room.get("spot"), "gather": room.get("gather", [])}

        rooms = {n: room_info(n, r, False) for n, r in c.rooms.items()}
        rooms.update({n: room_info(n, r, True) for n, r in c.offices.items()})
        return {"W": c.W, "H": c.H, "grid": c.grid, "rooms": rooms, "dock": c.dock,
                "exits": c.exits, "groups": GROUP_COLORS, "group_order": self.groups,
                "reach": REACH, "vision_r": VISION_R, "seconds_per_cell": SECONDS_PER_CELL}

    def room_labels(self):
        sim = self.sim
        labels = {}
        for name in sim.campus.rooms:
            lec = next((l for l in sim.tt.lectures if l.room == name and l.start <= sim.t
                        < sim.truth[l.id]["actual_end"]), None)
            if lec:
                labels[name] = f"{lec.group} · {lec.course}"
            else:
                n_in = sum(1 for s in sim.students if s["state"] == "inroom"
                           and sim.campus.room_of_cell(s["cell"]) == name)
                n_out = sim.hall_count(name)
                if n_in or n_out:
                    labels[name] = f"break: {n_in} inside, {n_out} in hall"
        return labels

    def state_payload(self, since, known_astar=-1):
        sim = self.sim
        nb = sim.next_break()
        mode = sim.mode_label()
        if sim.is_idle() and nb:
            mode += f" · next break {sec_to_hhmm(nb)}"
        gi = {g: i for i, g in enumerate(self.groups)}
        ex = dict(sim.explain)
        ex["bayes"] = {"zones": sim._belief_view(), "last": sim.explain["bayes"].get("last")}
        vis = []
        for r, c in sim.visible:
            vis += (r, c)
        payload = {
            "t": sim.t, "clock": sec_to_hhmm(sim.t, True), "mode": mode,
            "next_break": nb, "idle": sim.is_idle(),
            "robot": {"r": sim.robot["pos"][0], "c": sim.robot["pos"][1],
                      "interacting": sim.interacting},
            "students": [[s["id"], round(s["pos"][0], 3), round(s["pos"][1], 3),
                          gi.get(s["group"], 0), STATE_CODES.get(s["state"], 0)]
                         for s in sim.students],
            "seen": list(sim.seen),
            "vision": vis,
            "bubble": sim.bubble,
            "student_bubble": sim.student_bubble if sim.student_bubble
            and sim.t < sim.student_bubble["until"] else None,
            "chat": sim.chat[-6:],
            "plan": sim.planned_path if sim.tour else [], "guide": sim.guide_path,
            "tour": [{"room": r, "n": i + 1} for i, r in enumerate(sim.tour)],
            "labels": self.room_labels(),
            "stats": sim.stats,
            "explain": ex,
            "astar_id": sim.astar_viz["id"],
            "log": sim.log_lines[since:], "log_total": len(sim.log_lines),
        }
        if sim.astar_viz["id"] != known_astar:
            payload["astar_cells"] = sim.astar_viz["cells"]
        return payload

APP = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/api/map":
            with APP.lock:
                return self._send(200, APP.map_payload())
        if url.path == "/api/state":
            dt = min(max(float(q.get("dt", ["0"])[0]), 0.0), 900.0)
            since = int(q.get("since", ["0"])[0])
            known = int(q.get("av", ["-1"])[0])
            with APP.lock:
                if dt:
                    APP.sim.step(dt)
                return self._send(200, APP.state_payload(since, known))
        path = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
        full = os.path.normpath(os.path.join(WEB_DIR, path))
        if not full.startswith(WEB_DIR) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        with open(full, "rb") as f:
            self._send(200, f.read(), CONTENT_TYPES.get(os.path.splitext(full)[1], "text/plain"))

    def do_POST(self):
        url = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        with APP.lock:
            if url.path == "/api/ask":
                text = str(body.get("text", "")).strip()[:200]
                if not text:
                    return self._send(400, {"error": "empty question"})
                res = APP.sim.ask(text)
                return self._send(200, {"intent": res["intent"], "reply": res["reply"]})
            if url.path == "/api/control":
                action = body.get("action")
                if action == "next_break":
                    APP.sim.skip_to_next_break()
                elif action == "reset":
                    APP.reset(body.get("seed"))
                    return self._send(200, {"ok": True, "map": APP.map_payload()})
                return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})


def main():
    global APP
    ap = argparse.ArgumentParser(description="MysyqBot 3D simulation server")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", default="10:15")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    APP = App(args.seed, args.start)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"MysyqBot 3D is running at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye!")


if __name__ == "__main__":
    main()
