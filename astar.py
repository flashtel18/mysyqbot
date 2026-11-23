"""
astar.py - ALGORITHM 1: A* search (course topic: Week 2, heuristic search).

Problem:  move the robot from cell `start` to cell `goal` on the floor map
          as fast as possible. Corridor cells cost 1 step, the busy lobby
          costs 3 steps (crowd slows the robot down), walls are blocked.

A* keeps a priority queue ordered by  f(n) = g(n) + h(n)
    g(n) = real cost from start to n
    h(n) = Manhattan distance from n to goal (never overestimates, because
           every step costs at least 1  ->  admissible  ->  A* is optimal)

BFS is included only as a BASELINE for comparison: BFS finds the path with
the fewest cells but ignores costs, so it happily walks through the slow lobby.
"""
import heapq
from collections import deque


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    return path[::-1]


def astar(campus, start, goal, halls_only=False, trace=None):
    """Returns (path, cost, nodes_expanded). path includes start and goal.
    halls_only=True -> the robot may not enter rooms or doorways.
    trace: optional list - every expanded cell is appended (for visualisation)."""
    if start == goal:
        return [start], 0, 0
    open_heap = [(manhattan(start, goal), 0, start)]
    g = {start: 0}
    came_from = {}
    closed = set()
    expanded = 0
    while open_heap:
        f, g_n, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if trace is not None:
            trace.append(node)
        if node == goal:
            return reconstruct(came_from, node), g_n, expanded
        for nb in campus.neighbors(node, halls_only):
            new_g = g_n + campus.cost(nb)
            if new_g < g.get(nb, float("inf")):
                g[nb] = new_g
                came_from[nb] = node
                heapq.heappush(open_heap, (new_g + manhattan(nb, goal), new_g, nb))
    return None, float("inf"), expanded


def bfs(campus, start, goal, halls_only=False):
    """Baseline: fewest cells, ignores cell costs. Returns (path, cost, expanded)."""
    queue = deque([start])
    came_from = {}
    seen = {start}
    expanded = 0
    while queue:
        node = queue.popleft()
        expanded += 1
        if node == goal:
            path = reconstruct(came_from, node)
            return path, path_cost(campus, path), expanded
        for nb in campus.neighbors(node, halls_only):
            if nb not in seen:
                seen.add(nb)
                came_from[nb] = node
                queue.append(nb)
    return None, float("inf"), expanded


def path_cost(campus, path):
    return sum(campus.cost(c) for c in path[1:])


def describe_path(path):
    """Turn a cell path into short human directions: 'right 6, down 3, ...'."""
    if not path or len(path) < 2:
        return "you are already here"
    names = {(0, 1): "right", (0, -1): "left", (1, 0): "down", (-1, 0): "up"}
    steps = []
    for a, b in zip(path, path[1:]):
        d = names[(b[0] - a[0], b[1] - a[1])]
        if steps and steps[-1][0] == d:
            steps[-1][1] += 1
        else:
            steps.append([d, 1])
    return ", ".join(f"{d} {n}" for d, n in steps)


if __name__ == "__main__":
    from campus import Campus, SECONDS_PER_CELL
    campus = Campus()
    s, g = campus.dock, campus.spot("219")
    p1, c1, e1 = astar(campus, s, g, halls_only=True)
    p2, c2, e2 = bfs(campus, s, g, halls_only=True)
    print(f"A*  : cost={c1} ({c1 * SECONDS_PER_CELL}s) cells={len(p1)} expanded={e1}")
    print(f"BFS : cost={c2} ({c2 * SECONDS_PER_CELL}s) cells={len(p2)} expanded={e2}")
    print("A* directions:", describe_path(p1))
