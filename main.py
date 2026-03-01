import tkinter as tk
from tkinter import messagebox
import heapq
import math
import random
import time
from collections import deque

BG        = "#E8E8E8"
PANEL     = "#D8D8D8"
BORDER    = "#B8B8B8"
TEXT      = "#1A1A1A"
MUTED     = "#707070"
BTN_BG    = "#C8C8C8"
BTN_HOV   = "#B0B0B0"
SEL_BG    = "#4A90D9"

CELL_EMPTY    = "#F0F0F0"
CELL_WALL     = "#3A3A3A"
CELL_DYN      = "#C0392B"
CELL_START    = "#27AE60"
CELL_GOAL     = "#E74C3C"
CELL_FRONTIER = "#F9E04B"
CELL_VISITED  = "#85C1E9"
CELL_PATH     = "#E67E22"
CELL_AGENT    = "#2C3E50"
GRID_LINE     = "#D0D0D0"

SIDEBAR_W = 152
GRID_COLS = 22
GRID_ROWS = 24
CELL_SIZE = 23


def manhattan(a, b):
    row_diff = abs(a[0] - b[0])
    col_diff = abs(a[1] - b[1])
    return row_diff + col_diff


def euclidean(a, b):
    row_sq = (a[0] - b[0]) ** 2
    col_sq = (a[1] - b[1]) ** 2
    return math.sqrt(row_sq + col_sq)


def build_path(came_from, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def get_neighbours(grid, rows, cols, r, c):
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        in_bounds = 0 <= nr < rows and 0 <= nc < cols
        if in_bounds and grid[nr][nc] != 1:
            result.append((nr, nc))
    return result


def gbfs(grid, rows, cols, start, goal, hfn):
    heap           = []
    came_from      = {start: None}
    visited        = set()
    order          = []
    frontier       = {start}
    frontier_steps = []

    heapq.heappush(heap, (hfn(start, goal), start))

    while heap:
        _, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        frontier.discard(cur)
        visited.add(cur)
        order.append(cur)
        frontier_steps.append(frozenset(frontier))
        if cur == goal:
            return build_path(came_from, goal), order, frontier_steps
        r, c = cur
        for nb in get_neighbours(grid, rows, cols, r, c):
            if nb not in visited and nb not in came_from:
                came_from[nb] = cur
                frontier.add(nb)
                heapq.heappush(heap, (hfn(nb, goal), nb))

    return None, order, frontier_steps


def astar(grid, rows, cols, start, goal, hfn):
    heap           = []
    g_cost         = {start: 0}
    came_from      = {start: None}
    closed         = set()
    order          = []
    frontier       = {start}
    frontier_steps = []

    heapq.heappush(heap, (hfn(start, goal), 0, start))

    while heap:
        f, g, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        frontier.discard(cur)
        closed.add(cur)
        order.append(cur)
        frontier_steps.append(frozenset(frontier))
        if cur == goal:
            return build_path(came_from, goal), order, frontier_steps
        r, c = cur
        for nb in get_neighbours(grid, rows, cols, r, c):
            new_g  = g + 1
            better = nb not in g_cost or new_g < g_cost[nb]
            if nb not in closed and better:
                g_cost[nb]    = new_g
                came_from[nb] = cur
                frontier.add(nb)
                heapq.heappush(heap, (new_g + hfn(nb, goal), new_g, nb))

    return None, order, frontier_steps


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Pathfinding Agent")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.R  = GRID_ROWS
        self.C  = GRID_COLS
        self.CS = CELL_SIZE

        self.grid  = [[0] * self.C for _ in range(self.R)]
        self.start = (1, 1)
        self.goal  = (self.R - 2, self.C - 2)

        self.path           = []
        self.order          = []
        self.frontier_steps = []
        self.visited_set    = set()
        self.frontier_set   = set()
        self.path_set       = set()

        self.running = False
        self.phase   = "idle"
        self.vi      = 0
        self.pi      = 0
        self.agent   = None
        self.agent_i = 0

        self.total_nodes = 0
        self.replans     = 0

        self.var_algo = tk.StringVar(value="astar")
        self.var_heu  = tk.StringVar(value="manhattan")
        self.var_mode = tk.StringVar(value="wall")
        self.var_dyn  = tk.BooleanVar(value=False)
        self.var_rows = tk.StringVar(value=str(self.R))
        self.var_cols = tk.StringVar(value=str(self.C))

        self.lbl_nodes   = tk.StringVar(value="--")
        self.lbl_cost    = tk.StringVar(value="--")
        self.lbl_time    = tk.StringVar(value="--")
        self.lbl_replans = tk.StringVar(value="--")
        self.lbl_status  = tk.StringVar(value="Draw walls then click Run.")

        self.drag_mode     = None
        self.canvas        = None
        self.canvas_frame  = None
        self.canvas_parent = None

        self.build_ui()
        self.draw_all()

    def build_ui(self):
        self.build_header()
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        self.build_sidebar(body)
        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")
        self.canvas_parent = body
        self.build_canvas(body)
        self.build_metrics_bar()

    def build_header(self):
        hdr = tk.Frame(self.root, bg=PANEL, height=34)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text="PATHFINDING AGENT",
            font=("Helvetica", 10, "bold"),
            bg=PANEL,
            fg=TEXT
        ).pack(side="left", padx=10, pady=6)
        tk.Label(
            hdr,
            textvariable=self.lbl_status,
            font=("Helvetica", 8, "italic"),
            bg=PANEL,
            fg=MUTED
        ).pack(side="right", padx=10)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

    def build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=PANEL, width=SIDEBAR_W)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        run_stop = tk.Frame(sb, bg=PANEL)
        run_stop.pack(fill="x", padx=6, pady=(6, 3))
        tk.Button(
            run_stop,
            text="RUN",
            command=self.run,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 9, "bold"),
            relief="flat",
            cursor="hand2",
            pady=4
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(
            run_stop,
            text="STOP",
            command=self.stop,
            bg="#E53935",
            fg="white",
            font=("Helvetica", 9, "bold"),
            relief="flat",
            cursor="hand2",
            pady=4
        ).pack(side="left", expand=True, fill="x")

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", pady=(2, 0))

        self.sec(sb, "ALGORITHM")
        r = self.row(sb)
        self.rbtn(r, "A*",   self.var_algo, "astar")
        self.rbtn(r, "GBFS", self.var_algo, "gbfs")

        self.sec(sb, "HEURISTIC")
        r = self.row(sb)
        self.rbtn(r, "Manh.", self.var_heu, "manhattan")
        self.rbtn(r, "Eucl.", self.var_heu, "euclidean")

        self.sec(sb, "DRAW MODE")
        r = self.row(sb)
        self.rbtn(r, "Wall",  self.var_mode, "wall")
        self.rbtn(r, "Erase", self.var_mode, "erase")
        r = self.row(sb)
        self.rbtn(r, "Start", self.var_mode, "start")
        self.rbtn(r, "Goal",  self.var_mode, "goal")

        self.sec(sb, "DYNAMIC OBSTACLES")
        tk.Checkbutton(
            sb,
            text="Enable",
            variable=self.var_dyn,
            bg=PANEL, fg=TEXT,
            font=("Helvetica", 9),
            selectcolor=BG,
            activebackground=PANEL,
            cursor="hand2",
            anchor="w"
        ).pack(fill="x", padx=8, pady=1)

        self.sec(sb, "MAP")
        r = self.row(sb)
        tk.Button(
            r, text="Gen Maze", command=self.random_maze,
            bg=BTN_BG, fg=TEXT, font=("Helvetica", 8),
            relief="flat", cursor="hand2", pady=3
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(
            r, text="Clear", command=self.clear_all,
            bg=BTN_BG, fg=TEXT, font=("Helvetica", 8),
            relief="flat", cursor="hand2", pady=3
        ).pack(side="left", expand=True, fill="x")

        self.sec(sb, "RESIZE GRID")
        r = self.row(sb)
        tk.Entry(r, textvariable=self.var_rows, width=3,
                 font=("Helvetica", 8), bg=BG, relief="flat",
                 justify="center").pack(side="left")
        tk.Label(r, text="x", bg=PANEL, fg=TEXT,
                 font=("Helvetica", 8)).pack(side="left", padx=2)
        tk.Entry(r, textvariable=self.var_cols, width=3,
                 font=("Helvetica", 8), bg=BG, relief="flat",
                 justify="center").pack(side="left")
        tk.Button(
            r, text="Apply", command=self.resize_grid,
            bg=BTN_BG, fg=TEXT, font=("Helvetica", 8),
            relief="flat", cursor="hand2", padx=4, pady=2
        ).pack(side="left", padx=(4, 0))

        self.sec(sb, "LEGEND")
        legend_items = [
            (CELL_START,    "Start"),
            (CELL_GOAL,     "Goal"),
            (CELL_WALL,     "Wall"),
            (CELL_FRONTIER, "Frontier"),
            (CELL_VISITED,  "Visited"),
            (CELL_PATH,     "Path"),
            (CELL_DYN,      "Dyn.Wall"),
        ]
        for i in range(0, len(legend_items), 2):
            pair = tk.Frame(sb, bg=PANEL)
            pair.pack(fill="x", padx=8, pady=1)
            for color, label in legend_items[i:i+2]:
                item = tk.Frame(pair, bg=PANEL)
                item.pack(side="left", expand=True, anchor="w")
                tk.Frame(item, bg=color, width=9, height=9).pack(side="left")
                tk.Label(
                    item, text=" " + label,
                    bg=PANEL, fg=TEXT, font=("Helvetica", 7)
                ).pack(side="left")

    def sec(self, parent, title):
        tk.Label(
            parent,
            text=title,
            font=("Helvetica", 7, "bold"),
            bg=PANEL,
            fg=MUTED
        ).pack(anchor="w", padx=8, pady=(5, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8)

    def row(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=6, pady=2)
        return f

    def rbtn(self, parent, text, var, val):
        tk.Radiobutton(
            parent,
            text=text,
            variable=var,
            value=val,
            bg=PANEL, fg=TEXT,
            selectcolor=SEL_BG,
            activebackground=PANEL,
            font=("Helvetica", 8),
            indicatoron=0,
            relief="flat",
            padx=4, pady=3,
            cursor="hand2"
        ).pack(side="left", expand=True, fill="x", padx=(0, 1))

    def build_canvas(self, parent):
        self.canvas_frame = tk.Frame(parent, bg=BG)
        self.canvas_frame.pack(side="left", padx=6, pady=6)
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg=CELL_EMPTY,
            highlightthickness=1,
            highlightbackground=BORDER,
            width=self.C * self.CS,
            height=self.R * self.CS
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self.on_click)
        self.canvas.bind("<B1-Motion>",       self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def build_metrics_bar(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")
        bar = tk.Frame(self.root, bg=PANEL, height=40)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        metrics = [
            ("Nodes Visited", self.lbl_nodes),
            ("Path Cost",     self.lbl_cost),
            ("Time (ms)",     self.lbl_time),
            ("Re-plans",      self.lbl_replans),
        ]
        for label, var in metrics:
            blk = tk.Frame(bar, bg=PANEL)
            blk.pack(side="left", padx=18, pady=4)
            tk.Label(blk, text=label,
                     font=("Helvetica", 7), bg=PANEL, fg=MUTED).pack()
            tk.Label(blk, textvariable=var,
                     font=("Helvetica", 10, "bold"), bg=PANEL, fg=TEXT).pack()

    def get_color(self, r, c):
        pos = (r, c)
        if pos == self.agent and self.phase == "move":
            return CELL_AGENT
        if pos == self.start:
            return CELL_START
        if pos == self.goal:
            return CELL_GOAL
        if pos in self.path_set:
            return CELL_PATH
        if pos in self.visited_set:
            return CELL_VISITED
        if pos in self.frontier_set:
            return CELL_FRONTIER
        if self.grid[r][c] == 2:
            return CELL_DYN
        if self.grid[r][c] == 1:
            return CELL_WALL
        return CELL_EMPTY

    def paint(self, r, c):
        cs    = self.CS
        x1    = c * cs + 1
        y1    = r * cs + 1
        x2    = x1 + cs - 2
        y2    = y1 + cs - 2
        tag   = f"c{r},{c}"
        color = self.get_color(r, c)
        self.canvas.delete(tag)
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=color,
            outline=GRID_LINE,
            tags=tag
        )
        if (r, c) == self.agent and self.phase == "move":
            cx  = c * cs + cs // 2
            cy  = r * cs + cs // 2
            rad = cs // 4
            self.canvas.create_oval(
                cx - rad, cy - rad,
                cx + rad, cy + rad,
                fill=CELL_EMPTY,
                outline="",
                tags=f"ag{r},{c}"
            )

    def draw_all(self):
        self.canvas.delete("all")
        for r in range(self.R):
            for c in range(self.C):
                self.paint(r, c)

    def repaint(self, cells):
        for r, c in cells:
            if 0 <= r < self.R and 0 <= c < self.C:
                self.paint(r, c)

    def pixel_to_cell(self, px, py):
        r = py // self.CS
        c = px // self.CS
        if 0 <= r < self.R and 0 <= c < self.C:
            return r, c
        return None

    def on_click(self, event):
        cell = self.pixel_to_cell(event.x, event.y)
        if cell is not None:
            self.drag_mode = self.var_mode.get()
            self.edit_cell(cell)

    def on_drag(self, event):
        cell     = self.pixel_to_cell(event.x, event.y)
        can_drag = cell is not None and self.drag_mode in ("wall", "erase")
        if can_drag:
            self.edit_cell(cell)

    def on_release(self, event):
        self.drag_mode = None

    def edit_cell(self, cell):
        r, c    = cell
        mode    = self.drag_mode or self.var_mode.get()
        changed = []
        if mode == "wall" and cell not in (self.start, self.goal):
            self.grid[r][c] = 1
            changed.append(cell)
        elif mode == "erase":
            self.grid[r][c] = 0
            changed.append(cell)
        elif mode == "start":
            old        = self.start
            self.start = cell
            self.grid[r][c] = 0
            changed.append(old)
            changed.append(cell)
        elif mode == "goal":
            old       = self.goal
            self.goal = cell
            self.grid[r][c] = 0
            changed.append(old)
            changed.append(cell)
        self.repaint(changed)

    def run(self):
        self.stop()
        self.clear_path(keep_walls=True)
        self.total_nodes = 0
        self.replans     = 0
        self.search_from(self.start)

    def search_from(self, src):
        hfn = manhattan if self.var_heu.get()  == "manhattan" else euclidean
        alg = astar     if self.var_algo.get() == "astar"     else gbfs

        t0 = time.time()
        path, order, frontier_steps = alg(self.grid, self.R, self.C, src, self.goal, hfn)
        elapsed = (time.time() - t0) * 1000

        self.total_nodes   += len(order)
        self.path           = path or []
        self.order          = order
        self.frontier_steps = frontier_steps

        cost = len(self.path) - 1 if self.path else 0

        self.lbl_nodes.set(str(self.total_nodes))
        self.lbl_cost.set(str(cost))
        self.lbl_time.set(f"{elapsed:.1f}")
        self.lbl_replans.set(str(self.replans))

        if path:
            self.lbl_status.set("Path found — agent moving.")
            self.start_animation(src)
        else:
            self.lbl_status.set("No path found.")

    def start_animation(self, src):
        self.running      = True
        self.phase        = "visit"
        self.vi           = 0
        self.pi           = 0
        self.agent        = src
        self.agent_i      = 0
        self.visited_set  = set()
        self.frontier_set = set()
        self.path_set     = set()
        self.tick()

    def tick(self):
        if not self.running:
            return
        changed = []
        if self.phase == "visit":
            changed = self.tick_visit()
        elif self.phase == "path":
            changed = self.tick_path()
        elif self.phase == "move":
            done = self.tick_move(changed)
            if done:
                return
        self.repaint(changed)
        self.root.after(30, self.tick)

    def tick_visit(self):
        changed = []
        end = min(self.vi + 6, len(self.order))
        for i in range(self.vi, end):
            cell         = self.order[i]
            old_frontier = self.frontier_set.copy()
            if i < len(self.frontier_steps):
                self.frontier_set = set(self.frontier_steps[i])
            else:
                self.frontier_set = set()
            for fc in old_frontier - self.frontier_set:
                changed.append(fc)
            for fc in self.frontier_set - old_frontier:
                changed.append(fc)
            if cell != self.start and cell != self.goal:
                self.visited_set.add(cell)
                changed.append(cell)
        self.vi = end
        if self.vi >= len(self.order):
            self.frontier_set = set()
            self.phase        = "path"
        return changed

    def tick_path(self):
        changed = []
        end = min(self.pi + 2, len(self.path))
        for i in range(self.pi, end):
            cell = self.path[i]
            if cell != self.start and cell != self.goal:
                self.path_set.add(cell)
                changed.append(cell)
        self.pi = end
        if self.pi >= len(self.path):
            self.phase   = "move"
            self.agent_i = 0
        return changed

    def tick_move(self, changed):
        if self.agent_i < len(self.path):
            old          = self.agent
            self.agent   = self.path[self.agent_i]
            self.agent_i += 1
            changed.append(old)
            changed.append(self.agent)
            if self.var_dyn.get():
                new_walls = self.spawn_walls()
                if new_walls:
                    changed += list(new_walls)
                    self.repaint(changed)
                    remaining = set(self.path[self.agent_i:])
                    blocked   = bool(remaining & new_walls)
                    if blocked:
                        self.running  = False
                        self.replans += 1
                        self.lbl_replans.set(str(self.replans))
                        self.lbl_status.set(f"Blocked - re-planning #{self.replans}")
                        self.root.after(300, lambda: self.search_from(self.agent))
                        return True
            return False
        else:
            self.running = False
            self.phase   = "idle"
            self.agent   = self.goal
            changed.append(self.goal)
            self.repaint(changed)
            cost = len(self.path) - 1
            self.lbl_status.set(f"Done! Cost={cost}  Re-plans={self.replans}")
            return True

    def spawn_walls(self):
        new_walls = set()
        protected = {self.agent, self.start, self.goal}
        for _ in range(3):
            if random.random() < 0.06:
                r   = random.randint(0, self.R - 1)
                c   = random.randint(0, self.C - 1)
                pos = (r, c)
                if self.grid[r][c] == 0 and pos not in protected:
                    self.grid[r][c] = 2
                    new_walls.add(pos)
        return new_walls

    def stop(self):
        self.running = False
        self.phase   = "idle"
        self.agent   = None
        self.lbl_status.set("Stopped.")

    def clear_all(self):
        self.stop()
        self.grid         = [[0] * self.C for _ in range(self.R)]
        self.visited_set  = set()
        self.frontier_set = set()
        self.path_set     = set()
        self.lbl_nodes.set("--")
        self.lbl_cost.set("--")
        self.lbl_time.set("--")
        self.lbl_replans.set("--")
        self.draw_all()
        self.lbl_status.set("Grid cleared.")

    def clear_path(self, keep_walls=False):
        self.running        = False
        self.phase          = "idle"
        self.agent          = None
        self.path           = []
        self.order          = []
        self.frontier_steps = []
        self.visited_set    = set()
        self.frontier_set   = set()
        self.path_set       = set()
        if not keep_walls:
            for r in range(self.R):
                for c in range(self.C):
                    if self.grid[r][c] == 2:
                        self.grid[r][c] = 0
        self.draw_all()

    def random_maze(self):
        self.clear_path()
        for r in range(self.R):
            for c in range(self.C):
                is_special = (r, c) == self.start or (r, c) == self.goal
                if is_special:
                    self.grid[r][c] = 0
                elif random.random() < 0.30:
                    self.grid[r][c] = 1
                else:
                    self.grid[r][c] = 0
        if not self.path_reachable():
            self.carve_corridor()
        self.draw_all()
        self.lbl_status.set("Maze ready - click Run.")

    def path_reachable(self):
        visited = set()
        queue   = deque([self.start])
        while queue:
            node = queue.popleft()
            if node == self.goal:
                return True
            if node in visited:
                continue
            visited.add(node)
            r, c = node
            for nb in get_neighbours(self.grid, self.R, self.C, r, c):
                if nb not in visited:
                    queue.append(nb)
        return False

    def carve_corridor(self):
        r,  c  = self.start
        gr, gc = self.goal
        while (r, c) != (gr, gc):
            self.grid[r][c] = 0
            if r < gr:
                r += 1
            elif r > gr:
                r -= 1
            elif c < gc:
                c += 1
            else:
                c -= 1

    def resize_grid(self):
        try:
            new_r = int(self.var_rows.get())
            new_c = int(self.var_cols.get())
            valid = 5 <= new_r <= 30 and 5 <= new_c <= 35
            if not valid:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid size", "Rows: 5-30   Cols: 5-35")
            return

        self.stop()

        self.R  = new_r
        self.C  = new_c
        self.CS = max(14, min(26, min(506 // new_c, 552 // new_r)))

        self.grid  = [[0] * self.C for _ in range(self.R)]
        self.start = (1, 1)
        self.goal  = (self.R - 2, self.C - 2)

        self.path           = []
        self.order          = []
        self.frontier_steps = []
        self.visited_set    = set()
        self.frontier_set   = set()
        self.path_set       = set()

        self.canvas.destroy()
        self.canvas_frame.destroy()

        self.canvas_frame = tk.Frame(self.canvas_parent, bg=BG)
        self.canvas_frame.pack(side="left", padx=6, pady=6)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg=CELL_EMPTY,
            highlightthickness=1,
            highlightbackground=BORDER,
            width=self.C * self.CS,
            height=self.R * self.CS
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self.on_click)
        self.canvas.bind("<B1-Motion>",       self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.lbl_nodes.set("--")
        self.lbl_cost.set("--")
        self.lbl_time.set("--")
        self.lbl_replans.set("--")

        self.draw_all()
        self.lbl_status.set(f"Grid resized to {new_r} x {new_c}.")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()