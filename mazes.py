import time
import sys
import random
from collections import deque
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.columns import Columns
from rich.panel import Panel
from rich.live import Live
from rich.console import Group


class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [["#" for _ in range(self.width)] for _ in range(self.height)]
        self.generate()

    def generate(self):
        maze = self.maze
        start = (1, 1)
        stack = [start]
        maze[start[1]][start[0]] = "S"
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        while stack:
            x, y = stack[-1]
            random.shuffle(directions)
            for locx, locy in directions:
                nx = x + locx
                ny = y + locy
                if (
                    0 < nx < self.width - 1
                    and 0 < ny < self.height - 1
                    and maze[ny][nx] == "#"
                ):
                    maze[ny][nx] = " "
                    maze[ny - locy // 2][nx - locx // 2] = " "
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()

        for i in range(self.height):
            for j in range(self.width):
                if i == 0 or j == 0 or i == self.height - 1 or j == self.width - 1:
                    continue
                if maze[i][j] == "#":
                    adjacent_paths = sum(
                        1
                        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        if 0 <= i + di < self.height
                        and 0 <= j + dj < self.width
                        and maze[i + di][j + dj] in [" ", "S"]
                    )
                    if random.random() > 0.95 and adjacent_paths <= 2:
                        maze[i][j] = " "

        if maze[1][2] == "#" and maze[2][1] == "#":
            maze[1][2] = " "

        self._place_goal(start)
        return maze

    def _place_goal(self, start):  # ? gia na mpenei tyxaia alla TODO
        while True:
            goal_x = random.randint(1, self.width - 2)
            goal_y = random.randint(1, self.height - 2)

            if (goal_x, goal_y) != start and self.maze[goal_y][goal_x] == " ":
                self.maze[goal_y][goal_x] = "G"

                break
                self.maze[goal_y][goal_x] = " "

    def _is_solvable(self):
        start = (1, 1)
        goal = self.find_goal()
        if not goal:
            return False

        moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        frontier = deque([start])
        visited = set([start])

        while frontier:
            x, y = frontier.popleft()
            if (x, y) == goal:
                return True

            for dx, dy in moves:
                nx = x + dx
                ny = y + dy
                new_pos = (nx, ny)
                if new_pos in visited:
                    continue
                if self.maze[ny][nx] == "#":
                    continue
                frontier.append(new_pos)
                visited.add(new_pos)
        return False

    def find_goal(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == "G":
                    return (x, y)
        return None

    def render(self):
        output = "╔" + "═" * (self.width * 2) + "╗\n"
        for row in self.maze:
            pretty_row = "║ "
            for cell in row:
                if cell == "#":
                    pretty_row += "[#9B6B8F]█[/#9B6B8F] "
                elif cell == "S":
                    pretty_row += "[#4CAF50]S[/#4CAF50] "
                elif cell == "G":
                    pretty_row += "[#FF6B6B]G[/#FF6B6B] "
                else:
                    pretty_row += "  "
            pretty_row = pretty_row[:-1] + "║"
            output += pretty_row + "\n"
        output += "╚" + "═" * (self.width * 2) + "╝"
        return output


class PathFinder:
    def __init__(self, maze, algorithm="dfs"):
        self.maze = maze.maze
        self.width = len(self.maze[0])
        self.height = len(self.maze)
        self.start = (1, 1)
        self.goal = maze.find_goal()
        self.algorithm = algorithm.lower()
        self.time = 0

        if self.algorithm == "dfs":
            self.frontier = [self.start]  # LIFO dfs (STACK)
        else:  # BFS
            self.frontier = deque([self.start])  # FIFO bfs (queu)
        # TODO
        ## if self.algorithm == "dfs":
        ## else if self.algorithm == "bfs":
        ## else gia ID

        self.visited = set([self.start])
        self.parent = {self.start: None}
        self.frontier_cells = set([self.start])
        self.found = False
        self.path = []
        self.memory_usage = 0
        self.max_memory = 0

    def step(self):
        if not self.frontier or self.found:
            return False

        if self.algorithm == "dfs":
            current = self.frontier.pop()
        else:
            current = self.frontier.popleft()

        self.frontier_cells.remove(current)
        x, y = current

        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)

            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if next_pos in self.visited:
                continue
            if self.maze[ny][nx] == "#":
                continue

            if (nx, ny) == self.goal:
                self.found = True
                self.visited.add(next_pos)
                self.parent[next_pos] = current
                self._reconstruct_path()
                break

        if not self.found:
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if next_pos in self.visited:
                    continue
                if self.maze[ny][nx] == "#":
                    continue

                self.visited.add(next_pos)
                self.parent[next_pos] = current
                self.frontier.append(next_pos)
                self.frontier_cells.add(next_pos)

        self._update_memory_usage()
        return True

    def _reconstruct_path(self):
        if not self.found:
            return []

        path = [self.goal]
        current = self.goal
        while current != self.start:
            current = self.parent[current]
            path.append(current)
        path.reverse()
        self.path = path
        return self.path

    def _update_memory_usage(self):
        frontier_size = sys.getsizeof(self.frontier) + sum(
            sys.getsizeof(item) for item in self.frontier
        )
        visited_size = sys.getsizeof(self.visited) + sum(
            sys.getsizeof(item) for item in self.visited
        )
        self.memory_usage = frontier_size + visited_size
        self.max_memory = max(self.max_memory, self.memory_usage)

    def render(self, vis_maze=None):
        if vis_maze is None:
            vis_maze = [row[:] for row in self.maze]

        output = f"[bold]{self.algorithm.upper()}[/bold]\n"
        output += "╔" + "═" * (len(vis_maze[0]) * 2) + "╗\n"
        for y, row in enumerate(vis_maze):
            line = "║ "
            for x, cell in enumerate(row):
                pos = (x, y)
                if (
                    self.path
                    and pos in self.path
                    and pos not in [self.start, self.goal]
                ):
                    line += "[#1E90FF]●[/#1E90FF] "
                elif cell == "#":
                    line += "[#9B6B8F]█[/#9B6B8F] "
                elif cell == "S":
                    line += "[#4CAF50]S[/#4CAF50] "
                elif cell == "G":
                    line += "[#FF6B6B]G[/#FF6B6B] "
                elif pos in self.frontier_cells:
                    line += "[#FFB347]F[/#FFB347] "
                elif pos in self.visited:
                    line += "[#B19CD9]·[/#B19CD9] "
                else:
                    line += "  "
            line = line[:-1] + "║"
            output += line + "\n"
        output += "╚" + "═" * (len(vis_maze[0]) * 2) + "╝\n"

        output += (
            f"Visited: {len(self.visited)}  Frontier: {len(self.frontier_cells)}\n"
        )
        if self.found:
            output += f"Path found! Length: {len(self.path)}\n"

        output += f"Memory: {self.memory_usage} bytes (Max: {self.max_memory} bytes)"

        return output


class Visualiser:
    def __init__(self, console=None):
        self.console = console or Console()

    def display_maze(self, maze):
        self.console.print("")
        self.console.print("[italic]Your maze:[/italic]")
        self.console.print(maze.render())

    def visualise_algorithm(self, maze, algorithm="dfs"):

        pathfinder = PathFinder(maze, algorithm)
        vis_maze = [row[:] for row in maze.maze]

        with Live(pathfinder.render(vis_maze), refresh_per_second=10) as live:
            while pathfinder.step():
                live.update(pathfinder.render(vis_maze))
                time.sleep(0.05)

            live.update(pathfinder.render(vis_maze))

    def visualise_side_by_side(self, maze):
        dfs_finder = PathFinder(maze, "dfs")
        bfs_finder = PathFinder(maze, "bfs")

        def render_both():
            dfs_panel = Panel(
                dfs_finder.render(),
                title="Depth-First Search",
            )
            bfs_panel = Panel(
                bfs_finder.render(),
                title="Breadth-First Search",
            )
            legend = "[#4CAF50]S[/#4CAF50]: Start  [#FF6B6B]G[/#FF6B6B]: Goal  [#1E90FF]●[/#1E90FF]: Path  [#B19CD9]·[/#B19CD9]: Visited  [#FFB347]F[/#FFB347]: Frontier  [#9B6B8F]█[/#9B6B8F]: Wall"
            return Group(Columns([dfs_panel, bfs_panel], equal=True), legend)

        with Live(render_both(), refresh_per_second=60) as live:
            dfs_finder.time = time.time()
            bfs_finder.time = time.time()
            while dfs_finder.step() or bfs_finder.step():
                live.update(render_both())
                time.sleep(0.01)

            bfs_finder.time = time.time() - bfs_finder.time
            dfs_finder.time = time.time() - dfs_finder.time
            live.update(render_both())

    def run_and_export(self, maze, algorithms, export_format="text"):
        results = {}

        with self.console.status("[bold green]Running..."):
            for algorithm in algorithms:
                pathfinder = PathFinder(maze, algorithm)
                start_time = time.time()
                while pathfinder.step():
                    pass
                end_time = time.time()

                results[algorithm] = {
                    "finder": pathfinder,
                    "time": end_time - start_time,
                    "path_length": len(pathfinder.path) if pathfinder.found else 0,
                    "nodes_visited": len(pathfinder.visited),
                    "max_memory": pathfinder.max_memory,
                }

        if export_format == "text":
            self._export_to_text(maze, results)
        elif export_format == "image":
            self._export_to_image(maze, results)
        elif export_format == "graph":
            self._export_graph(maze, results)
        elif export_format == "all":
            self._export_to_text(maze, results)
            self._export_to_image(maze, results)
            self._export_graph(results)

        self.console.print("[bold green]Export complete![/bold green]")

    def _export_to_text(self, maze, results):
        filename = f"maze_comparison_{time.strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, "w") as f:
            f.write(f"Maze size: {maze.width}x{maze.height}\n\n")
            f.write("Original Maze:\n")
            maze_string = maze.render()
            maze_string = maze_string.replace(
                "[#9B6B8F]█[/#9B6B8F]", "█"
            )  # vgazoume ta xromata gt einai tou rich
            maze_string = maze_string.replace("[#4CAF50]S[/#4CAF50]", "S")
            maze_string = maze_string.replace("[#FF6B6B]G[/#FF6B6B]", "G")
            f.write(maze_string + "\n\n")

            f.write("Algorithm Comparison:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Algorithm':<12} {'Path Length':<15} {'Nodes Visited':<15} {'Time (s)':<15} {'Max Memory':<20}\n"
            )
            f.write("-" * 80 + "\n")

            for algorithm, data in results.items():
                memory_bytes = data["max_memory"]
                if memory_bytes >= 1024 * 1024 * 1024:
                    memory_str = f"{memory_bytes/(1024*1024*1024):.2f}GB"
                elif memory_bytes >= 1024 * 1024:
                    memory_str = f"{memory_bytes/(1024*1024):.2f}MB"
                else:
                    memory_str = f"{memory_bytes/1024:.2f}KB"
                f.write(
                    f"{algorithm.upper():<12} {data['path_length']:<15} {data['nodes_visited']:<15} {data['time']:.6f}{' ':<8} {memory_str:<20}\n"
                )
            f.write("-" * 80 + "\n\n")

            for algorithm, data in results.items():
                f.write(f"{algorithm.upper()} Solution:\n")

                vis_maze = [row[:] for row in maze.maze]
                finder = data["finder"]

                solution_str = self._get_solution_string(vis_maze, finder)
                f.write(solution_str + "\n\n")

        self.console.print(f"Results exported to [bold]{filename}[/bold]")

    def _get_solution_string(self, vis_maze, finder):
        solution_str = ""
        solution_str += "╔" + "═" * (len(vis_maze[0]) * 2) + "╗\n"

        for y, row in enumerate(vis_maze):
            line = "║ "
            for x, cell in enumerate(row):
                pos = (x, y)
                if (
                    finder.path
                    and pos in finder.path
                    and pos not in [finder.start, finder.goal]
                ):
                    line += "● "  # monopati
                elif cell == "#":
                    line += "█ "  # toixos
                elif cell == "S":
                    line += "S "  # arxi
                elif cell == "G":
                    line += "G "  # telos
                else:
                    line += "  "
            line = line[:-1] + "║"
            solution_str += line + "\n"

        solution_str += "╚" + "═" * (len(vis_maze[0]) * 2) + "╝"
        return solution_str

    def _export_to_image(self, maze, results):
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            self.console.print("[bold red]Error:[/bold red] PIL missing.")
            return

        cell_size = 20
        padding = 50
        title_height = 30
        stats_height = 60

        maze_width = maze.width * cell_size
        maze_height = maze.height * cell_size

        num_algorithms = len(results)
        cols = min(3, num_algorithms)
        rows = (num_algorithms + cols - 1) // cols

        image_width = padding + cols * (maze_width + padding)
        image_height = (
            padding + title_height + rows * (maze_height + stats_height + padding)
        )

        img = Image.new("RGB", (image_width, image_height), color="white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        draw.text(
            (padding, padding // 2),
            f"search comparison ({maze.width}x{maze.height})",
            fill="black",
            font=title_font,
        )

        row, col = 0, 0
        for algorithm, data in results.items():
            finder = data["finder"]

            x_pos = padding + col * (maze_width + padding)
            y_pos = (
                padding + title_height + row * (maze_height + stats_height + padding)
            )

            draw.text(
                (x_pos, y_pos - 20),
                f"{algorithm.upper()}",
                fill="black",
                font=title_font,
            )
            for y in range(maze.height):
                for x in range(maze.width):
                    cell_x = x_pos + x * cell_size
                    cell_y = y_pos + y * cell_size
                    pos = (x, y)

                    if maze.maze[y][x] == "#":
                        draw.rectangle(
                            [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                            fill="#9B6B8F",
                        )
                    elif maze.maze[y][x] == "S":
                        draw.rectangle(
                            [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                            fill="#4CAF50",
                        )
                    elif maze.maze[y][x] == "G":
                        draw.rectangle(
                            [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                            fill="#FF6B6B",
                        )
                    elif finder.path and pos in finder.path:
                        draw.rectangle(
                            [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                            fill="#1E90FF",
                        )
                    elif pos in finder.visited:
                        draw.rectangle(
                            [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                            fill="#B19CD9",
                        )
                    else:
                        draw.rectangle(
                            [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                            fill="white",
                        )

                    draw.rectangle(
                        [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                        outline="#6B4D57",
                    )

            stats_y = y_pos + maze_height + 10
            draw.text(
                (x_pos, stats_y),
                f"Path length: {data['path_length']}",
                fill="black",
                font=font,
            )
            draw.text(
                (x_pos, stats_y + 15),
                f"Nodes visited: {data['nodes_visited']}",
                fill="black",
                font=font,
            )
            draw.text(
                (x_pos, stats_y + 30),
                f"Time: {data['time']:.6f} seconds",
                fill="black",
                font=font,
            )
            memory_bytes = data["max_memory"]
            if memory_bytes >= 1024 * 1024 * 1024:
                memory_str = f"{memory_bytes/(1024*1024*1024)::.2f}GB"
            elif memory_bytes >= 1024 * 1024:
                memory_str = f"{memory_bytes/(1024*1024):.2f}MB"
            else:
                memory_str = f"{memory_bytes/1024:.2f}KB"
            draw.text(
                (x_pos, stats_y + 45),
                f"Max memory: {memory_str}",
                fill="black",
                font=font,
            )

            col += 1
            if col >= cols:
                col = 0
                row += 1
        filename = f"maze_comparison_{time.strftime('%Y%m%d_%H%M%S')}.png"
        img.save(filename)
        self.console.print(f"Image exported to [bold]{filename}[/bold]")

    def _export_graph(self, results):
       #TODO
       #pygraphviz? networkx
       pass


def main():
    console = Console()
    console.print("[bold green]BFS vs DFS Search Comparison[/bold green]")
    console.print("")

    width = IntPrompt.ask("Width:", default=15)
    height = IntPrompt.ask("Height:", default=15)

    maze = Maze(width, height)
    visualiser = Visualiser(console)
    if width < 5 or height < 5:
        console.print("[bold red]Maze dimensions must be at least 5x5.[/bold red]")
        return
    if width > 50 or height > 50:
        pic = Prompt.ask(
            "[green]big maze! preview?[/green]", choices=["yes", "no"], default="no"
        )
        if pic == "yes":
            visualiser.display_maze(maze)

    mode = Prompt.ask("Choose mode", choices=["visualise", "export"], default="export")

    if mode == "visualise":
        visualiser.visualise_side_by_side(maze)
    else:
        export_format = Prompt.ask(
            "Export format", choices=["text", "image", "graph", "all"], default="text"
        )
        algorithms = ["dfs", "bfs"]
        console.print("\n[bold]Running algorithms in background, wait![/bold]")
        visualiser.run_and_export(maze, algorithms, export_format)


if __name__ == "__main__":
    main()
