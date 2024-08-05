import random
import time
from typing import List

from grid import Grid
from grid_data import Cell

# Type alias for the id of visited objects
VISITED = List[List[int]]


# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def dfs_recursive_list(grid: Grid, visited: VISITED, r: int, c: int, color: int, object_id: int):
    # Base conditions to stop recursion
    if r < 0 or r >= len(grid.data) or c < 0 or c >= len(grid.data[0]):
        return
    if visited[r][c] or grid.data[r][c] != color:
        return

    # Mark the cell as visited with the current object ID
    visited[r][c] = object_id

    # Direction vectors for 8 directions (N, NE, E, SE, S, SW, W, NW)
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]

    # Recursively visit all 8 neighbors
    for dr, dc in directions:
        dfs_recursive_list(grid, visited, r + dr, c + dc, color, object_id)


def detect_objects_list(grid: Grid) -> tuple[int, VISITED]:
    rows = len(grid.data)
    cols = len(grid.data[0])
    visited : VISITED = [[0 for _ in range(cols)] for _ in range(rows)]
    object_id = 0

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] == 0:
                object_id += 1
                dfs_recursive_list(grid, visited, r, c,
                                   grid.data[r][c], object_id)

    return object_id, visited




GRID_SIZE = 300


def test_lists():
    # Generate a grid with random integers for benchmarking
    random.seed(42)  # Seed for reproducibility
    medium_grid = Grid([[random.randint(0, 10) for _ in range(
        GRID_SIZE)] for _ in range(GRID_SIZE)])
    # Benchmark the recursive DFS implementation
    start_time_dfs_recursive = time.time()
    object_count_dfs_recursive, _visited_dfs_recursive = detect_objects_list(
        medium_grid)
    end_time_dfs_recursive = time.time()
    execution_time_dfs_recursive = end_time_dfs_recursive - start_time_dfs_recursive

    print(
        f"DFS Lists: Total Objects Detected: {object_count_dfs_recursive}, Execution Time: {execution_time_dfs_recursive:.3f} seconds")


if __name__ == "__main__":
    test_lists()
    test_lists()
