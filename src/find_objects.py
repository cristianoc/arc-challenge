from typing import List

from grid import Grid
from grid_data import Cell, Object

# Type alias for the id of visited objects
VISITED = List[List[bool]]

# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def dfs_recursive_list(grid: Grid, visited: VISITED, r: int, c: int, color: int, component: ConnectedComponent):
    # Base conditions to stop recursion
    if r < 0 or r >= len(grid.data) or c < 0 or c >= len(grid.data[0]):
        return
    if visited[r][c] or grid.data[r][c] != color:
        return

    # Mark the cell as visited
    visited[r][c] = True

    # Add the cell to the current component
    component.append((r, c))

    # Direction vectors for 8 directions (N, NE, E, SE, S, SW, W, NW)
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]

    # Recursively visit all 8 neighbors
    for dr, dc in directions:
        dfs_recursive_list(grid, visited, r + dr, c + dc,
                           color, component)


def find_connected_components(grid: Grid) -> List[ConnectedComponent]:
    rows = len(grid.data)
    cols = len(grid.data[0])
    visited: VISITED = [[False for _ in range(cols)] for _ in range(rows)]
    connected_components: List[ConnectedComponent] = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] == False:
                # Create a new component
                component: ConnectedComponent = []
                dfs_recursive_list(grid, visited, r, c,
                                   grid.data[r][c], component)
                # Add the component to the list of connected components
                if component:
                    connected_components.append(component)

    return connected_components

def create_object(grid: Grid, component: ConnectedComponent) -> Object:
    """
    Create an object from a connected component in a grid
    """
    min_row = min(r for r, _ in component)
    min_col = min(c for _, c in component)
    rows = max(r for r, _r in component) - min_row + 1
    columns = max(c for _, c in component) - min_col + 1
    data = Grid.empty(rows=rows, columns=columns).data
    for r, c in component:
        data[r - min_row][c - min_col] = grid.data[r][c]
    return Object((min_row, min_col), data)

def detect_objects(grid: Grid) -> List[Object]:
    connected_components = find_connected_components(grid)
    detected_objects = [create_object(grid, component) for component in connected_components]
    return detected_objects


def test():
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    objects = detect_objects(Grid(grid))
    for obj in objects:
        print(f"Detected object: {obj}")
