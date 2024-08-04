from typing import List, Set

from flood_fill import EnclosedCells, find_enclosed_cells
from grid import Grid
from grid_data import Cell, Object

"""
This module detects and identifies enclosed objects within a 2D grid.
It works by identifying enclosed regions, finding connected components 
using depth-first search, and determining contours to define object boundaries.

Each detected object is represented by its bounding box, origin, width, height, 
and relative cell positions.
"""


# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def find_connected_components(enclosed_cells: EnclosedCells) -> List[ConnectedComponent]:
    def dfs(x: int, y: int) -> ConnectedComponent:
        # Stack for DFS
        stack = [(x, y)]
        component: ConnectedComponent = []

        # Directions for moving in the grid (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while stack:
            cx, cy = stack.pop()
            if visited[cx][cy]:
                continue

            # Mark as visited and add to component
            visited[cx][cy] = True
            component.append((cx, cy))

            # Explore neighbors
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and enclosed_cells[nx][ny]:
                    stack.append((nx, ny))

        return component

    rows = len(enclosed_cells)
    cols = len(enclosed_cells[0])
    visited = [[False] * cols for _ in range(rows)]
    components: List[ConnectedComponent] = []

    for i in range(rows):
        for j in range(cols):
            if enclosed_cells[i][j] and not visited[i][j]:
                # Start a new component
                new_component = dfs(i, j)
                components.append(new_component)

    return components


def find_contour(grid: Grid, component: ConnectedComponent) -> List[Cell]:
    """
    Find the contour of a connected component in a grid: all the cells that are reachable in one step from
    a cell in the component, and which have a non-zero value.
    """
    rows = len(grid.data)
    cols = len(grid.data[0])
    contour: Set[Cell] = set()  # Use a set to avoid duplicate entries

    # Directions for moving in the grid (right, down, left, up), and diagonals
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for x, y in component:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if the new position is within bounds, has a non-zero value, and is not part of the component
            if 0 <= nx < rows and 0 <= ny < cols and grid.data[nx][ny] != 0 and (nx, ny) not in component:
                # Add to the set to ensure uniqueness
                contour.add((nx, ny))

    # Convert the set back to a list before returning
    return sorted(contour)


def detect_object(grid: Grid, contour: List[Cell]) -> Object:
    """
    Given a list of cells forming a contour, return them as a detected object.
    """
    min_row = min(r for r, _ in contour)
    min_col = min(c for _, c in contour)
    height = max(r for r, _r in contour) - min_row + 1
    width = max(c for _, c in contour) - min_col + 1
    data = Grid.empty(rows=height, columns=width).data
    for r, c in contour:
        data[r - min_row][c - min_col] = grid.data[r][c]
    return Object((min_row, min_col), height, width, data)


def detect_objects(grid: Grid) -> List[Object]:
    enclosed_cells = find_enclosed_cells(grid.data)
    detected_objects = [
        detect_object(grid, find_contour(grid, component))
        for component in find_connected_components(enclosed_cells)
    ]
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
