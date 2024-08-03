from dataclasses import dataclass
from typing import List, Set

from flood_fill import Cell, EnclosedCells, find_enclosed_cells

ConnectedComponent = List[Cell]


@dataclass
class DetectedObject:
    origin: Cell  # top-left corner of the bounding box
    width: int
    height: int
    cells: List[Cell]  # cells w.r.t the origin


def find_connected_components(enclosed_cells: EnclosedCells) -> List[ConnectedComponent]:
    def dfs(x: int, y: int):
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


def find_contour(grid: List[List[int]], component: ConnectedComponent) -> List[Cell]:
    """
    Find the contour of a connected component in a grid: all the cells that are reachable in one step from
    a cell in the component, and which have a non-zero value.
    """
    rows = len(grid)
    cols = len(grid[0])
    contour: Set[Cell] = set()  # Use a set to avoid duplicate entries

    # Directions for moving in the grid (right, down, left, up), and diagonal
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for x, y in component:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if the new position is within bounds, has a non-zero value, and is not part of the component
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 0 and (nx, ny) not in component:
                # Add to the set to ensure uniqueness
                contour.add((nx, ny))

    # Convert the set back to a list before returning
    return sorted(contour)


def detect_object(contour: List[Cell]) -> DetectedObject:
    """
    Given a list of cells forming a contour, return them as a detected object.
    """
    min_x = min(x for x, _y in contour)
    min_y = min(y for _x, y in contour)
    width = max(x for x, _y in contour) - min_x + 1
    height = max(y for _x, y in contour) - min_y + 1
    return DetectedObject((min_x, min_y), width, height, [(x - min_x, y - min_y) for x, y in contour])


def test():
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    enclosed_cells = find_enclosed_cells(grid)
    print(f"Enclosed cells: {enclosed_cells}")

    detected_objects = [
        detect_object(find_contour(grid, component))
        for component in find_connected_components(enclosed_cells)
    ]
    for obj in detected_objects:
        print(f"Detected object: {obj}")
