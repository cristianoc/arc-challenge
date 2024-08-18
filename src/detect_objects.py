from typing import List

from grid_data import DIRECTIONS4, DIRECTIONS8, Cell, GridData

# Type alias for the id of visited objects
VISITED = List[List[bool]]

# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def dfs_recursive_list(data: GridData, visited: VISITED, r: int, c: int, color: int, component: ConnectedComponent, diagonals: bool):
    # Base conditions to stop recursion
    if r < 0 or r >= len(data) or c < 0 or c >= len(data[0]):
        return
    if visited[r][c] or data[r][c] != color:
        return

    # Mark the cell as visited
    visited[r][c] = True

    # Add the cell to the current component
    component.append((r, c))

    # Recursively visit all 8 neighbors
    directions = DIRECTIONS8 if diagonals else DIRECTIONS4
    for dr, dc in directions:
        dfs_recursive_list(data, visited, r + dr, c + dc,
                           color, component, diagonals)


def find_connected_components(data: GridData, diagonals: bool, allow_black: bool) -> List[ConnectedComponent]:
    rows = len(data)
    cols = len(data[0])
    visited: VISITED = [[False for _ in range(cols)] for _ in range(rows)]
    connected_components: List[ConnectedComponent] = []

    for r in range(rows):
        for c in range(cols):
            # Skip cells with color 0
            if visited[r][c] == False and (allow_black or data[r][c] != 0):
                # Create a new component
                component: ConnectedComponent = []
                dfs_recursive_list(data, visited, r, c,
                                   data[r][c], component, diagonals)
                # Add the component to the list of connected components
                if component:
                    connected_components.append(component)

    return connected_components
