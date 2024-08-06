from typing import List

from grid_data import DIRECTIONS8, Cell, GridData

# Type alias for the id of visited objects
VISITED = List[List[bool]]

# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def dfs_recursive_list(data: GridData, visited: VISITED, r: int, c: int, color: int, component: ConnectedComponent):
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
    for dr, dc in DIRECTIONS8:
        dfs_recursive_list(data, visited, r + dr, c + dc,
                           color, component)


def find_connected_components(data: GridData) -> List[ConnectedComponent]:
    rows = len(data)
    cols = len(data[0])
    visited: VISITED = [[False for _ in range(cols)] for _ in range(rows)]
    connected_components: List[ConnectedComponent] = []

    for r in range(rows):
        for c in range(cols):
            # Skip cells with color 0
            if visited[r][c] == False and data[r][c] != 0:
                # Create a new component
                component: ConnectedComponent = []
                dfs_recursive_list(data, visited, r, c,
                                   data[r][c], component)
                # Add the component to the list of connected components
                if component:
                    connected_components.append(component)

    return connected_components


