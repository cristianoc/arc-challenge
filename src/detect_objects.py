from typing import List

from grid_types import DIRECTIONS4, DIRECTIONS8, Cell, GridData

from typing import TYPE_CHECKING

# To avoid circular imports
if TYPE_CHECKING:
    from objects import Object as Object_t
else:
    Object_t = None


# Type alias for the id of visited objects
VISITED = List[List[bool]]

# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def dfs_recursive_list(
    grid: Object_t,
    visited: VISITED,
    r: int,
    c: int,
    color: int,
    component: ConnectedComponent,
    diagonals: bool,
):
    # Base conditions to stop recursion
    if r < 0 or r >= grid.height or c < 0 or c >= grid.width:
        return
    if visited[r][c] or grid[c, r] != color:
        return

    # Mark the cell as visited
    visited[r][c] = True

    # Add the cell to the current component
    component.append((r, c))

    # Recursively visit all 8 neighbors
    directions = DIRECTIONS8 if diagonals else DIRECTIONS4
    for dr, dc in directions:
        dfs_recursive_list(grid, visited, r + dr, c + dc, color, component, diagonals)


def find_connected_components(
    grid: Object_t, diagonals: bool, allow_black: bool
) -> List[ConnectedComponent]:
    cols, rows = grid.size
    visited: VISITED = [[False for _ in range(cols)] for _ in range(rows)]
    connected_components: List[ConnectedComponent] = []

    for r in range(rows):
        for c in range(cols):
            # Skip cells with color 0
            if visited[r][c] == False and (allow_black or grid[c, r] != 0):
                # Create a new component
                component: ConnectedComponent = []
                dfs_recursive_list(
                    grid, visited, r, c, grid[c, r], component, diagonals
                )
                # Add the component to the list of connected components
                if component:
                    connected_components.append(component)

    return connected_components
