from typing import TYPE_CHECKING, List, Optional

from grid_types import DIRECTIONS4, DIRECTIONS8, Cell, GridData

# To avoid circular imports
if TYPE_CHECKING:
    from objects import Object as Object_t
else:
    Object_t = None

# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def dfs_recursive_list(
    grid: Object_t,
    visited: Object_t,
    x: int,
    y: int,
    color: Optional[int],
    component: ConnectedComponent,
    diagonals: bool,
    allow_black: bool,
):
    # Base conditions to stop recursion
    if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
        return
    if visited[x, y]:
        return
    if color is not None and grid[x, y] != color:
        return
    if grid[x, y] == 0 and not allow_black:
        return

    # Mark the cell as visited
    visited[x, y] = 1

    # Add the cell to the current component
    component.append((x, y))

    # Recursively visit all 8 neighbors
    directions = DIRECTIONS8 if diagonals else DIRECTIONS4
    for dx, dy in directions:
        dfs_recursive_list(
            grid, visited, x + dx, y + dy, color, component, diagonals, allow_black
        )


def find_connected_components(
    grid: Object_t, diagonals: bool, allow_black: bool, multicolor: bool
) -> List[ConnectedComponent]:
    width, height = grid.size
    from objects import Object

    visited: Object_t = Object.empty(size=grid.size)
    connected_components: List[ConnectedComponent] = []

    for x in range(width):
        for y in range(height):
            # Skip cells with color 0
            if visited[x, y] == 0 and (allow_black or grid[x, y] != 0):
                # Create a new component
                component: ConnectedComponent = []
                color = grid[x, y] if not multicolor else None
                dfs_recursive_list(
                    grid, visited, x, y, color, component, diagonals, allow_black
                )
                # Add the component to the list of connected components
                if component:
                    connected_components.append(component)

    return connected_components
