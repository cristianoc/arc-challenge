from collections import deque
from typing import TYPE_CHECKING, List

import numpy as np

from grid_types import DIRECTIONS4
from logger import logger

# To avoid circular imports
if TYPE_CHECKING:
    from objects import Object as Object_t
else:
    Object_t = None


EnclosedCells = List[List[bool]]


def find_enclosed_cells(grid: Object_t) -> EnclosedCells:
    width, height = grid.size

    # Initialize arrays
    enclosed = [[False for _ in range(width)] for _ in range(height)]
    visited = [[False for _ in range(width)] for _ in range(height)]

    def is_enclosed(x: int, y: int):
        # BFS queue
        queue = deque([(x, y)])
        component = [(x, y)]
        visited[x][y] = True
        enclosed_area = True

        while queue:
            cx, cy = queue.popleft()

            # Check if this cell is at the boundary
            if cx == 0 or cy == 0 or cx == height - 1 or cy == width - 1:
                enclosed_area = False

            # Explore neighbors
            for dx, dy in DIRECTIONS4:
                nx, ny = cx + dx, cy + dy

                # Check if within bounds and not visited and is a free cell
                if (
                    0 <= nx < height
                    and 0 <= ny < width
                    and not visited[nx][ny]
                    and grid[ny, nx] == 0
                ):
                    visited[nx][ny] = True
                    queue.append((nx, ny))
                    component.append((nx, ny))

        # Mark the component as enclosed or not
        if enclosed_area:
            for ex, ey in component:
                enclosed[ex][ey] = True

    # Iterate over each cell in the grid
    for x in range(height):
        for y in range(width):
            if grid[y, x] == 0 and not visited[x][y]:
                is_enclosed(x, y)

    return enclosed


def test():
    # Example grid
    from objects import Object
    grid = Object(np.array([
        [1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],    
    ]))

    # Find all enclosed areas
    enclosed_grid = find_enclosed_cells(grid)

    logger.info("Result:")

    # Print the result
    for row in enclosed_grid:
        logger.info(" ".join(["x" if cell else "-" for cell in row]))
