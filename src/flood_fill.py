from collections import deque
from typing import List

from grid_data import GridData

EnclosedCells = List[List[bool]]

def find_enclosed_cells(grid: GridData) -> EnclosedCells:
    height = len(grid)
    width = len(grid[0])

    # Initialize arrays
    enclosed = [[False for _ in range(width)] for _ in range(height)]
    visited = [[False for _ in range(width)] for _ in range(height)]

    # Directions for moving in the grid: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

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
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy

                # Check if within bounds and not visited and is a free cell
                if 0 <= nx < height and 0 <= ny < width and not visited[nx][ny] and grid[nx][ny] == 0:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
                    component.append((nx, ny))

        # Mark the component as enclosed or not
        if enclosed_area:
            for (ex, ey) in component:
                enclosed[ex][ey] = True

    # Iterate over each cell in the grid
    for x in range(height):
        for y in range(width):
            if grid[x][y] == 0 and not visited[x][y]:
                is_enclosed(x, y)

    return enclosed


def test():
    # Example grid
    grid = [
        [1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    # Find all enclosed areas
    enclosed_grid = find_enclosed_cells(grid)

    print("Result:")

    # Print the result
    for row in enclosed_grid:
        print(' '.join(['x' if cell else '-' for cell in row]))
