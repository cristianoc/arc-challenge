import time
from typing import Optional, Tuple
import random

from grid_data import GridData

"""
A Frame represents a rectangular region in a grid, defined by the coordinates (top, left, bottom, right).

A frame is specifically characterized by having all cells along the border of the rectangle filled (i.e., with a value of 1).
The interior cells of the rectangle are not considered part of the frame.
"""
Frame = Tuple[int, int, int, int]


def calculate_area(top: int, left: int, bottom: int, right: int) -> int:
    """Calculate the area of the rectangle defined by the corners (top, left) to (bottom, right)."""
    return (bottom - top + 1) * (right - left + 1)


def precompute_sums(grid: GridData, foreground: int) -> Tuple[GridData, GridData]:
    """Precompute the row and column sums for the grid to optimize the frame-checking process."""
    rows = len(grid)
    cols = len(grid[0])

    row_sum = [[0] * cols for _ in range(rows)]
    col_sum = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == foreground:
                row_sum[i][j] = row_sum[i][j-1] + 1 if j > 0 else 1
                col_sum[i][j] = col_sum[i-1][j] + 1 if i > 0 else 1

    return row_sum, col_sum


def find_largest_frame(grid: GridData, foreground: int) -> Optional[Frame]:
    """Find the largest frame (rectangle with a fully filled border) in the grid."""
    row_sum, col_sum = precompute_sums(grid, foreground)
    max_area = 0
    max_frame = None

    rows = len(grid)
    cols = len(grid[0])

    for top in range(rows):
        for left in range(cols):
            for bottom in range(top, rows):
                # Early termination if the potential max area is less than the current max_area
                if (rows - top) * (cols - left) <= max_area:
                    break
                for right in range(left, cols):
                    if is_frame_dp(row_sum, col_sum, top, left, bottom, right):
                        area = calculate_area(top, left, bottom, right)
                        if area > max_area:
                            max_area = area
                            max_frame = (top, left, bottom, right)

    return max_frame


def is_frame_dp(row_sum: GridData, col_sum: GridData, top: int, left: int, bottom: int, right: int) -> bool:
    """Check if the rectangle defined by (top, left) to (bottom, right) forms a frame using precomputed sums."""
    if row_sum[top][right] - (row_sum[top][left-1] if left > 0 else 0) != right - left + 1:
        return False
    if row_sum[bottom][right] - (row_sum[bottom][left-1] if left > 0 else 0) != right - left + 1:
        return False
    if col_sum[bottom][left] - (col_sum[top-1][left] if top > 0 else 0) != bottom - top + 1:
        return False
    if col_sum[bottom][right] - (col_sum[top-1][right] if top > 0 else 0) != bottom - top + 1:
        return False
    return True


def eval():
    # Define sizes
    width = 50
    height = 30

    # Create test grids
    empty_grid = [[0 for _ in range(width)] for _ in range(width)]
    full_grid = [[1 for _ in range(width)] for _ in range(width)]
    square_grid = [[random.choice([0, 1])
                    for _ in range(width)] for _ in range(width)]
    wide_grid = [[random.choice([0, 1]) for _ in range(width)]
                 for _ in range(height)]
    tall_grid = [[random.choice([0, 1]) for _ in range(height)]
                 for _ in range(width)]
    biased_random_grid = [[1 if random.random() < 0.9 else 0 for _ in range(
        width)] for _ in range(width)]  # Biased random grid

    # List of grids and their names
    grids = [
        ("Empty Square Grid", empty_grid),
        ("Full Square Grid", full_grid),
        ("Random Square Grid", square_grid),
        ("Random Wide Grid", wide_grid),
        ("Random Tall Grid", tall_grid),
        ("Biased Random Grid", biased_random_grid)
    ]

    # Run tests for each grid
    for grid_name, grid in grids:
        start_time = time.time()
        foreground = 1
        max_frame = find_largest_frame(grid, foreground)
        max_area = calculate_area(*max_frame) if max_frame else 0
        end_time = time.time()

        execution_time = end_time - start_time

        if not max_frame:
            print(f"{grid_name}: No valid frame found. "
                  f"Time: {execution_time:.6f} seconds\n")
        else:
            start_row, start_col, end_row, end_col = max_frame
            frame_height = end_row - start_row + 1
            frame_width = end_col - start_col + 1
            print(f"{grid_name}: Frame at ({start_row},{start_col}) to ({end_row},{end_col}), "
                  f"Size: {frame_height}x{frame_width}, Area: {max_area}, "
                  f"Time: {execution_time:.6f} seconds\n")
