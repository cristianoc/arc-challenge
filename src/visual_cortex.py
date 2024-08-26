import time
from typing import Optional, Tuple
import random

from grid_data import GridData, logger

"""
A Frame represents a rectangular region in a grid, defined by the coordinates (top, left, bottom, right).

A frame is specifically characterized by having all cells along the border of the rectangle filled (i.e., with a value of 1).
The interior cells of the rectangle are not considered part of the frame.
"""
Frame = Tuple[int, int, int, int]


def calculate_area(top: int, left: int, bottom: int, right: int) -> int:
    """Calculate the area of the rectangle defined by the corners (top, left) to (bottom, right)."""
    return (bottom - top + 1) * (right - left + 1)


def precompute_sums(grid: GridData, color: int) -> Tuple[GridData, GridData]:
    """Precompute the row and column sums for the grid to optimize the frame-checking process."""
    rows = len(grid)
    cols = len(grid[0])

    row_sum = [[0] * cols for _ in range(rows)]
    col_sum = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == color:
                row_sum[i][j] = row_sum[i][j-1] + 1 if j > 0 else 1
                col_sum[i][j] = col_sum[i-1][j] + 1 if i > 0 else 1

    return row_sum, col_sum


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


def is_frame(grid: GridData, top: int, left: int, bottom: int, right: int, color: int) -> bool:
    """Check if the rectangle defined by (top, left) to (bottom, right) forms a frame."""
    for i in range(left, right + 1):
        if grid[top][i] != color or grid[bottom][i] != color:
            return False
    for i in range(top, bottom + 1):
        if grid[i][left] != color or grid[i][right] != color:
            return False
    return True


def find_largest_frame(grid: GridData, color: Optional[int]) -> Optional[Frame]:
    """
    Find the largest frame in the grid that has all border cells matching the specified color.
    If the color is None, the function will find the largest frame with any color.
    """
    row_sum, col_sum = precompute_sums(grid, color) if color else (None, None)
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
                    top_left_corner_color = grid[top][left]
                    if is_frame_dp(row_sum, col_sum, top, left, bottom, right) if row_sum and col_sum else is_frame(grid, top, left, bottom, right, top_left_corner_color):
                        area = calculate_area(top, left, bottom, right)
                        if area > max_area:
                            max_area = area
                            max_frame = (top, left, bottom, right)

    return max_frame


def find_smallest_frame(grid: GridData, color: Optional[int], min_size: Optional[Tuple[int, int]] = None) -> Optional[Frame]:
    """
    Find the smallest frame in the grid that has all border cells matching the specified color.
    If the color is None, the function will find the smallest frame with any color.
    """
    row_sum, col_sum = precompute_sums(grid, color) if color else (None, None)
    min_area = float('inf')
    min_frame = None

    rows = len(grid)
    cols = len(grid[0])

    for top in range(rows):
        for left in range(cols):
            for bottom in range(top, rows):
                # Early termination if the potential min area is greater than the current min_area
                if (rows - top) * (cols - left) >= min_area:
                    break
                for right in range(left, cols):
                    height = bottom - top + 1
                    width = right - left + 1
                    if min_size and (height < min_size[0] or width < min_size[1]):
                        continue
                    top_left_corner_color = grid[top][left]
                    if is_frame_dp(row_sum, col_sum, top, left, bottom, right) if row_sum and col_sum else is_frame(grid, top, left, bottom, right, top_left_corner_color):
                        area = calculate_area(top, left, bottom, right)
                        if area < min_area:
                            min_area = area
                            min_frame = (top, left, bottom, right)

    return min_frame

def is_frame_part_of_lattice(grid: GridData, frame: Frame, foreground: int) -> bool:
    """
    Determines whether the rectangular frame defined by the coordinates (top, left) to (bottom, right) can be part of a 
    repeating pattern in the grid, where all the borders of the frame match the specified foreground color.

    This function checks if all the border cells of the frame, with dimensions (bottom - top + 1, right - left + 1), 
    match the given foreground color. The function aligns the starting points within the grid based on the frame's 
    height and width, ensuring that the check covers all possible locations where a full frame fits within the grid.

    The function iterates over the grid in steps of the frame's height and width. It checks only the regions where 
    the entire frame fits within the grid bounds. If any border cell of the frame does not match the foreground color, 
    the function returns False. The function returns True only if all checked frames match the foreground color.

    Note: The function does not enforce that the entire grid forms a uniform lattice pattern, only that the specified 
    frame could be part of such a repeating pattern. 

    Parameters:
    - grid (GridData): The 2D grid where each cell contains an integer representing a color.
    - top (int): The top row index of the frame.
    - left (int): The left column index of the frame.
    - bottom (int): The bottom row index of the frame.
    - right (int): The right column index of the frame.
    - foreground (int): The integer representing the color that the frame borders should match.

    Returns:
    - bool: True if the frame could be part of a repeating pattern in the grid; False otherwise.
    """
    top, left, bottom, right = frame
    frame_height = bottom - top + 1
    frame_width = right - left + 1
    rows = len(grid)
    cols = len(grid[0])

    # Adjust the starting points to ensure alignment with the top-left corner
    start_y = top % frame_height
    start_x = left % frame_width

    if frame_height <= 1 or frame_width <= 1:
        return False
    for y in range(start_y, rows, frame_height-1):
        for x in range(start_x, cols, frame_width-1):
            # Check if the frame fits within the grid bounds
            if y + frame_height > rows or x + frame_width > cols:
                continue
            # Check top and bottom borders of the frame
            for i in range(frame_width):
                if y < rows and (x + i) < cols:
                    if grid[y][x + i] != foreground or grid[y + frame_height - 1][x + i] != foreground:
                        return False
            # Check left and right borders of the frame
            for j in range(frame_height):
                if y + j < rows and x < cols:
                    if grid[y + j][x] != foreground or grid[y + j][x + frame_width - 1] != foreground:
                        return False
    return True


def eval_with_lattice_check():
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
        is_lattice = is_frame_part_of_lattice(
            grid, max_frame, foreground) if max_frame else False
        end_time = time.time()

        execution_time = end_time - start_time

        if not max_frame:
            logger.info(f"{grid_name}: No valid frame found. "
                  f"Time: {execution_time:.6f} seconds\n")
        else:
            start_row, start_col, end_row, end_col = max_frame
            frame_height = end_row - start_row + 1
            frame_width = end_col - start_col + 1
            logger.info(f"{grid_name}: Frame at ({start_row},{start_col}) to ({end_row},{end_col}), "
                  f"Size: {frame_height}x{frame_width}, Area: {max_area}, "
                  f"Part of lattice: {is_lattice}, "
                  f"Time: {execution_time:.6f} seconds\n")


def test_lattices():
    # Correct Lattice Grid
    grid = [
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    ]
    frame = (2, 2, 5, 8)
    is_lattice = is_frame_part_of_lattice(grid, frame, 1)
    assert is_lattice == True, f"Correct Lattice Grid: Frame {frame}"

    # Interrupted Lattice Grid
    grid = [
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 9, 1, 0],  # Break in the lattice pattern
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    ]
    frame = (2, 2, 5, 5)
    is_lattice = is_frame_part_of_lattice(grid, frame, 1)
    assert is_lattice == False, f"Interrupted Lattice Grid: Frame {frame}"

    # Break outside frames that fit in the grid does not affect lattice check
    grid = [
        [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
        [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
        [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 9],  # Break near edge
        [0, 0, 2, 0, 0, 2, 0, 0, 2, 0]
    ]
    frame = (2, 2, 5, 5)
    is_lattice = is_frame_part_of_lattice(grid, frame, 2)
    assert is_lattice == True, f"Break outside frames: Frame {frame}"
