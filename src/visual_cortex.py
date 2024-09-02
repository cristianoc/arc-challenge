import time
from typing import List, Optional, Tuple
import random

from grid_types import GridData, logger, Cell
from objects import Object


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
                row_sum[i][j] = row_sum[i][j - 1] + 1 if j > 0 else 1
                col_sum[i][j] = col_sum[i - 1][j] + 1 if i > 0 else 1

    return row_sum, col_sum


def is_frame_dp(
    row_sum: GridData, col_sum: GridData, top: int, left: int, bottom: int, right: int
) -> bool:
    """Check if the rectangle defined by (top, left) to (bottom, right) forms a frame using precomputed sums."""
    if (
        row_sum[top][right] - (row_sum[top][left - 1] if left > 0 else 0)
        != right - left + 1
    ):
        return False
    if (
        row_sum[bottom][right] - (row_sum[bottom][left - 1] if left > 0 else 0)
        != right - left + 1
    ):
        return False
    if (
        col_sum[bottom][left] - (col_sum[top - 1][left] if top > 0 else 0)
        != bottom - top + 1
    ):
        return False
    if (
        col_sum[bottom][right] - (col_sum[top - 1][right] if top > 0 else 0)
        != bottom - top + 1
    ):
        return False
    return True


def is_frame(
    grid: GridData, top: int, left: int, bottom: int, right: int, color: int
) -> bool:
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
                    if (
                        is_frame_dp(row_sum, col_sum, top, left, bottom, right)
                        if row_sum and col_sum
                        else is_frame(
                            grid, top, left, bottom, right, top_left_corner_color
                        )
                    ):
                        area = calculate_area(top, left, bottom, right)
                        if area > max_area:
                            max_area = area
                            max_frame = (top, left, bottom, right)

    return max_frame


def find_smallest_frame(
    grid: GridData, color: Optional[int], min_size: Optional[Tuple[int, int]] = None
) -> Optional[Frame]:
    """
    Find the smallest frame in the grid that has all border cells matching the specified color.
    If the color is None, the function will find the smallest frame with any color.
    """
    row_sum, col_sum = precompute_sums(grid, color) if color else (None, None)
    min_area = float("inf")
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
                    if (
                        is_frame_dp(row_sum, col_sum, top, left, bottom, right)
                        if row_sum and col_sum
                        else is_frame(
                            grid, top, left, bottom, right, top_left_corner_color
                        )
                    ):
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
    for y in range(start_y, rows, frame_height - 1):
        for x in range(start_x, cols, frame_width - 1):
            # Check if the frame fits within the grid bounds
            if y + frame_height > rows or x + frame_width > cols:
                continue
            # Check top and bottom borders of the frame
            for i in range(frame_width):
                if y < rows and (x + i) < cols:
                    if (
                        grid[y][x + i] != foreground
                        or grid[y + frame_height - 1][x + i] != foreground
                    ):
                        return False
            # Check left and right borders of the frame
            for j in range(frame_height):
                if y + j < rows and x < cols:
                    if (
                        grid[y + j][x] != foreground
                        or grid[y + j][x + frame_width - 1] != foreground
                    ):
                        return False
    return True


# A Subgrid is a list of lists of grids
Subgrid = List[List[Object]]


def find_dividing_lines(grid: Object, color: int) -> Tuple[List[int], List[int]]:
    """Find the indices of vertical and horizontal lines that span the entire grid."""

    horizontal_lines: List[int] = []
    vertical_lines: List[int] = []

    for i in range(grid.height):
        if all(grid.data[i][j] == color for j in range(grid.width)):
            horizontal_lines.append(i)

    for j in range(grid.width):
        if all(grid.data[i][j] == color for i in range(grid.height)):
            vertical_lines.append(j)

    return horizontal_lines, vertical_lines


def extract_subgrid_of_color(grid: Object, color: int) -> Optional[Subgrid]:
    """Extract a subgrid from the grid based on vertical and horizontal dividing lines of the same color."""
    horizontal_lines, vertical_lines = find_dividing_lines(grid, color)

    if not horizontal_lines or not vertical_lines:
        return None  # No dividing lines found
    for i in range(len(horizontal_lines) - 1):
        if horizontal_lines[i] + 1 == horizontal_lines[i + 1]:
            return None
    for j in range(len(vertical_lines) - 1):
        if vertical_lines[j] + 1 == vertical_lines[j + 1]:
            return None

    subgrid: Subgrid = []
    prev_h = 0

    for h in horizontal_lines + [grid.height]:
        row: List[Object] = []
        prev_v = 0
        for v in vertical_lines + [grid.width]:
            # Extract the subgrid bounded by (prev_h, prev_v) and (h-1, v-1)
            if prev_v == v or prev_h == h:
                continue
            sub_grid_data = [row[prev_v:v] for row in grid.data[prev_h:h]]
            row.append(Object(sub_grid_data))
            prev_v = v + 1
        subgrid.append(row)
        prev_h = h + 1

    return subgrid


def extract_subgrid(grid: Object, color: Optional[int]) -> Optional[Subgrid]:
    if color is not None:
        return extract_subgrid_of_color(grid, color)
    for c in grid.get_colors():
        subgrid = extract_subgrid_of_color(grid, c)
        if subgrid:
            return subgrid


def eval_with_lattice_check():
    # Define sizes
    width = 50
    height = 30

    # Create test grids
    empty_grid = [[0 for _ in range(width)] for _ in range(width)]
    full_grid = [[1 for _ in range(width)] for _ in range(width)]
    square_grid = [[random.choice([0, 1]) for _ in range(width)] for _ in range(width)]
    wide_grid = [[random.choice([0, 1]) for _ in range(width)] for _ in range(height)]
    tall_grid = [[random.choice([0, 1]) for _ in range(height)] for _ in range(width)]
    biased_random_grid = [
        [1 if random.random() < 0.9 else 0 for _ in range(width)] for _ in range(width)
    ]  # Biased random grid

    # List of grids and their names
    grids = [
        ("Empty Square Grid", empty_grid),
        ("Full Square Grid", full_grid),
        ("Random Square Grid", square_grid),
        ("Random Wide Grid", wide_grid),
        ("Random Tall Grid", tall_grid),
        ("Biased Random Grid", biased_random_grid),
    ]

    # Run tests for each grid
    for grid_name, grid in grids:
        start_time = time.time()
        foreground = 1
        max_frame = find_largest_frame(grid, foreground)
        max_area = calculate_area(*max_frame) if max_frame else 0
        is_lattice = (
            is_frame_part_of_lattice(grid, max_frame, foreground)
            if max_frame
            else False
        )
        end_time = time.time()

        execution_time = end_time - start_time

        if not max_frame:
            logger.info(
                f"{grid_name}: No valid frame found. "
                f"Time: {execution_time:.6f} seconds\n"
            )
        else:
            start_row, start_col, end_row, end_col = max_frame
            frame_height = end_row - start_row + 1
            frame_width = end_col - start_col + 1
            logger.info(
                f"{grid_name}: Frame at ({start_row},{start_col}) to ({end_row},{end_col}), "
                f"Size: {frame_height}x{frame_width}, Area: {max_area}, "
                f"Part of lattice: {is_lattice}, "
                f"Time: {execution_time:.6f} seconds\n"
            )


def test_lattices():
    # Correct Lattice Grid
    grid = [
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
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
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
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
        [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
    ]
    frame = (2, 2, 5, 5)
    is_lattice = is_frame_part_of_lattice(grid, frame, 2)
    assert is_lattice == True, f"Break outside frames: Frame {frame}"


def test_subgrid_extraction():
    # Example grid with dividing lines
    grid = Object(
        [
            [2, 2, 1, 3, 3, 1, 4, 4, 1, 5],
            [2, 2, 1, 3, 3, 1, 4, 4, 1, 5],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [6, 6, 1, 7, 7, 1, 8, 8, 1, 9],
            [6, 6, 1, 7, 7, 1, 8, 8, 1, 9],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 1, 3, 3, 1, 4, 4, 1, 5],
        ]
    )

    subgrid = extract_subgrid(grid, 1)
    assert subgrid is not None, "Test failed: No subgrid extracted"
    height = len(subgrid)
    width = len(subgrid[0])
    logger.info(f"Subgrid height: {height}, Subgrid width: {width}")
    assert (height, width) == (
        3,
        4,
    ), f"Test failed: Subgrid dimensions: {height}x{width}"
    assert subgrid[0][0] == Object([[2, 2], [2, 2]]), "Test failed: Subgrid[0][0]"
    assert subgrid[0][1] == Object([[3, 3], [3, 3]]), "Test failed: Subgrid[0][1]"
    assert subgrid[0][3] == Object([[5], [5]]), "Test failed: Subgrid[0][3]"
    assert subgrid[2][3] == Object([[5]]), "Test failed: Subgrid[2][3]"


def extract_object_by_color(grid: Object, color: int) -> Object:
    # find the bounding box of the object with the given color
    rows = grid.height
    cols = grid.width
    top = rows
    left = cols
    bottom = 0
    right = 0
    for i in range(rows):
        for j in range(cols):
            if grid.data[i][j] == color:
                top = min(top, i)
                left = min(left, j)
                bottom = max(bottom, i)
                right = max(right, j)
    origin = (top, left)
    data = [row[left : right + 1] for row in grid.data[top : bottom + 1]]
    # remove other colors
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] != color:
                data[i][j] = 0
    return Object(data, origin)


def find_colored_objects(grid: Object) -> List[Object]:
    """
    Finds and returns a list of all distinct objects within the grid based on color.

    This function scans the grid, identifies all unique colors (excluding the
    background color), and extracts each object corresponding to these colors.
    Each object is represented as an instance of the `Object` class.
    """
    grid_as_object = Object(grid.data)
    background_color = grid_as_object.main_color(allow_black=True)
    colors = grid.get_colors(allow_black=True)
    objects: List[Object] = []
    for color in colors:
        if color == background_color:
            continue
        object = extract_object_by_color(grid, color)
        objects.append(object)
    return objects


def find_rectangular_objects(data: GridData, allow_multicolor: bool) -> List[Object]:
    objects: List[Object] = []
    rows, cols = len(data), len(data[0])

    def cell_contained_in_objects(cell: Cell) -> bool:
        return any(obj.contains_cell(cell) for obj in objects)

    def is_valid_rectangle(origin: Cell, height: int, width: int, color: int) -> bool:
        start_r, start_c = origin
        if (
            start_r < 0
            or start_c < 0
            or start_r + height > rows
            or start_c + width > cols
        ):
            return False
        for r in range(start_r, start_r + height):
            for c in range(start_c, start_c + width):
                if not allow_multicolor and data[r][c] != color and data[r][c] != 0:
                    return False
        # check that the first and last rows and columns are not all 0
        if all(data[start_r][c] == 0 for c in range(start_c, start_c + width)):
            return False
        if all(
            data[start_r + height - 1][c] == 0 for c in range(start_c, start_c + width)
        ):
            return False
        if all(data[r][start_c] == 0 for r in range(start_r, start_r + height)):
            return False
        if all(
            data[r][start_c + width - 1] == 0 for r in range(start_r, start_r + height)
        ):
            return False
        return True

    for r in range(rows):
        for c in range(cols):
            if not cell_contained_in_objects((r, c)) and data[r][c] != 0:
                main_color = data[r][c]
                origin: Cell = (r, c)
                height, width = 1, 1

                logger.debug(f"\nstarting new object at {origin}")

                while True:
                    expanded = False

                    # Try expanding rightwards
                    if is_valid_rectangle(origin, height, width + 1, main_color):
                        width += 1
                        expanded = True
                        logger.debug(
                            f"expanded rightwards new dimensions: {origin, height, width}"
                        )

                    # Try expanding downwards
                    if is_valid_rectangle(origin, height + 1, width, main_color):
                        height += 1
                        expanded = True
                        logger.debug(
                            f"expanded downwards new dimensions: {origin, height, width}"
                        )

                    # Try expanding right-downwards
                    if is_valid_rectangle(origin, height + 1, width + 1, main_color):
                        height += 1
                        width += 1
                        expanded = True
                        logger.debug(
                            f"expanded right-downwards new dimensions: {origin, height, width}"
                        )

                    # Try expanding leftwards
                    if is_valid_rectangle(
                        (origin[0], origin[1] - 1), height, width, main_color
                    ):
                        origin = (origin[0], origin[1] - 1)
                        width += 1
                        expanded = True
                        logger.debug(
                            f"expanded leftwards new dimensions: {origin, height, width}"
                        )

                    # Try expanding upwards
                    if is_valid_rectangle(
                        (origin[0] - 1, origin[1]), height, width, main_color
                    ):
                        origin = (origin[0] - 1, origin[1])
                        height += 1
                        expanded = True
                        logger.debug(
                            f"expanded upwards new dimensions: {origin, height, width}"
                        )

                    # If no further expansion is possible, break the loop
                    if not expanded:
                        break

                # Once the largest rectangle is found, create the grid data for the object
                object_grid_data = [
                    [data[r][c] for c in range(origin[1], origin[1] + width)]
                    for r in range(origin[0], origin[0] + height)
                ]
                current_object = Object(
                    object_grid_data,
                    origin,
                )
                objects.append(current_object)

    return objects


def test_detect_rectangular_objects():
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    objects: List[Object] = find_rectangular_objects(grid, allow_multicolor=False)
    for obj in objects:
        logger.info(f"Detected rectangular object: {obj}")
    object_dims = [(obj.origin, obj.size) for obj in objects]
    assert object_dims == [((1, 1), (4, 4))]


def test_several_rectangular_objects_of_different_color():
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 2, 0],
        [0, 0, 1, 0, 2, 2],
        [0, 0, 0, 1, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    objects = find_rectangular_objects(grid, allow_multicolor=False)
    for obj in objects:
        logger.info(f"Detected rectangular object: {obj}")
    object_dims = [(obj.origin, obj.size) for obj in objects]
    assert object_dims == [((1, 1), (4, 3)), ((2, 4), (3, 2))]
