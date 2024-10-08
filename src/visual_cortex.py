import random
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from numpy import ndarray

from grid_types import Cell
from logger import logger

# To avoid circular imports
if TYPE_CHECKING:
    from objects import Object as Object_t
else:
    Object_t = None


"""
A Frame represents a rectangular region in a grid, defined by the coordinates (top, left, bottom, right).

A frame is specifically characterized by having all cells along the border of the rectangle filled (i.e., with a value of 1).
The interior cells of the rectangle are not considered part of the frame.
"""
Frame = Tuple[int, int, int, int]


def calculate_area(top: int, left: int, bottom: int, right: int) -> int:
    """Calculate the area of the rectangle defined by the corners (top, left) to (bottom, right)."""
    return (bottom - top + 1) * (right - left + 1)


def frame_is_color(
    grid: Object_t,
    top: int,
    left: int,
    bottom: int,
    right: int,
    color: int,
    corner_extent: int,
) -> bool:
    """
    Check if the cells around the corners of a rectangle defined by
    (top, left) to (bottom, right) are of the specified color.

    Args:
        grid (Object_t): The grid to check.
        top (int): The top boundary of the rectangle.
        left (int): The left boundary of the rectangle.
        bottom (int): The bottom boundary of the rectangle.
        right (int): The right boundary of the rectangle.
        color (int): The color to check for.
        corner_extent (int): The extent of cells around each corner to check.
            - corner_extent=1 checks the 4 corner cells.
            - corner_extent=2 checks the 3 cells around each corner, including
              one on each edge adjacent to the corner.

    Returns:
        bool: True if the specified corner cells match the given color,
              False otherwise.
    """

    """Check if the specified cells of the rectangle defined by (top, left) to (bottom, right) are of the given color."""
    if left < 0 or top < 0 or right >= grid.width or bottom >= grid.height:
        return True

    width = right - left + 1
    height = bottom - top + 1

    if corner_extent == 0:
        offset = max(width, height)  # Check all cells if not specified
    else:
        offset = corner_extent

    # Extract the frame edges
    top_edge = grid._data[top, left : right + 1]
    bottom_edge = grid._data[bottom, left : right + 1]
    left_edge = grid._data[top : bottom + 1, left]
    right_edge = grid._data[top : bottom + 1, right]

    if not np.all(bottom_edge[:offset] == color):
        return False
    if not np.all(top_edge[:offset] == color):
        return False
    if not np.all(left_edge[:offset] == color):
        return False
    if not np.all(right_edge[:offset] == color):
        return False

    if corner_extent == 0:
        return True  # no need to check the negative offset

    if not np.all(bottom_edge[-offset:] == color):
        return False
    if not np.all(top_edge[-offset:] == color):
        return False
    if not np.all(left_edge[-offset:] == color):
        return False
    if not np.all(right_edge[-offset:] == color):
        return False

    return True


def frame_is_not_color(
    grid: Object_t,
    top: int,
    left: int,
    bottom: int,
    right: int,
    color: int,
    corner_extent: int,
) -> bool:
    if left < 0 or top < 0 or right >= grid.width or bottom >= grid.height:
        return True

    width = right - left + 1
    height = bottom - top + 1

    if corner_extent == 0:
        offset = max(width, height)  # Check all cells if not specified
    else:
        offset = corner_extent

    # Extract the frame edges
    top_edge = grid._data[top, left : right + 1]
    bottom_edge = grid._data[bottom, left : right + 1]
    left_edge = grid._data[top : bottom + 1, left]
    right_edge = grid._data[top : bottom + 1, right]

    if not np.all(bottom_edge[:offset] != color):
        return False
    if not np.all(top_edge[:offset] != color):
        return False
    if not np.all(left_edge[:offset] != color):
        return False
    if not np.all(right_edge[:offset] != color):
        return False

    if corner_extent == 0:
        return True  # no need to check the negative offset

    if not np.all(bottom_edge[-offset:] != color):
        return False
    if not np.all(top_edge[-offset:] != color):
        return False
    if not np.all(left_edge[-offset:] != color):
        return False
    if not np.all(right_edge[-offset:] != color):
        return False

    return True


def is_frame(
    grid: Object_t,
    top: int,
    left: int,
    bottom: int,
    right: int,
    color: Optional[int],
    background: int,
    check_precise: bool,
    corner_extent: int,
) -> bool:
    """Check if the rectangle defined by (top, left) to (bottom, right) forms a frame."""
    if color is not None:
        frame_color_ok = frame_is_color(
            grid, top, left, bottom, right, color, corner_extent
        )
    else:
        frame_color_ok = frame_is_not_color(
            grid, top, left, bottom, right, background, corner_extent
        )
    if not frame_color_ok:
        return False

    if not check_precise:
        return True
    # check that the inside is not color
    if color is not None:
        inside_color_ok = frame_is_not_color(
            grid, top + 1, left + 1, bottom - 1, right - 1, color, corner_extent
        )
    else:
        inside_color_ok = frame_is_color(
            grid, top + 1, left + 1, bottom - 1, right - 1, background, corner_extent
        )
    if not inside_color_ok:
        return False
    # check that the outside is not color
    if color is not None:
        outside_color_ok = frame_is_not_color(
            grid, top - 1, left - 1, bottom + 1, right + 1, color, corner_extent
        )
    else:
        outside_color_ok = frame_is_color(
            grid, top - 1, left - 1, bottom + 1, right + 1, background, corner_extent
        )
    if not outside_color_ok:
        return False
    return True


def find_largest_frame(
    grid: Object_t,
    color: Optional[int],
    background: int = 0,
    check_precise: bool = False,
    invert_min_max: bool = False,
    corner_extent: int = 0,
) -> Optional[Frame]:
    """
    Find the largest frame in the grid that has all border cells matching the specified color.
    If the color is None, the function will find the largest frame with any color.
    """
    max_area = 0
    max_frame = None

    width, height = grid.size

    for top in range(height):
        for left in range(width):
            top_left_corner_color = grid[left, top]
            if color is None and top_left_corner_color == 0:
                continue
            if color is not None and top_left_corner_color != color:
                continue
            for bottom in range(top + 1, height):
                # Early termination if the potential max area is less than the current max_area
                if (height - top) * (width - left) <= max_area:
                    break
                for right in range(left + 1, width):
                    frame_color = None if color is None else top_left_corner_color
                    if is_frame(
                        grid,
                        top,
                        left,
                        bottom,
                        right,
                        frame_color,
                        background,
                        check_precise,
                        corner_extent,
                    ):
                        area = calculate_area(top, left, bottom, right)
                        if ((not invert_min_max) and area > max_area) or (
                            invert_min_max and area < max_area
                        ):
                            max_area = area
                            max_frame = (top, left, bottom, right)

    return max_frame


def find_smallest_frame(
    grid: Object_t,
    color: Optional[int],
    background: int = 0,
    check_precise: bool = False,
    corner_extent: int = 0,
) -> Optional[Frame]:
    return find_largest_frame(
        grid,
        color,
        background,
        check_precise,
        invert_min_max=True,
        corner_extent=corner_extent,
    )


def is_frame_part_of_lattice(grid: Object_t, frame: Frame, foreground: int) -> bool:
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
    - grid: The 2D grid where each cell contains an integer representing a color.
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
    rows = grid.height
    cols = grid.width

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
                        grid[x + i, y] != foreground
                        or grid[x + i, y + frame_height - 1] != foreground
                    ):
                        return False
            # Check left and right borders of the frame
            for j in range(frame_height):
                if y + j < rows and x < cols:
                    if (
                        grid[x, y + j] != foreground
                        or grid[x + frame_width - 1, y + j] != foreground
                    ):
                        return False
    return True


def find_dividing_lines(grid: Object_t, color: int) -> Tuple[List[int], List[int]]:
    """Find the indices of vertical and horizontal lines that span the entire grid."""

    horizontal_lines: List[int] = []
    vertical_lines: List[int] = []

    for i in range(grid.height):
        if all(grid[j, i] == color for j in range(grid.width)):
            horizontal_lines.append(i)

    for j in range(grid.width):
        if all(grid[j, i] == color for i in range(grid.height)):
            vertical_lines.append(j)

    return horizontal_lines, vertical_lines


def extract_subgrid_of_color(
    grid: Object_t, color: int
) -> Optional[List[List[Object_t]]]:
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

    subgrid: List[List[Object_t]] = []
    prev_h = 0

    for h in horizontal_lines + [grid.height]:
        row: List[Object_t] = []
        prev_v = 0
        for v in vertical_lines + [grid.width]:
            # Extract the subgrid bounded by (prev_h, prev_v) and (h-1, v-1)
            if prev_v == v or prev_h == h:
                continue
            sub_grid_data = grid._data[prev_h:h, prev_v:v]
            from objects import Object

            row.append(Object(np.array(sub_grid_data)))
            prev_v = v + 1
        subgrid.append(row)
        prev_h = h + 1

    return subgrid


def extract_lattice_subgrids(
    grid: Object_t, color: Optional[int] = None
) -> Optional[List[List[Object_t]]]:
    """
    Extracts subgrids from the grid based on dividing lines of the specified color.
    If color is None, attempts to find dividing lines of any color.


    Original Grid:
    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+
    | 4 | 5 | 6 |
    +---+---+---+
    | 7 | 8 | 9 |
    +---+---+---+

    Extracted Subgrids: the 9 subgrids

    **Parameters:**
    - `grid (Object_t)`: The grid to analyze.
    - `color (Optional[int])`: The color of the dividing lines. If `None`, any color is considered.

    **Returns:**
    - `Optional[List[List['Object']]]`: A nested list of subgrids organized as rows of subgrids if extraction is successful; otherwise, `None`.

    """
    if color is not None:
        return extract_subgrid_of_color(grid, color)
    for c in grid.get_colors():
        subgrid = extract_subgrid_of_color(grid, c)
        if subgrid:
            return subgrid
    return None


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
    from objects import Object

    grid = Object(
        np.array(
            [
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            ]
        )
    )
    frame = (2, 2, 5, 8)
    is_lattice = is_frame_part_of_lattice(grid, frame, 1)
    assert is_lattice == True, f"Correct Lattice Grid: Frame {frame}"

    # Interrupted Lattice Grid
    grid = Object(
        np.array(
            [
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 9, 1, 0],  # Break in the lattice pattern
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            ]
        )
    )
    frame = (2, 2, 5, 5)
    is_lattice = is_frame_part_of_lattice(grid, frame, 1)
    assert is_lattice == False, f"Interrupted Lattice Grid: Frame {frame}"

    # Break outside frames that fit in the grid does not affect lattice check
    grid = Object(
        np.array(
            [
                [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
                [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
                [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 9],  # Break near edge
                [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
            ]
        )
    )
    frame = (2, 2, 5, 5)
    is_lattice = is_frame_part_of_lattice(grid, frame, 2)
    assert is_lattice == True, f"Break outside frames: Frame {frame}"


def test_subgrid_extraction():
    # Example grid with dividing lines
    from objects import Object

    grid = Object(
        np.array(
            [
                [2, 2, 1, 3, 3, 1, 4, 4, 1, 5],
                [2, 2, 1, 3, 3, 1, 4, 4, 1, 5],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [6, 6, 1, 7, 7, 1, 8, 8, 1, 9],
                [6, 6, 1, 7, 7, 1, 8, 8, 1, 9],
                [6, 6, 1, 7, 7, 1, 8, 8, 1, 9],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 1, 3, 3, 1, 4, 4, 1, 5],
            ]
        )
    )

    subgrid = extract_lattice_subgrids(grid, 1)
    assert subgrid is not None, "Test failed: No subgrid extracted"
    height = len(subgrid)
    width = len(subgrid[0])
    logger.info(f"Subgrid height: {height}, Subgrid width: {width}")
    assert (height, width) == (
        3,
        4,
    ), f"Test failed: Subgrid dimensions: {height}x{width}"
    assert subgrid[0][0] == Object(
        np.array([[2, 2], [2, 2]])
    ), "Test failed: Subgrid[0][0]"
    assert subgrid[0][1] == Object(
        np.array([[3, 3], [3, 3]])
    ), "Test failed: Subgrid[0][1]"
    assert subgrid[0][3] == Object(np.array([[5], [5]])), "Test failed: Subgrid[0][3]"
    assert subgrid[2][3] == Object(np.array([[5]])), "Test failed: Subgrid[2][3]"


def extract_object_by_color(grid: Object_t, color: int) -> Object_t:
    # find the bounding box of the object with the given color
    rows = grid.height
    cols = grid.width
    top = rows
    left = cols
    bottom = 0
    right = 0
    for i in range(rows):
        for j in range(cols):
            if grid[j, i] == color:
                top = min(top, i)
                left = min(left, j)
                bottom = max(bottom, i)
                right = max(right, j)
    origin = (left, top)

    # Slicing the array for the specified region
    data = grid._data[top : bottom + 1, left : right + 1].copy()

    # Remove other colors by setting elements not equal to the desired color to 0
    data[data != color] = 0

    from objects import Object

    return Object(np.array(data), origin)


def find_colored_objects(
    grid: Object_t, background_color: Optional[int]
) -> List[Object_t]:
    """
    Finds and returns a list of all distinct objects within the grid based on color.

    This function scans the grid, identifies all unique colors (excluding the
    background color), and extracts each object corresponding to these colors.
    Each object is represented as an instance of the `Object` class.
    """
    from objects import Object

    colors = grid.get_colors(allow_black=True)
    objects: List[Object] = []
    for color in colors:
        if background_color is not None and color == background_color:
            continue
        object = extract_object_by_color(grid, color)
        objects.append(object)
    return objects


def find_rectangular_objects(
    grid: Object_t, allow_multicolor: bool, background_color: int = 0
) -> List[Object_t]:
    objects: List[Object_t] = []
    rows, cols = grid.height, grid.width
    data = grid._data

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
                if (
                    not allow_multicolor
                    and data[r, c] != color
                    and data[r, c] != background_color
                ):
                    return False
        # check that the first and last rows and columns are not all 0
        if all(
            data[start_r, c] == background_color
            for c in range(start_c, start_c + width)
        ):
            return False
        if all(
            data[start_r + height - 1, c] == background_color
            for c in range(start_c, start_c + width)
        ):
            return False
        if all(
            data[r, start_c] == background_color
            for r in range(start_r, start_r + height)
        ):
            return False
        if all(
            data[r, start_c + width - 1] == background_color
            for r in range(start_r, start_r + height)
        ):
            return False
        return True

    for r in range(rows):
        for c in range(cols):
            if not cell_contained_in_objects((r, c)) and data[r, c] != background_color:
                main_color = data[r, c]
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
                    [data[r, c] for c in range(origin[1], origin[1] + width)]
                    for r in range(origin[0], origin[0] + height)
                ]
                from objects import Object

                current_object = Object(
                    np.array(object_grid_data),
                    origin,
                )
                objects.append(current_object)

    return objects


def regularity_score(grid: Object_t) -> float:
    """
    Score how regular a grid is by scoring every cell.
    A cell is penalized if one of those in the 8 directions around it has the same color.
    The score of the cell is the sum of those in the 8 directions.
    The score of the grid is the average score of all cells.
    """
    width, height = grid.width, grid.height
    total_score = 0
    from grid_types import DIRECTIONS8

    for x in range(width):
        for y in range(height):
            cell_score = 0
            cell_color = grid[x, y]
            for dx, dy in DIRECTIONS8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[nx, ny] == cell_color:
                        cell_score += 1
            total_score += cell_score

    return total_score / (width * height * 8)


def test_detect_rectangular_objects() -> None:
    from objects import Object

    grid = Object(
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )

    objects: List[Object_t] = find_rectangular_objects(grid, allow_multicolor=False)
    for obj in objects:
        logger.info(f"Detected rectangular object: {obj}")
    object_dims = [(obj.origin, obj.size) for obj in objects]
    assert object_dims == [((1, 1), (4, 4))]


def test_several_rectangular_objects_of_different_color():
    from objects import Object

    grid = Object(
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 2, 0],
                [0, 0, 1, 0, 2, 2],
                [0, 0, 0, 1, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )

    objects = find_rectangular_objects(grid, allow_multicolor=False)
    for obj in objects:
        logger.info(f"Detected rectangular object: {obj}")
    object_dims = [(obj.origin, obj.size) for obj in objects]
    assert object_dims == [((1, 1), (3, 4)), ((2, 4), (2, 3))]


def test_find_largest_frame():
    from objects import Object

    grid = Object(
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 1],
                [0, 2, 2, 0, 1, 1],
            ]
        )
    )

    frame = find_largest_frame(grid, color=None, corner_extent=1)
    assert frame == (1, 1, 5, 5)

    frame = find_largest_frame(grid, color=None, corner_extent=0)
    assert frame == None

    frame = find_largest_frame(grid, color=None, corner_extent=2)
    assert frame == (1, 1, 5, 5)

    frame = find_largest_frame(grid, color=None, corner_extent=3)
    assert frame == None
