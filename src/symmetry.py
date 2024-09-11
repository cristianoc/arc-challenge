import numpy as np
from typing import Optional
from objects import Object


def check_vertical_symmetry_with_unknowns(grid: Object, period: int, unknown: int):
    """
    Check if rows repeat every 'period' rows, allowing for unknown cells.
    """
    width, height = grid.size
    for x in range(width):
        for y in range(period, height):
            if (
                grid[x, y] != unknown
                and grid[x, y - period] != unknown
                and grid[x, y] != grid[x, y - period]
            ):
                return False
    return True


def check_horizontal_symmetry_with_unknowns(grid: Object, period: int, unknown: int):
    """
    Check if columns repeat every 'period' columns, allowing for unknown cells.
    """
    width, height = grid.size
    for x in range(period, width):
        for y in range(height):
            if (
                grid[x, y] != unknown
                and grid[x - period, y] != unknown
                and grid[x, y] != grid[x - period, y]
            ):
                return False
    return True


def check_diagonal_symmetry_with_unknowns(grid: Object, period: int, unknown: int):
    """
    Check if the grid has diagonal symmetry with a given period, allowing for unknown cells.
    Moving diagonally, we check that the same element is found every 'period' steps, without wrapping around.
    """
    width, height = grid.size

    # Only iterate over the range where diagonal steps are valid
    for x in range(width):
        for y in range(height):
            next_x = x + period
            next_y = y + period
            if next_x >= width or next_y >= height:
                continue

            if (
                grid[x, y] != unknown
                and grid[next_x, next_y] != unknown
                and grid[x, y] != grid[next_x, next_y]
            ):
                return False
    return True


def find_symmetry_with_unknowns(grid: Object, unknown: int):
    """
    Find the smallest periods px, py, and pd (if any) with unknowns.
    """
    width, height = grid.size

    # Find smallest horizontal symmetry modulo px
    px = None
    for possible_px in range(1, width // 2 + 1):
        if width % possible_px == 0 and check_horizontal_symmetry_with_unknowns(
            grid, possible_px, unknown
        ):
            px = possible_px
            break

    # Find smallest vertical symmetry modulo py
    py = None
    for possible_py in range(1, height // 2 + 1):
        if height % possible_py == 0 and check_vertical_symmetry_with_unknowns(
            grid, possible_py, unknown
        ):
            py = possible_py
            break

    # Find smallest diagonal symmetry modulo pd
    pd = None
    # Ensure the grid is square for diagonal symmetry
    if width == height:
        for possible_pd in range(1, width // 2 + 1):
            if check_diagonal_symmetry_with_unknowns(grid, possible_pd, unknown):
                pd = possible_pd
                break

    return px, py, pd


def find_source_value(
    filled_grid: Object,
    x_dest: int,
    y_dest: int,
    px: Optional[int],
    py: Optional[int],
    pd: Optional[int],
    unknown: int,
):
    """
    Find a source value for the given destination coordinates based on symmetry.
    """
    width, height = filled_grid.size
    for x_src in range(x_dest % px, width, px) if px is not None else [x_dest]:
        for y_src in range(y_dest % py, height, py) if py is not None else [y_dest]:
            if filled_grid[x_src, y_src] != unknown:
                return filled_grid[x_src, y_src]

    # Search based on diagonal (pd) symmetry if provided
    if pd is not None and width == height:
        size = (width // pd) * pd

        # Walk along the diagonal in both directions, by starting negative and going positive
        for i in range(-size, size, pd):
            x_src = x_dest + i
            y_src = y_dest + i

            if (
                0 <= x_src < size
                and 0 <= y_src < size
                and filled_grid[x_src, y_src] != unknown
            ):
                return filled_grid[x_src, y_src]

    return unknown


def fill_grid(
    grid: Object,
    px: Optional[int],
    py: Optional[int],
    pd: Optional[int],
    unknown: int = 0,
):
    """
    Fills the unknown cells in a grid based on detected horizontal and vertical symmetries.

    This function fills each unknown cell in the grid by propagating values from known cells,
    using the provided horizontal (px) and vertical (py) symmetry periods. It starts at each
    destination cell and looks for a matching source cell at symmetrical positions, based on
    the periods px and py.

    Args:
        grid (Object): The grid containing known and unknown values to be filled.
        px (Optional[int]): The horizontal symmetry period. If None, no horizontal symmetry is used.
        py (Optional[int]): The vertical symmetry period. If None, no vertical symmetry is used.
        unknown (int): The value representing unknown cells in the grid, which will be filled.

    Returns:
        Object: A new grid with all unknown cells filled using the provided symmetry periods.
    """
    width, height = grid.size
    filled_grid = grid.copy()

    # Loop over all destination cells
    for x_dest in range(width):
        for y_dest in range(height):
            if (
                filled_grid[x_dest, y_dest] == unknown
            ):  # If the destination cell is unknown
                filled_grid[x_dest, y_dest] = find_source_value(
                    filled_grid, x_dest, y_dest, px, py, pd, unknown
                )

    return filled_grid


def test_find_and_fill_symmetry():

    grid_xy = Object(
        np.array(
            [
                [0, 0, 0, 2, 1, 2],
                [0, 7, 0, 0, 3, 7],
                [1, 2, 0, 2, 1, 2],
                [0, 0, 3, 0, 0, 0],
                [1, 2, 0, 2, 1, 2],
                [0, 0, 3, 0, 0, 0],
            ]
        )
    )

    grid_y = Object(
        np.array(
            [
                [1, 2, 1, 9, 1, 2],
                [3, 7, 3, 7, 3, 7],
                [1, 0, 1, 9, 0, 2],
                [3, 7, 3, 7, 3, 7],
                [0, 0, 1, 9, 0, 0],
                [3, 7, 3, 7, 0, 0],
            ]
        )
    )

    grid_diagonal = Object(
        np.array(
            [
                [1, 2, 1, 2, 4, 2],
                [3, 0, 3, 0, 3, 7],
                [1, 2, 0, 2, 1, 2],
                [3, 7, 3, 7, 3, 7],
                [5, 9, 0, 2, 1, 2],
                [8, 7, 3, 7, 3, 7],
            ]
        )
    )

    def test_grid(grid: Object, unknown: int, title: str):
        px, py, pd = find_symmetry_with_unknowns(grid, unknown)
        print(f"{title}: px: {px}, py: {py}, pd: {pd}")
        filled_grid = fill_grid(grid, px, py, pd, unknown)
        assert unknown not in filled_grid._data
        return filled_grid

    test_grid(grid_xy, 0, "grid_xy") # horizontal and vertical symmetry
    test_grid(grid_y, 0, "grid_y") # vertical symmetry
    test_grid(grid_diagonal, 0, "grid_diagonal") # diagonal symmetry


if __name__ == "__main__":
    test_find_and_fill_symmetry()
