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


def find_symmetry_with_unknowns(grid: Object, unknown: int):
    """
    Find the smallest periods px and py (if any) with unknowns.
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

    return px, py


# Function to fill the grid by starting at dest modulo px and py, proceed in steps of px and py
def fill_grid(
    grid: Object, px: Optional[int] = None, py: Optional[int] = None, unknown: int = 0
):
    width, height = grid.size
    filled_grid = grid.copy()

    # Loop over all destination cells
    for x_dest in range(width):
        for y_dest in range(height):
            if (
                filled_grid[x_dest, y_dest] == unknown
            ):  # If the destination cell is unknown
                # Loop over all source cells but start from (y_dest % py, x_dest % px)
                for x_src in (
                    range(x_dest % px, width, px) if px is not None else [x_dest]
                ):
                    for y_src in (
                        range(y_dest % py, height, py) if py is not None else [y_dest]
                    ):
                        # Check if the source is valid (not unknown)
                        if filled_grid[x_src, y_src] != unknown:
                            filled_grid[x_dest, y_dest] = filled_grid[x_src, y_src]
                            break  # Fill the cell and break out of source loop

    return filled_grid


def test_find_and_fill_symmetry():

    def find_and_fill_symmetry(grid: Object, unknown: int):
        px, py = find_symmetry_with_unknowns(grid, unknown)
        print(f"px: {px}, py: {py}")
        filled_grid = fill_grid(grid, px, py, unknown)
        return filled_grid

    # Example usage:
    grid_with_unknowns1 = Object(
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

    unknown = 0
    # Find and fill symmetries
    filled_grid1 = find_and_fill_symmetry(grid_with_unknowns1, unknown)

    print(f"Filled grid 1: {filled_grid1}")

    assert unknown not in filled_grid1._data

    xy_grid = Object(
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

    filled_grid2 = find_and_fill_symmetry(xy_grid, unknown)
    print(f"Filled grid 2: {filled_grid2}")
    assert unknown not in filled_grid2._data


if __name__ == "__main__":
    test_find_and_fill_symmetry()
