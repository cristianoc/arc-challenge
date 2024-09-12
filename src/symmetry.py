from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from objects import Object
from grid_types import Symmetry
from typing import TYPE_CHECKING
from math import gcd, lcm

# To avoid circular imports
if TYPE_CHECKING:
    from objects import Object as Object_t
else:
    Object_t = None


@dataclass(frozen=True)
class PeriodicGridSymmetry:
    px: Optional[int] = None  # periodic horizontal
    py: Optional[int] = None  # periodic vertical
    pd: Optional[int] = None  # periodic diagonal
    pa: Optional[int] = None  # periodic anti-diagonal

    def intersection(self, other: "PeriodicGridSymmetry") -> "PeriodicGridSymmetry":
        def intersect(a: Optional[int], b: Optional[int]) -> Optional[int]:
            if a is None or b is None:
                return None
            if a == b:
                return a
            else:
                return lcm(a, b)

        return PeriodicGridSymmetry(
            intersect(self.px, other.px),
            intersect(self.py, other.py),
            intersect(self.pd, other.pd),
            intersect(self.pa, other.pa),
        )


@dataclass(frozen=True)
class NonPeriodicGridSymmetry:
    hx: bool = False  # non-periodic horizontal
    vy: bool = False  # non-periodic vertical
    dg: bool = False  # non-periodic diagonal
    ag: bool = False  # non-periodic anti-diagonal
    offset: Tuple[int, int] = (0, 0)  # offset for symmetry checks

    def intersection(
        self, other: "NonPeriodicGridSymmetry"
    ) -> "NonPeriodicGridSymmetry":
        # If offsets differ, symmetries involving translations should be invalidated (set to False)
        if self.offset != other.offset:
            return NonPeriodicGridSymmetry(
                hx=False,
                vy=False,
                dg=False,
                ag=False,
                offset=(0, 0),  # Reset the offset since they are different
            )
        else:
            # If offsets are the same, apply logical "and" to the symmetries
            return NonPeriodicGridSymmetry(
                hx=self.hx and other.hx,
                vy=self.vy and other.vy,
                dg=self.dg and other.dg,
                ag=self.ag and other.ag,
                offset=self.offset,  # Offsets match, so we keep the offset
            )


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


def check_anti_diagonal_symmetry_with_unknowns(grid: Object, period: int, unknown: int):
    """
    Check if the grid has anti-diagonal symmetry with a given period, allowing for unknown cells.
    Moving anti-diagonally (bottom-left to top-right), we check that the same element is found every 'period' steps.
    """
    width, height = grid.size

    # Only iterate over the range where anti-diagonal steps are valid
    for x in range(width):
        for y in range(height):
            next_x = x + period
            next_y = y - period
            if next_x >= width or next_y < 0:
                continue

            if (
                grid[x, y] != unknown
                and grid[next_x, next_y] != unknown
                and grid[x, y] != grid[next_x, next_y]
            ):
                return False
    return True


def find_periodic_symmetry_with_unknowns(
    grid: Object, unknown: int
) -> PeriodicGridSymmetry:
    """
    Find the smallest periods px, py, pd, pa (if any) and non-periodic symmetries with unknowns.
    """
    width, height = grid.size

    # Find smallest horizontal symmetry modulo px
    px = None
    for possible_px in range(1, width // 2 + 1):
        if check_horizontal_symmetry_with_unknowns(grid, possible_px, unknown):
            px = possible_px
            break

    # Find smallest vertical symmetry modulo py
    py = None
    for possible_py in range(1, height // 2 + 1):
        if check_vertical_symmetry_with_unknowns(grid, possible_py, unknown):
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

    # Find smallest anti-diagonal symmetry modulo pa
    pa = None
    # Ensure the grid is square for anti-diagonal symmetry
    if width == height:
        for possible_pa in range(1, width // 2 + 1):
            if check_anti_diagonal_symmetry_with_unknowns(grid, possible_pa, unknown):
                pa = possible_pa
                break

    return PeriodicGridSymmetry(px, py, pd, pa)


def find_non_periodic_symmetry(grid: Object, unknown: int) -> NonPeriodicGridSymmetry:
    """
    Find the non-periodic symmetries of the grid, considering offsets.
    """
    width, height = grid.size
    max_distance = max(width, height) // 2

    def check_symmetry_with_offset(symmetry_func):
        offset = find_matching_subgrid_offset(
            grid, symmetry_func(grid), max_distance, unknown
        )
        return offset is not None, offset if offset else (0, 0)

    hx, hx_offset = check_symmetry_with_offset(lambda g: g.flip(Symmetry.HORIZONTAL))
    vy, vy_offset = check_symmetry_with_offset(lambda g: g.flip(Symmetry.VERTICAL))
    dg, dg_offset = check_symmetry_with_offset(lambda g: g.flip(Symmetry.DIAGONAL))
    ag, ag_offset = check_symmetry_with_offset(lambda g: g.flip(Symmetry.ANTI_DIAGONAL))

    # combine the offsets
    offset = (0, 0)
    if hx:
        # for horizontal symmetry, only the x-offset is relevant
        offset = (hx_offset[0], offset[1])
    if vy:
        # for vertical symmetry, only the y-offset is relevant
        offset = (offset[0], vy_offset[1])
    # diagonal symmetry is invariant wrt translations, so the offset is always (0, 0)
    if ag and (hx or vy):
        if ag_offset != offset:
            # anti-diagonal symmetry is not invariant wrt translations
            # so we set all symmetries to False as this is a contradiction
            hx = False
            vy = False
            dg = False
            offset = (0, 0)

    return NonPeriodicGridSymmetry(hx, vy, dg, ag, offset)


def find_source_value(
    filled_grid: Object,
    x_dest: int,
    y_dest: int,
    periodic_symmetry: PeriodicGridSymmetry,
    non_periodic_symmetry: NonPeriodicGridSymmetry,
    unknown: int,
):
    """
    Find a source value for the given destination coordinates based on symmetry.
    """
    px, py, pd, pa = (
        periodic_symmetry.px,
        periodic_symmetry.py,
        periodic_symmetry.pd,
        periodic_symmetry.pa,
    )
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

    # Search based on anti-diagonal symmetry (bottom-left to top-right)
    if pa is not None and width == height:
        size = (width // pa) * pa

        # Walk along the anti-diagonal in both directions, by starting negative and going positive
        for i in range(-size, size, pa):
            x_src = x_dest + i
            y_src = y_dest - i

            if (
                0 <= x_src < size
                and 0 <= y_src < size
                and filled_grid[x_src, y_src] != unknown
            ):
                return filled_grid[x_src, y_src]

    # Check non-periodic symmetries with offset
    offset_x, offset_y = non_periodic_symmetry.offset
    if non_periodic_symmetry.hx:
        x_src = width - 1 - (x_dest - offset_x)
        if 0 <= x_src < width and filled_grid[x_src, y_dest] != unknown:
            return filled_grid[x_src, y_dest]

    if non_periodic_symmetry.vy:
        y_src = height - 1 - (y_dest - offset_y)
        if 0 <= y_src < height and filled_grid[x_dest, y_src] != unknown:
            return filled_grid[x_dest, y_src]

    if non_periodic_symmetry.dg and width == height:
        x_src, y_src = y_dest - offset_x, x_dest - offset_y
        if (
            0 <= x_src < width
            and 0 <= y_src < height
            and filled_grid[x_src, y_src] != unknown
        ):
            return filled_grid[x_src, y_src]

    if non_periodic_symmetry.ag and width == height:
        x_src, y_src = width - 1 - (y_dest - offset_x), height - 1 - (x_dest - offset_y)
        if (
            0 <= x_src < width
            and 0 <= y_src < height
            and filled_grid[x_src, y_src] != unknown
        ):
            return filled_grid[x_src, y_src]

    return unknown


def fill_grid(
    grid: Object,
    periodic_symmetry: PeriodicGridSymmetry = PeriodicGridSymmetry(),
    non_periodic_symmetry: NonPeriodicGridSymmetry = NonPeriodicGridSymmetry(),
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
        symmetry (Symmetry): The symmetry object containing the periods px, py, pd, and pa.
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
                    filled_grid,
                    x_dest,
                    y_dest,
                    periodic_symmetry,
                    non_periodic_symmetry,
                    unknown,
                )

    return filled_grid


def test_find_and_fill_symmetry():
    from objects import Object

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
        periodic_symmetry = find_periodic_symmetry_with_unknowns(grid, unknown)
        non_periodic_symmetry = find_non_periodic_symmetry(grid, unknown)
        print(f"{title}: {periodic_symmetry}, {non_periodic_symmetry}")
        filled_grid = fill_grid(grid, periodic_symmetry, non_periodic_symmetry, unknown)
        if False:
            print(f"grid: {grid}")
            print(f"filled_grid: {filled_grid}")
        assert unknown not in filled_grid._data
        return filled_grid

    test_grid(grid_xy, 0, "grid_xy")  # horizontal and vertical symmetry
    test_grid(grid_y, 0, "grid_y")  # vertical symmetry
    test_grid(grid_diagonal, 0, "grid_diagonal")  # diagonal symmetry

    # Add a new test case for offset symmetry
    grid_offset = Object(
        np.array(
            [
                [9, 1, 2, 3, 0, 1],
                [9, 0, 0, 6, 5, 4],
                [9, 7, 8, 9, 8, 7],
                [9, 1, 2, 3, 2, 1],
                [9, 4, 5, 6, 5, 4],
                [9, 1, 2, 3, 2, 1],
            ]
        )
    )
    test_grid(grid_offset, 0, "grid_offset")  # horizontal symmetry with offset


# Function to check if the visible parts of grid g2 match grid g1 at offset (ox, oy)
# with an "unknown" value that is equal to any other value in comparisons
def check_visible_subgrid_with_unknown(
    g1: Object, g2: Object, ox: int, oy: int, unknown: int
) -> bool:
    W1, H1 = g1.size  # Width and height of g1
    W2, H2 = g2.size  # Width and height of g2

    # Iterate over g2's width (W2) and height (H2)
    for x in range(W2):
        for y in range(H2):
            gx, gy = x + ox, y + oy
            # Only check for visible parts (i.e., within the bounds of g1)
            if 0 <= gx < W1 and 0 <= gy < H1:
                val_g1 = g1[gx, gy]
                val_g2 = g2[x, y]
                # Treat 'unknown' as matching any value
                if val_g1 != val_g2 and val_g1 != unknown and val_g2 != unknown:
                    return False
    return True


# Iterator that yields all offsets with increasing Manhattan distance
def manhattan_offset_iterator():
    d = 0
    while True:
        for i in range(-d, d + 1):
            j = d - abs(i)
            yield (i, j)
            if j != 0:  # Avoid adding the same point twice (i, 0) and (i, -0)
                yield (i, -j)
        d += 1


# Brute-force search to find the matching subgrid offset with expanding Manhattan distances
def find_matching_subgrid_offset(
    g1: Object, g2: Object, max_distance: int, unknown: int
) -> Optional[Tuple[int, int]]:
    """
    Returns a tuple (ox, oy) or None, where:

    - (ox, oy) is the offset that satisfies the following properties:
        1. The Manhattan distance |ox| + |oy| is minimized and is <= max_distance.
        2. For all coordinates (x, y) in g2, if (x + ox, y + oy) is within the bounds of g1,
           g1 and g2 are considered equal at those coordinates, *modulo unknown values*.

    - Definition of equality modulo unknown:
        g1[a, b] ~= g2[c, d] if:
            - g1[a, b] == g2[c, d], or
            - g1[a, b] == unknown, or
            - g2[c, d] == unknown.

        In other words, the unknown value is treated as matching any other value.

    - None is returned if no such offset exists within the given max_distance.
    """

    offset_iter = manhattan_offset_iterator()

    # Iterate through the offsets generated by the iterator
    for _ in range(
        (2 * max_distance + 1) ** 2
    ):  # Check all offsets within the LxL limit
        ox, oy = next(offset_iter)
        # If the Manhattan distance exceeds the limit, stop the search
        if abs(ox) + abs(oy) > max_distance:
            break
        if check_visible_subgrid_with_unknown(g1, g2, ox, oy, unknown):
            return (ox, oy)  # Return the offset if a match is found

    return None  # Return None if no valid offset is found


def test_find_matching_subgrid_offset():
    # Example usage with unknown value
    g1 = Object(
        np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]
        )
    )

    # Modify g2 so the center has some values from g1 with 0 treated as unknown
    g2 = Object(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4],
                [0, 6, 7, 8, 9],
                [0, 11, 12, 13, 14],
            ]
        )
    )

    unknown_value = 0  # Treat 0 as "unknown"
    result = find_matching_subgrid_offset(g1, g2, max_distance=3, unknown=unknown_value)
    assert result == (-1, -2), f"Expected (-1, -2), but got {result}"


if __name__ == "__main__":
    test_find_and_fill_symmetry()
    test_find_matching_subgrid_offset()
