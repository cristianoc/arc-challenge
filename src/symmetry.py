from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from cardinality_predicates import (
    CardinalityPredicate,
    fill_grid_based_on_predicate,
    find_cardinality_predicates,
)
from grid_types import Symmetry
from logger import logger
from objects import Object, display

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

    def __str__(self):
        symmetries = []
        if self.px:
            symmetries.append(f"px={self.px}")
        if self.py:
            symmetries.append(f"py={self.py}")
        if self.pd:
            symmetries.append(f"pd={self.pd}")
        if self.pa:
            symmetries.append(f"pa={self.pa}")

        return f"PeriodicGridSymmetry({', '.join(symmetries)})"

    def __repr__(self):
        return self.__str__()


@dataclass(frozen=True)
class NonPeriodicGridSymmetry:
    hx: bool = False  # non-periodic horizontal
    vy: bool = False  # non-periodic vertical
    dg: bool = False  # non-periodic diagonal
    ag: bool = False  # non-periodic anti-diagonal
    offset: Tuple[int, int] = (0, 0)  # offset for symmetry checks
    hxm: Optional[Object] = None  # mask for horizontal symmetry
    vym: Optional[Object] = None  # mask for vertical symmetry
    dgm: Optional[Object] = None  # mask for diagonal symmetry
    agm: Optional[Object] = None  # mask for anti-diagonal symmetry

    def __str__(self):
        symmetries = []
        if self.hx:
            symmetries.append("hx")
        if self.vy:
            symmetries.append("vy")
        if self.dg:
            symmetries.append("dg")
        if self.ag:
            symmetries.append("ag")
        if self.hxm is not None:
            symmetries.append(f"hxm={self.hxm.num_cells(None) * 100 // self.hxm.area}%")
        if self.vym is not None:
            symmetries.append(f"vym={self.vym.num_cells(None) * 100 // self.vym.area}%")
        if self.dgm is not None:
            symmetries.append(f"dgm={self.dgm.num_cells(None) * 100 // self.dgm.area}%")
        if self.agm is not None:
            symmetries.append(f"agm={self.agm.num_cells(None) * 100 // self.agm.area}%")
        symmetry_str = ", ".join(symmetries)

        if self.offset != (0, 0):
            return f"NonPeriodicGridSymmetry({symmetry_str}, offset={self.offset})"
        elif symmetries:
            return f"NonPeriodicGridSymmetry({symmetry_str})"
        else:
            return "NonPeriodicGridSymmetry()"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def mask_intersection(g1: Object, g2: Object) -> Object:
        mask = g1.empty(g1.size)
        for x in range(g1.width):
            for y in range(g1.height):
                if g1[x, y] == 1 and g2[x, y] == 1:
                    mask[x, y] = 1
        return mask

    def intersection(
        self, other: "NonPeriodicGridSymmetry"
    ) -> "NonPeriodicGridSymmetry":
        hxm = None
        vym = None
        dgm = None
        agm = None
        if self.hxm is not None and other.hxm is not None:
            hxm = NonPeriodicGridSymmetry.mask_intersection(self.hxm, other.hxm)
        if self.vym is not None and other.vym is not None:
            vym = NonPeriodicGridSymmetry.mask_intersection(self.vym, other.vym)
        if self.dgm is not None and other.dgm is not None:
            dgm = NonPeriodicGridSymmetry.mask_intersection(self.dgm, other.dgm)
        if self.agm is not None and other.agm is not None:
            agm = NonPeriodicGridSymmetry.mask_intersection(self.agm, other.agm)
        # If offsets differ, symmetries involving translations should be invalidated (set to False)
        if self.offset != other.offset:
            return NonPeriodicGridSymmetry(
                hx=False,
                vy=False,
                dg=False,
                ag=False,
                offset=(0, 0),  # Reset the offset since they are different
                hxm=hxm,
                vym=vym,
                dgm=dgm,
                agm=agm,
            )
        else:
            # If offsets are the same, apply logical "and" to the symmetries
            return NonPeriodicGridSymmetry(
                hx=self.hx and other.hx,
                vy=self.vy and other.vy,
                dg=self.dg and other.dg,
                ag=self.ag and other.ag,
                offset=self.offset,  # Offsets match, so we keep the offset
                hxm=hxm,
                vym=vym,
                dgm=dgm,
                agm=agm,
            )


def check_vertical_symmetry_with_unknowns(
    grid: Object, period: int, unknown: int, mask: Optional[Object]
):
    """
    Check if rows repeat every 'period' rows, allowing for unknown cells.
    """
    width, height = grid.size
    for x in range(width):
        for y in range(period, height):
            if (
                (mask is None or mask[x, y] == 0)
                and grid[x, y] != unknown
                and (mask is None or mask[x, y - period] == 0)
                and grid[x, y - period] != unknown
                and grid[x, y] != grid[x, y - period]
            ):
                return False
    return True


def check_horizontal_symmetry_with_unknowns(
    grid: Object, period: int, unknown: int, mask: Optional[Object]
):
    """
    Check if columns repeat every 'period' columns, allowing for unknown cells.
    """
    width, height = grid.size
    for x in range(period, width):
        for y in range(height):
            if (
                (mask is None or mask[x, y] == 0)
                and grid[x, y] != unknown
                and (mask is None or mask[x - period, y] == 0)
                and grid[x - period, y] != unknown
                and grid[x, y] != grid[x - period, y]
            ):
                return False
    return True


def check_diagonal_symmetry_with_unknowns(
    grid: Object, period: int, unknown: int, mask: Optional[Object]
):
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
                (mask is None or mask[x, y] == 0)
                and grid[x, y] != unknown
                and (mask is None or mask[next_x, next_y] == 0)
                and grid[next_x, next_y] != unknown
                and grid[x, y] != grid[next_x, next_y]
            ):
                return False
    return True


def check_anti_diagonal_symmetry_with_unknowns(
    grid: Object, period: int, unknown: int, mask: Optional[Object]
):
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
                (mask is None or mask[x, y] == 0)
                and grid[x, y] != unknown
                and (mask is None or mask[next_x, next_y] == 0)
                and grid[next_x, next_y] != unknown
                and grid[x, y] != grid[next_x, next_y]
            ):
                return False
    return True


def find_periodic_symmetry_predicates(
    grid: Object, unknown: int, mask: Optional[Object]
) -> PeriodicGridSymmetry:
    """
    Find the smallest periods px, py, pd, pa (if any) and non-periodic symmetries with unknowns.
    """
    width, height = grid.size

    # Find smallest horizontal symmetry modulo px
    px = None
    for possible_px in range(1, width // 2 + 1):
        if check_horizontal_symmetry_with_unknowns(grid, possible_px, unknown, mask):
            px = possible_px
            break

    # Find smallest vertical symmetry modulo py
    py = None
    for possible_py in range(1, height // 2 + 1):
        if check_vertical_symmetry_with_unknowns(grid, possible_py, unknown, mask):
            py = possible_py
            break

    # Find smallest diagonal symmetry modulo pd
    pd = None
    # Ensure the grid is square for diagonal symmetry
    if width == height:
        for possible_pd in range(1, width // 2 + 1):
            if check_diagonal_symmetry_with_unknowns(grid, possible_pd, unknown, mask):
                pd = possible_pd
                break

    # Find smallest anti-diagonal symmetry modulo pa
    pa = None
    # Ensure the grid is square for anti-diagonal symmetry
    if width == height:
        for possible_pa in range(1, width // 2 + 1):
            if check_anti_diagonal_symmetry_with_unknowns(
                grid, possible_pa, unknown, mask
            ):
                pa = possible_pa
                break

    return PeriodicGridSymmetry(px, py, pd, pa)


@dataclass(frozen=True)
class SymmetryResult:
    detected: bool
    offset: Tuple[int, int]
    score: float
    mask: Optional["Object"]  # Assuming Object represents a grid/mask


def find_non_periodic_symmetry_predicates(
    grid: Object,
    unknown: int,
    min_mask_weight: float = 0.5,  # Threshold for considering a symmetry significant
) -> NonPeriodicGridSymmetry:
    """
    Find the non-periodic symmetries of the grid, considering offsets.
    Handles both complete and partial symmetries by determining optimal offsets.
    """
    width, height = grid.size
    max_distance = max(width, height) // 2

    # Initialize the result dictionary
    symmetry_results: Dict[Symmetry, SymmetryResult] = {}

    for symmetry in Symmetry:
        best_offset = (0, 0)
        best_score = 0.0
        best_mask = None

        # Iterate over all possible offsets within the max_distance
        for dx in range(-max_distance, max_distance + 1):
            for dy in range(-max_distance, max_distance + 1):
                # Compute the score for the current symmetry with the given offset
                match_count, total_pairs = compute_symmetry_score(
                    grid, symmetry, (dx, dy), unknown
                )
                score = match_count / total_pairs if total_pairs > 0 else 0.0

                # Update the best offset if the current score is higher
                if score > best_score:
                    best_score = score
                    best_offset = (dx, dy)

        # Determine if the best score exceeds the threshold for partial symmetry
        if best_score >= min_mask_weight:
            # Generate the mask based on the best offset
            mask = compute_symmetry_mask(grid, symmetry, best_offset, unknown)
            detected = True
        else:
            mask = None
            detected = False
            best_offset = (0, 0)  # Reset offset if symmetry is not significant

        # Store the result for the current symmetry
        symmetry_results[symmetry] = SymmetryResult(
            detected=detected, offset=best_offset, score=best_score, mask=mask
        )

        # Initialize combined offset
        combined_offset = (0, 0)

        # Flags for symmetries
        rx = (
            symmetry_results[Symmetry.HORIZONTAL]
            if Symmetry.HORIZONTAL in symmetry_results
            else None
        )
        ry = (
            symmetry_results[Symmetry.VERTICAL]
            if Symmetry.VERTICAL in symmetry_results
            else None
        )
        ra = (
            symmetry_results[Symmetry.ANTI_DIAGONAL]
            if Symmetry.ANTI_DIAGONAL in symmetry_results
            else None
        )
        hx_detected = rx.detected if rx is not None else False
        vy_detected = ry.detected if ry is not None else False
        ag_detected = ra.detected if ra is not None else False

        # Retrieve individual offsets
        hx_offset = (
            symmetry_results[Symmetry.HORIZONTAL].offset if hx_detected else (0, 0)
        )
        vy_offset = (
            symmetry_results[Symmetry.VERTICAL].offset if vy_detected else (0, 0)
        )
        ag_offset = (
            symmetry_results[Symmetry.ANTI_DIAGONAL].offset if ag_detected else (0, 0)
        )

        # Combine horizontal and vertical offsets
        if hx_detected:
            combined_offset = (
                hx_offset[0],
                combined_offset[1],
            )  # Only x-offset from horizontal symmetry

        if vy_detected:
            combined_offset = (
                combined_offset[0],
                vy_offset[1],
            )  # Only y-offset from vertical symmetry

    # Handle anti-diagonal symmetry conflicts
    if ag_detected and (hx_detected or vy_detected):
        # For diagonal symmetry in square grids, assume dx == dy
        # Check if anti-diagonal offset matches the combined offset
        if ag_offset != combined_offset:
            # Contradiction detected; invalidate all symmetries
            symmetry_results = {}
            combined_offset = (0, 0)  # Reset combined offset

    offset = combined_offset

    hx = False
    vy = False
    dg = False
    ag = False
    hxm = None
    vym = None
    dgm = None
    agm = None
    for symmetry, result in symmetry_results.items():
        if symmetry == Symmetry.HORIZONTAL:
            hx = True
        elif symmetry == Symmetry.VERTICAL:
            vy = True
        elif symmetry == Symmetry.DIAGONAL:
            dg = True
        elif symmetry == Symmetry.ANTI_DIAGONAL:
            ag = True
        else:
            raise ValueError(f"Unknown symmetry type: {symmetry}")
        if result.score < 1:
            if symmetry == Symmetry.HORIZONTAL:
                hxm = result.mask
            elif symmetry == Symmetry.VERTICAL:
                vym = result.mask
            elif symmetry == Symmetry.DIAGONAL:
                dgm = result.mask
            elif symmetry == Symmetry.ANTI_DIAGONAL:
                agm = result.mask

    return NonPeriodicGridSymmetry(
        hx=hx,
        vy=vy,
        dg=dg,
        ag=ag,
        offset=offset,
        hxm=hxm,
        vym=vym,
        dgm=dgm,
        agm=agm,
    )


def compute_symmetry_score(
    grid: "Object", symmetry: Symmetry, offset: Tuple[int, int], unknown: int
) -> Tuple[int, int]:
    """
    Computes the number of matching cell pairs and the total number of relevant pairs
    for a given symmetry and offset.

    Returns:
        A tuple of (match_count, total_pairs)
    """
    width, height = grid.size
    dx, dy = offset

    # Get the NumPy array data
    grid_data = grid._data

    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")

    # Apply symmetry transformation
    x_trans, y_trans = apply_symmetry_vectorized(
        x_coords, y_coords, symmetry, width, height
    )

    # Apply offsets
    x_trans += dx
    y_trans += dy

    # Check bounds
    valid_mask = (
        (x_trans >= 0) & (x_trans < width) & (y_trans >= 0) & (y_trans < height)
    )

    # Prepare original and transformed cell values
    # Note: grid_data[y, x] corresponds to grid[x, y]
    cell_original = grid_data[x_coords[valid_mask], y_coords[valid_mask]]
    cell_transformed = grid_data[x_trans[valid_mask], y_trans[valid_mask]]

    # Create unknown masks
    unknown_mask = (cell_original == unknown) | (cell_transformed == unknown)

    # Compute match count
    matches = (cell_original == cell_transformed) | unknown_mask
    match_count = np.sum(matches)
    total_pairs = matches.size

    return match_count, total_pairs


def apply_symmetry_vectorized(
    x: np.ndarray, y: np.ndarray, symmetry: Symmetry, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    if symmetry == Symmetry.HORIZONTAL:
        return width - 1 - x, y
    elif symmetry == Symmetry.VERTICAL:
        return x, height - 1 - y
    elif symmetry == Symmetry.DIAGONAL:
        return y, x
    elif symmetry == Symmetry.ANTI_DIAGONAL:
        return height - 1 - y, width - 1 - x
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry}")


def apply_symmetry(
    x: int, y: int, symmetry: Symmetry, width: int, height: int
) -> Tuple[int, int]:
    """
    Applies the specified symmetry transformation to the given coordinates.

    Returns:
        Transformed (x, y) coordinates.
    """
    if symmetry == Symmetry.HORIZONTAL:
        return (width - 1 - x, y)
    elif symmetry == Symmetry.VERTICAL:
        return (x, height - 1 - y)
    elif symmetry == Symmetry.DIAGONAL:
        return (y, x)
    elif symmetry == Symmetry.ANTI_DIAGONAL:
        return (height - 1 - y, width - 1 - x)
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry}")


def compute_symmetry_mask(
    grid: "Object", symmetry: Symmetry, offset: Tuple[int, int], unknown: int
) -> Optional["Object"]:
    """
    Generates a mask indicating where the symmetry holds based on the optimal offset.

    Returns:
        A mask object where matching cells are marked (e.g., with 1) and non-matching cells are unmarked (e.g., with 0).
        Returns None if no mask is applicable.
    """
    width, height = grid.size
    dx, dy = offset
    mask = grid.empty(
        size=grid.size
    )  # Assuming grid.empty() creates an empty mask with the same size

    match_count = 0
    total_pairs = 0

    for x in range(width):
        for y in range(height):
            # Compute the transformed coordinates based on the symmetry
            x_trans, y_trans = apply_symmetry(x, y, symmetry, width, height)

            # Apply the offset
            x_trans += dx
            y_trans += dy

            # Check if the transformed coordinates are within bounds
            if 0 <= x_trans < width and 0 <= y_trans < height:
                cell_original = grid[x, y]
                cell_transformed = grid[x_trans, y_trans]
                if (
                    cell_original == cell_transformed
                    or cell_original == unknown
                    or cell_transformed == unknown
                ):
                    mask[x, y] = 1  # Mark as matching
                    match_count += 1
                else:
                    mask[x, y] = 0  # Mark as non-matching
                total_pairs += 1
            else:
                mask[x, y] = 0  # Out of bounds is non-matching
                total_pairs += 1

    # Optionally, additional processing can be done on the mask
    return (
        mask if match_count / total_pairs >= 0.5 else None
    )  # Reconfirming the threshold


def find_source_value(
    filled_grid: Object,
    x: int,
    y: int,
    periodic_symmetry: PeriodicGridSymmetry,
    non_periodic_symmetry: NonPeriodicGridSymmetry,
    unknown: int,
    mask: Optional[Object],
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
    for x_src in range(x % px, width, px) if px is not None else [x]:
        for y_src in range(y % py, height, py) if py is not None else [y]:
            if filled_grid[x_src, y_src] != unknown and (
                mask is None or mask[x_src, y_src] == 0
            ):
                return filled_grid[x_src, y_src]

    # Search based on diagonal (pd) symmetry if provided
    if pd is not None and width == height:
        size = (width // pd) * pd

        # Walk along the diagonal in both directions, by starting negative and going positive
        for i in range(-size, size, pd):
            x_src = x + i
            y_src = y + i

            if (
                0 <= x_src < size
                and 0 <= y_src < size
                and filled_grid[x_src, y_src] != unknown
                and (mask is None or mask[x_src, y_src] == 0)
            ):
                return filled_grid[x_src, y_src]

    # Search based on anti-diagonal symmetry (bottom-left to top-right)
    if pa is not None and width == height:
        size = (width // pa) * pa

        # Walk along the anti-diagonal in both directions, by starting negative and going positive
        for i in range(-size, size, pa):
            x_src = x + i
            y_src = y - i

            if (
                0 <= x_src < size
                and 0 <= y_src < size
                and filled_grid[x_src, y_src] != unknown
                and (mask is None or mask[x_src, y_src] == 0)
            ):
                return filled_grid[x_src, y_src]

    # Check non-periodic symmetries with offset
    dx, dy = non_periodic_symmetry.offset

    def fill_from_symmetry(x_src, y_src, m):
        if (
            0 <= x_src < width
            and 0 <= y_src < height
            and filled_grid[x_src, y_src] != unknown
            and (m is None or m[x_src, y_src] == 1)
            and (mask is None or mask[x_src, y_src] == 0)
        ):
            return True
        else:
            return False

    hx = non_periodic_symmetry.hx
    vy = non_periodic_symmetry.vy
    dg = non_periodic_symmetry.dg
    ag = non_periodic_symmetry.ag
    hxm = non_periodic_symmetry.hxm
    vym = non_periodic_symmetry.vym
    dgm = non_periodic_symmetry.dgm
    agm = non_periodic_symmetry.agm

    if hx:
        # (x,y) -> (x-dx, y-dy) -> ((w-dx)-1-x+dx, y-dy) ->
        # -> ((w-dx)-1-x+dx+dx, y-dy+dy) = (w-1-x+dx, y)
        x_src, y_src = width - 1 - x + dx, y
        if fill_from_symmetry(x_src, y_src, hxm):
            return filled_grid[x_src, y_src]

    if vy:
        x_src, y_src = x, height - 1 - y + dy
        if fill_from_symmetry(x_src, y_src, vym):
            return filled_grid[x_src, y_src]

    if dg:
        # (x,y) -> (x-dx, y-dy) -> (y-dy, x-dx) -> (y-dy+dx, x-dx+dy)
        x_src, y_src = y - dy + dx, x - dx + dy
        if fill_from_symmetry(x_src, y_src, dgm):
            return filled_grid[x_src, y_src]

    if ag:
        # (x,y) -> (x-dx, y-dy) -> ((h-dy)-1-y+dy, (w-dx)-1-x+dx) ->
        # -> (h-dy-1-y+dy+dx, w-dx-1-x+dx+dy) == (h-1-y+dx, w-1-x+dy)
        x_src = height - 1 - y + dx
        y_src = width - 1 - x + dy
        if fill_from_symmetry(x_src, y_src, agm):
            return filled_grid[x_src, y_src]

    return unknown


def fill_grid(
    grid: Object,
    mask: Optional[Object] = None,
    periodic_symmetry: PeriodicGridSymmetry = PeriodicGridSymmetry(),
    non_periodic_symmetry: NonPeriodicGridSymmetry = NonPeriodicGridSymmetry(),
    cardinality_predicates: List[CardinalityPredicate] = [],
    unknown: int = 0,
):
    """
    Fills the unknown cells in a grid based on detected horizontal and vertical symmetries and cardinality predicates.

    This function fills each unknown cell in the grid by propagating values from known cells,
    using the provided horizontal (px) and vertical (py) symmetry periods. It starts at each
    destination cell and looks for a matching source cell at symmetrical positions, based on
    the periods px and py.

    Args:
        grid (Object): The grid containing known and unknown values to be filled.
        periodic_symmetry (PeriodicGridSymmetry): The symmetry object containing the periods px, py, pd, and pa.
        non_periodic_symmetry (NonPeriodicGridSymmetry): The non-periodic symmetry object.
        cardinality_predicates (List[Union[CardinalityInRowPredicate, CardinalityInColumnPredicate]]): List of cardinality predicates.
        unknown (int): The value representing unknown cells in the grid, which will be filled.

    Returns:
        Object: A new grid with all unknown cells filled using the provided symmetry periods and cardinality predicates.
    """
    width, height = grid.size
    filled_grid = grid.copy()

    changes_made = True
    while changes_made:
        changes_made = False

        # Fill based on symmetry
        for x_dest in range(width):
            for y_dest in range(height):
                if filled_grid[x_dest, y_dest] == unknown and (
                    mask is None or mask[x_dest, y_dest] == 0
                ):  # If the destination cell is unknown
                    color = find_source_value(
                        filled_grid,
                        x_dest,
                        y_dest,
                        periodic_symmetry,
                        non_periodic_symmetry,
                        unknown,
                        mask,
                    )
                    if color == unknown:
                        continue
                    changes_made = True
                    filled_grid[x_dest, y_dest] = color
        # Fill based on cardinality predicates
        for predicate in cardinality_predicates:
            if fill_grid_based_on_predicate(filled_grid, predicate, unknown):
                changes_made = True

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
        mask = grid.copy()
        periodic_symmetry = find_periodic_symmetry_predicates(grid, unknown, mask)
        non_periodic_symmetry = find_non_periodic_symmetry_predicates(grid, unknown)
        cardinality_predicates = find_cardinality_predicates(grid)
        print(
            f"{title}: {periodic_symmetry}, {non_periodic_symmetry}, {cardinality_predicates}"
        )
        mask = grid.copy()
        filled_grid = fill_grid(
            grid,
            None,
            periodic_symmetry,
            non_periodic_symmetry,
            cardinality_predicates,
            unknown,
        )
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


def mask_visible_subgrid_with_unknown(g1: Object, g2: Object) -> Object:
    """
    Returns a mask indicating the positions where g1 and g2 are considered equal,
    modulo unknown values, at the given offset (ox, oy).
    """
    W1, H1 = g1.size  # Width and height of g1
    W2, H2 = g2.size  # Width and height of g2

    mask = Object.empty(g2.size, background_color=1)

    display(g1, g2, title=f"mask_visible_subgrid_with_unknown")

    for x in range(W2):
        for y in range(H2):
            if 0 <= x < W1 and 0 <= y < H1:
                val_g1 = g1[x, y]
                val_g2 = g2[x, y]
                if val_g1 == val_g2:
                    mask[x, y] = 1
                else:
                    print(f"masking {x}, {y}")
                    mask[x, y] = 0

    return mask


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
    g1: Object,
    g2: Object,
    max_distance: int,
    unknown: int,
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
