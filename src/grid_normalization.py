from objects import Object
from typing import Tuple, Optional
from math import sqrt

from grid_types import RigidTransformation, ClockwiseRotation, XReflection
import numpy as np


def calculate_mass(color: int, background_color: int) -> int:
    return 1 if color != background_color else 0


def calculate_center_of_mass(
    grid: Object, background_color: int
) -> Tuple[float, float]:
    """
    Calculate the center of mass of the grid, excluding cells with the background color.
    """
    width, height = grid.size
    total_weight = 0
    sum_x = 0
    sum_y = 0

    for i in range(width):
        for j in range(height):
            mass = calculate_mass(grid[i, j], background_color)
            total_weight += mass
            sum_x += i * mass
            sum_y += j * mass
    X_cm = sum_x / total_weight
    Y_cm = sum_y / total_weight
    return X_cm, Y_cm


def mass_of_top_left_half(grid: Object, background_color: int) -> int:
    total_mass = 0
    for y in range(grid.height):
        for x in range(grid.width):
            if x + y < grid.width:  # This condition includes the diagonal
                total_mass += calculate_mass(grid[x, y], background_color)
    return total_mass


def mass_of_bottom_right_half(grid: Object, background_color: int) -> int:
    total_mass = 0
    for y in range(grid.height):
        for x in range(grid.width):
            if x + y >= grid.width - 1:  # This condition includes the diagonal
                total_mass += calculate_mass(grid[x, y], background_color)
    return total_mass


def find_normalizing_transformation(
    grid: Object, background_color: int
) -> RigidTransformation:
    """
    Determine the rigid transformation (rotation and reflection) needed to normalize a grid
    by aligning it with a standard orientation based on its center of mass and the mass distribution.

    Args:
        grid (Object): The grid object representing the 2D grid.
        background_color (int): The color considered as the background (ignored in transformations).

    Returns:
        RigidTransformation: The rotation and reflection needed to bring the grid to its standard form.
    """

    width, height = grid.size

    def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points."""
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Calculate the center of mass of the grid
    center_of_mass = calculate_center_of_mass(grid, background_color)

    # Define the four corners of the grid as tuples
    top_left = (0, 0)
    top_right = (width - 1, 0)
    bottom_left = (0, height - 1)
    bottom_right = (width - 1, height - 1)

    # Calculate the distances from the center of mass to each corner
    distances = [
        distance(bottom_left, center_of_mass),  # 0
        distance(top_left, center_of_mass),  # 1
        distance(top_right, center_of_mass),  # 2
        distance(bottom_right, center_of_mass),  # 3
    ]

    # Find the index of the nearest corner based on the smallest distance
    nearest_corner_index = distances.index(min(distances))

    # Rotate the grid to align the nearest corner with the bottom-left
    unrotated_grid = grid.rot90_clockwise(-nearest_corner_index)

    # Calculate the mass differences to determine the need for reflection
    mass_of_top_left = mass_of_top_left_half(unrotated_grid, background_color)
    mass_of_bottom_right = mass_of_bottom_right_half(unrotated_grid, background_color)
    flip_x = mass_of_top_left > mass_of_bottom_right

    # Adjust rotation if flipping is required
    if flip_x:
        # Adjust rotation to account for the flip, mirroring the original rotation
        # 3 - nearest_corner_index reverses the effect of rotation after a flip
        original_rotation = 3 - nearest_corner_index
    else:
        original_rotation = nearest_corner_index

    # Construct the rigid transformation object
    rotation = ClockwiseRotation(original_rotation % 4)
    x_reflection = XReflection(flip_x)
    rigid_transformation = RigidTransformation(rotation, x_reflection)

    return rigid_transformation


def apply_inverse_transformation(
    grid: Object, rigid_transformation: RigidTransformation
) -> Object:
    return grid.apply_rigid_xform(rigid_transformation.inverse())


def test_normalize_grid():
    # Example usage
    initial_grid = Object(
        np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 2, 0, 0, 0],
                [1, 1, 3, 0, 0],
                [1, 1, 1, 4, 1],
                [1, 1, 1, 1, 5],
            ]
        )
    )
    background_color = 0

    # Try all 8 rigidtransformations (rotations + flips)
    i = 0
    for rotation in ClockwiseRotation:
        for x_reflection in XReflection:
            rigid_transformation = RigidTransformation(rotation, x_reflection)
            print(f"\nTransformation {i+1}/{8}: {rigid_transformation}")
            i += 1
            grid = initial_grid.apply_rigid_xform(rigid_transformation)
            print(f"Transformed grid: {grid}")
            normalizing_transformation = find_normalizing_transformation(
                grid, background_color
            )
            print(f"Normalizing transformation: {normalizing_transformation}")
            normalized_grid = apply_inverse_transformation(
                grid, normalizing_transformation
            )
            print(f"Normalized grid: {normalized_grid}")
            assert normalized_grid == initial_grid


def match_grids_and_find_transformation(
    grid1: Object, grid2: Object, background_color: int
) -> Optional[RigidTransformation]:
    """
    Check if two grids match after normalization and return the transformation between them.

    Args:
        grid1 (Object): The first grid to compare.
        grid2 (Object): The second grid to compare.
        background_color (int): The color considered as the background.

    Returns:
        Optional[RigidTransformation]: The transformation from grid1 to grid2 if they match
        after normalization, or None if they do not match.
    """
    nt1 = find_normalizing_transformation(grid1, background_color)
    nt2 = find_normalizing_transformation(grid2, background_color)
    ng1 = grid1.apply_rigid_xform(nt1.inverse())
    ng2 = grid2.apply_rigid_xform(nt2.inverse())
    if ng1 == ng2:
        g1_to_g2 = nt1.inverse().compose_with(nt2)
        return g1_to_g2
    else:
        return None

def test_equal_after_normalization():
    """
    Test that normalizing any two rigid transformations of a grid yields identical results.

    Applies all possible rigid transformations to a test grid, normalizes the results,
    and asserts that all normalized grids are equal, regardless of the initial transformation.
    """
    initial_grid = Object(
        np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 2, 0, 0, 0],
                [1, 1, 3, 0, 0],
                [1, 1, 1, 4, 1],
                [1, 1, 1, 1, 5],
            ]
        )
    )
    background_color = 0
    # Try all 8 rigid transformations (rotations + flips)
    for rotation1 in ClockwiseRotation:
        for x_reflection1 in XReflection:
            for rotation2 in ClockwiseRotation:
                for x_reflection2 in XReflection:
                    t1 = RigidTransformation(rotation1, x_reflection1)
                    t2 = RigidTransformation(rotation2, x_reflection2)
                    g1 = initial_grid.apply_rigid_xform(t1)
                    g2 = initial_grid.apply_rigid_xform(t2)

                    g1_to_g2 = match_grids_and_find_transformation(g1, g2, background_color)
                    assert g1_to_g2 is not None
                    assert g1.apply_rigid_xform(g1_to_g2) == g2
