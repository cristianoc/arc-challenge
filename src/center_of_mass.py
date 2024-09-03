from objects import Object, Rotation, Axis
from typing import Tuple
from math import sqrt

from enum import Enum, auto
from typing import NamedTuple
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


def find_inverse_transformation(
    grid: Object, background_color: int
) -> RigidTransformation:
    """
    Determine the rigid transformation (rotation and reflection) needed to normalize a grid
    by aligning it with a standard orientation based on its center of mass.

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
        if nearest_corner_index in {
            1,
            3,
        }:  # If the nearest corner is top-left or bottom-right
            # Adjust rotation to account for the flip, mirroring the original rotation
            # 3 - nearest_corner_index reverses the effect of rotation after a flip
            original_rotation = 3 - nearest_corner_index
        else:
            # Adjust rotation differently for bottom-left or top-right corners after a flip
            # 1 - nearest_corner_index ensures correct orientation post-flip
            original_rotation = 1 - nearest_corner_index
    else:
        original_rotation = nearest_corner_index  # No flip, keep the rotation based on the nearest corner

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
    grid0 = Object(
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

    assert grid0[0, 0] == 1
    assert grid0[4, 4] == 5
    assert grid0[4, 3] == 1

    background_color = 0
    # List all 8 rigidtransformations (rotations + flips)

    # Enumerate rigid transformations
    i = 0
    for rotation in ClockwiseRotation:
        for x_reflection in XReflection:
            rigid_transformation = RigidTransformation(rotation, x_reflection)
            print(f"\nGrid {i} xform: {rigid_transformation}")
            i += 1
            grid = grid0.apply_rigid_xform(rigid_transformation)
            print(f"Transformed grid: {grid}")
            inverse_transformation = find_inverse_transformation(grid, background_color)
            print(f"Inverse transformation: {inverse_transformation}")
            original_grid = apply_inverse_transformation(grid, inverse_transformation)
            print(f"Original grid: {original_grid}")
            assert original_grid == grid0


if __name__ == "__main__":
    test_normalize_grid()
