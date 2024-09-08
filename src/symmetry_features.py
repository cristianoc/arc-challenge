from enum import Enum, auto

from objects import Object
from rule_based_selector import Features
import numpy as np

# Define the SymmetryFeatures Enum
class SymmetryFeatures(Enum):
    VERTICAL = auto()  # Reflective symmetry along the vertical axis
    HORIZONTAL = auto()  # Reflective symmetry along the horizontal axis
    # Reflective symmetry along the diagonal from top-left to bottom-right
    DIAGONAL_LEFT = auto()
    # Reflective symmetry along the diagonal from top-right to bottom-left
    DIAGONAL_RIGHT = auto()
    ROTATIONAL_90 = auto()  # Rotational symmetry at 90 degrees
    ROTATIONAL_180 = auto()  # Rotational symmetry at 180 degrees
    ROTATIONAL_270 = auto()  # Rotational symmetry at 270 degrees


# Unpack SymmetryFeatures members into the local scope
(
    VERTICAL,
    HORIZONTAL,
    DIAGONAL_LEFT,
    DIAGONAL_RIGHT,
    ROTATIONAL_90,
    ROTATIONAL_180,
    ROTATIONAL_270,
) = SymmetryFeatures


# Function to check for vertical symmetry
def check_vertical_symmetry(obj: Object) -> bool:
    width, height = obj.size  # Correcting the variable name
    for x in range(width // 2):  # Iterating only half the width
        for y in range(height):  # Iterating the full height
            if obj[x, y] != obj[width - x - 1, y]:  # Symmetry condition
                return False
    return True


# Function to check for horizontal symmetry
def check_horizontal_symmetry(obj: Object) -> bool:
    width, height = obj.size  # Correcting the variable name
    for x in range(width):  # Iterating the full width
        for y in range(height // 2):  # Iterating only half the height
            if obj[x, y] != obj[x, height - y - 1]:  # Symmetry condition
                return False
    return True


# Function to check for diagonal left symmetry
def check_diagonal_left_symmetry(grid: Object) -> bool:
    height, width = grid.size
    if height != width:
        return False  # Diagonal symmetry is only valid for square grids
    for y in range(height):
        for x in range(y + 1):
            if grid[x, y] != grid[y, x]:
                return False
    return True


# Function to check for diagonal right symmetry
def check_diagonal_right_symmetry(grid: Object) -> bool:
    height, width = grid.size
    if height != width:
        return False  # Diagonal symmetry is only valid for square grids
    for y in range(height):
        for x in range(width - y - 1, width):
            if grid[x, y] != grid[y, x]:
                return False
    return True


# Function to check for rotational symmetry at 90 degrees
def check_rotational_90_symmetry(obj: Object) -> bool:
    height, width = obj.size
    if height != width:
        return False  # Rotational symmetry is easiest to check for square grids
    for y in range(height):
        for x in range(width):
            if obj[x, y] != obj[height - y - 1, x]:
                return False
    return True


# Function to check for rotational symmetry at 180 degrees
def check_rotational_180_symmetry(grid: Object) -> bool:
    width, height = grid.size
    for y in range(height):
        for x in range(width):
            if grid[x, y] != grid[width - x - 1, height - y - 1]:
                return False
    return True


# Function to check for rotational symmetry at 270 degrees
def check_rotational_270_symmetry(grid: Object) -> bool:
    height, width = grid.size
    if height != width:
        return False  # Rotational symmetry is easiest to check for square grids
    for y in range(height):
        for x in range(width):
            if grid[y, x] != grid[width - x - 1, height - y - 1]:
                return False
    return True


# Function to detect all symmetries in the grid
def detect_symmetry_features(obj: Object) -> Features:
    features: Features = {}
    features[VERTICAL.name] = check_vertical_symmetry(obj)
    features[HORIZONTAL.name] = check_horizontal_symmetry(obj)
    features[DIAGONAL_LEFT.name] = check_diagonal_left_symmetry(obj)
    features[DIAGONAL_RIGHT.name] = check_diagonal_right_symmetry(obj)
    features[ROTATIONAL_90.name] = check_rotational_90_symmetry(obj)
    features[ROTATIONAL_180.name] = check_rotational_180_symmetry(obj)
    features[ROTATIONAL_270.name] = check_rotational_270_symmetry(obj)
    return features


def test_symmetries():
    data1 = Object(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
    data2 = Object(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]]))
    symmetries1 = detect_symmetry_features(data1)
    symmetries2 = detect_symmetry_features(data2)

    true_keys1 = [key for key, value in symmetries1.items() if value]
    assert true_keys1 == [
        s.name
        for s in [
            VERTICAL,
            HORIZONTAL,
            DIAGONAL_LEFT,
            DIAGONAL_RIGHT,
            ROTATIONAL_90,
            ROTATIONAL_180,
            ROTATIONAL_270,
        ]
    ]

    true_keys2 = [key for key, value in symmetries2.items() if value]
    assert true_keys2 == [s.name for s in [VERTICAL]]
