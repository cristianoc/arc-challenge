from enum import Enum, auto

from grid_data import GridData
from rule_based_selector import Embedding

# Define the SymmetryType Enum


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


# Unpack SymmetryType members into the local scope
VERTICAL, HORIZONTAL, DIAGONAL_LEFT, DIAGONAL_RIGHT, ROTATIONAL_90, ROTATIONAL_180, ROTATIONAL_270 = SymmetryFeatures

# Function to check for vertical symmetry


def check_vertical_symmetry(grid: GridData) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols // 2):
            if grid[i][j] != grid[i][cols - j - 1]:
                return False
    return True

# Function to check for horizontal symmetry


def check_horizontal_symmetry(grid: GridData) -> bool:
    rows = len(grid)
    for i in range(rows // 2):
        if grid[i] != grid[rows - i - 1]:
            return False
    return True


# Function to check for diagonal left symmetry
def check_diagonal_left_symmetry(grid: GridData) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    if rows != cols:
        return False  # Diagonal symmetry is only valid for square grids
    for i in range(rows):
        for j in range(i + 1):
            if grid[i][j] != grid[j][i]:
                return False
    return True


# Function to check for diagonal right symmetry
def check_diagonal_right_symmetry(grid: GridData) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    if rows != cols:
        return False  # Diagonal symmetry is only valid for square grids
    for i in range(rows):
        for j in range(cols - i - 1, cols):
            if grid[i][j] != grid[cols - j - 1][rows - i - 1]:
                return False
    return True


# Function to check for rotational symmetry at 90 degrees
def check_rotational_90_symmetry(grid: GridData) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    if rows != cols:
        return False  # Rotational symmetry is easiest to check for square grids
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != grid[j][rows - i - 1]:
                return False
    return True


# Function to check for rotational symmetry at 180 degrees
def check_rotational_180_symmetry(grid: GridData) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != grid[rows - i - 1][cols - j - 1]:
                return False
    return True


# Function to check for rotational symmetry at 270 degrees
def check_rotational_270_symmetry(grid: GridData) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    if rows != cols:
        return False  # Rotational symmetry is easiest to check for square grids
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != grid[cols - j - 1][i]:
                return False
    return True


# Function to detect all symmetries in the grid
def detect_symmetry_features(data: GridData) -> Embedding:
    embedding : Embedding = {}
    embedding[VERTICAL.name] = check_vertical_symmetry(data)
    embedding[HORIZONTAL.name] = check_horizontal_symmetry(data)
    embedding[DIAGONAL_LEFT.name] = check_diagonal_left_symmetry(data)
    embedding[DIAGONAL_RIGHT.name] = check_diagonal_right_symmetry(data)
    embedding[ROTATIONAL_90.name] = check_rotational_90_symmetry(data)
    embedding[ROTATIONAL_180.name] = check_rotational_180_symmetry(data)
    embedding[ROTATIONAL_270.name] = check_rotational_270_symmetry(data)
    return embedding


def test_symmetries():
    data1 = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]
    data2 = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1]
    ]
    symmetries1 = detect_symmetry_features(data1)
    symmetries2 = detect_symmetry_features(data2)

    true_keys1 = [key for key, value in symmetries1.items() if value]
    assert true_keys1 == [s.name for s in [VERTICAL, HORIZONTAL, DIAGONAL_LEFT, DIAGONAL_RIGHT, ROTATIONAL_90, ROTATIONAL_180, ROTATIONAL_270]]

    true_keys2 = [key for key, value in symmetries2.items() if value]
    assert true_keys2 == [s.name for s in [VERTICAL]]
