# type: ignore

import numpy as np


def calculate_center_of_mass(grid, background_color):
    """
    Calculate the center of mass of the grid, excluding cells with the background color.
    """
    width, height = grid.shape
    total_weight = 0
    sum_x = 0
    sum_y = 0

    for i in range(width):
        for j in range(height):
            if grid[i, j] != background_color:
                total_weight += 1
                sum_x += j
                sum_y += i

    if total_weight == 0:
        return None  # Handle empty grids

    X_cm = sum_x / total_weight
    Y_cm = sum_y / total_weight
    return X_cm, Y_cm


def calculate_shift(grid, background_color):
    """
    Calculate the shift of the grid, excluding cells with the background color.
    """
    center_of_mass = calculate_center_of_mass(grid, background_color)
    width, height = grid.shape
    mid_point = ((width - 1) / 2, (height - 1) / 2)
    shift_x = center_of_mass[0] - mid_point[0]
    shift_y = center_of_mass[1] - mid_point[1]
    return shift_x, shift_y


def find_inverse_transformation(grid, background_color):
    """
    Normalize the grid by applying the appropriate transformations to get it into its standard form.
    """

    width, height = grid.shape

    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    center_of_mass = calculate_center_of_mass(grid, background_color)
    top_left = (0, 0)
    top_right = (width - 1, 0)
    bottom_left = (0, height - 1)
    bottom_right = (width - 1, height - 1)
    distances = [
        distance(bottom_left, center_of_mass),  # 0
        distance(top_left, center_of_mass),  # 1
        distance(top_right, center_of_mass),  # 2
        distance(bottom_right, center_of_mass),  # 3
    ]
    nearest_corner = np.argmin(distances)

    rotation = nearest_corner

    unrotated_grid = np.rot90(grid, rotation)

    mass_of_top_minus_bottom = mass_of_top_half(unrotated_grid) - mass_of_bottom_half(
        unrotated_grid
    )
    flip_x = mass_of_top_minus_bottom > 0

    original_rotation = rotation
    if flip_x:
        rotation = (rotation - 1) % 4
        if rotation in {0, 2}:
            original_rotation = 2 - rotation
    if flip_x and rotation in {0, 2}:
        original_rotation = 2 - rotation
    else:
        original_rotation = rotation

    return flip_x, original_rotation

def apply_inverse_transformation(grid, flip_x, original_rotation):
    if flip_x:
        grid = np.fliplr(grid)
    grid = np.rot90(grid, original_rotation)
    return grid


# Example usage
grid0 = np.array(
    [
        [1, 0, 0, 0, 0],
        [1, 2, 0, 0, 0],
        [1, 1, 3, 0, 0],
        [1, 1, 1, 4, 1],
        [1, 1, 1, 1, 5],
    ]
)

background_color = 0


def mass_of_top_half(grid):
    width, height = grid.shape
    return np.sum(grid[: (width + 1) // 2])  # Include middle row in top half


def mass_of_bottom_half(grid):
    width, height = grid.shape
    return np.sum(grid[(width - 1) // 2 :])  # Include middle row in bottom half


def mass_of_left_half(grid):
    width, height = grid.shape
    return np.sum(grid[:, : (height + 1) // 2])  # Include middle row in left half


def mass_of_right_half(grid):
    width, height = grid.shape
    return np.sum(grid[:, (height - 1) // 2 :])  # Include middle row in right half


def test_normalize_grid():
    # List all 8 transformations (rotations + flips)
    grids = []
    for rotation in [0, 1, 2, 3]:
        rotated_grid = np.rot90(grid0, -rotation)
        normalized_grid = find_inverse_transformation(rotated_grid, background_color)
        name = f"R{rotation}"
        grids.append((name, rotated_grid))
        name += "X"
        grids.append((name, np.fliplr(rotated_grid)))
    for i, (name, grid) in enumerate(grids):
        print(f"\nGrid {i} xform: {name}\n{grid}")
        flip_x, original_rotation = find_inverse_transformation(grid, background_color)

        original_grid = apply_inverse_transformation(grid, flip_x, original_rotation)
        assert np.all(original_grid == grid0)

        print(f"Original Transformation: R{original_rotation}{'X' if flip_x else ''}")
        print(f"Inverse Transformation: {'X' if flip_x else ''}R{(-original_rotation) % 4}")
        print(f"Untransformed:\n{original_grid}")
