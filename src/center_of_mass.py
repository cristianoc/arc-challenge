from grid_data import Object, Rotation, Axis
from typing import Tuple, List
from math import sqrt


def calculate_center_of_mass(
    grid: Object, background_color: int
) -> Tuple[float, float]:
    """
    Calculate the center of mass of the grid, excluding cells with the background color.
    """
    height, width = grid.size
    total_weight = 0
    sum_x = 0
    sum_y = 0

    for i in range(width):
        for j in range(height):
            if grid[i, j] != background_color:
                total_weight += 1
                sum_x += i
                sum_y += j
    X_cm = sum_x / total_weight
    Y_cm = sum_y / total_weight
    return X_cm, Y_cm


def rot90(grid: Object, k: int) -> Object:
    k = k % 4
    if k == 0:
        return grid
    elif k == 1:
        return grid.rotate(Rotation.COUNTERCLOCKWISE)
    elif k == 2:
        return grid.rotate(Rotation.COUNTERCLOCKWISE).rotate(Rotation.COUNTERCLOCKWISE)
    else:  # k == 3:
        return grid.rotate(Rotation.CLOCKWISE)


def find_inverse_transformation(grid: Object, background_color: int):
    """
    Normalize the grid by applying the appropriate transformations to get it into its standard form.
    """

    height, width = grid.size

    def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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
    nearest_corner: int = distances.index(min(distances))

    rotation = nearest_corner

    unrotated_grid = rot90(grid, rotation)

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


def apply_inverse_transformation(
    grid: Object, flip_x: bool, original_rotation: int
) -> Object:
    if flip_x:
        grid = grid.flip(Axis.HORIZONTAL)
    grid = rot90(grid, original_rotation)
    return grid


# Example usage
grid0 = Object(
    [
        [1, 0, 0, 0, 0],
        [1, 2, 0, 0, 0],
        [1, 1, 3, 0, 0],
        [1, 1, 1, 4, 1],
        [1, 1, 1, 1, 5],
    ]
)

assert grid0[0, 0] == 1
assert grid0[4, 4] == 5
assert grid0[4, 3] == 1

background_color = 0


def mass_of_top_half(grid: Object) -> int:
    return sum(
        sum(row) for row in grid.data[: (grid.height + 1) // 2]
    )  # Include middle row in top half for odd heights

def mass_of_bottom_half(grid: Object) -> int:
    return sum(
        sum(row) for row in grid.data[grid.height // 2 :]
    )  # Include middle row in bottom half for odd heights

def test_normalize_grid():
    # List all 8 transformations (rotations + flips)
    grids: List[Tuple[str, Object]] = []
    for rotation in [0, 1, 2, 3]:
        rotated_grid = rot90(grid0, -rotation)
        name = f"R{rotation}"
        grids.append((name, rotated_grid))
        name += "X"
        grids.append((name, rotated_grid.flip(Axis.HORIZONTAL)))
    for i, (name, grid) in enumerate(grids):
        print(f"\nGrid {i} xform: {name}{grid}")
        flip_x, original_rotation = find_inverse_transformation(grid, background_color)

        original_grid = apply_inverse_transformation(grid, flip_x, original_rotation)

        print(f"Original Transformation: R{original_rotation}{'X' if flip_x else ''}")
        print(
            f"Inverse Transformation: {'X' if flip_x else ''}R{(-original_rotation) % 4}"
        )
        print(f"Untransformed:{original_grid}")
        assert original_grid.data == grid0.data


if __name__ == "__main__":
    test_normalize_grid()
