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


def normalize_grid(grid, background_color):
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
    # print(f"Rotation: {rotation}")
    return rotation


# Example usage
grid = np.array(
    [
        [1, 0, 0, 0, 0],
        [1, 2, 0, 0, 0],
        [1, 1, 3, 0, 0],
        [1, 1, 1, 4, 1],
        [1, 1, 1, 1, 5],
    ]
)

background_color = 0

# List all 16 transformations (rotations + flips)
grids = []
for rotation in [0, 1, 2, 3]:
    rotated_grid = np.rot90(grid, -rotation)
    normalized_grid = normalize_grid(rotated_grid, background_color)
    name = f"R{rotation}"
    grids.append((name, rotated_grid))
    name += "Y"
    grids.append((name, np.flipud(rotated_grid)))
    # name += "X"
    # grids.append((name, np.fliplr(rotated_grid)))
for i, (name, grid) in enumerate(grids):
    print(f"\nGrid {i} xform: {name}\n{grid}")
    rotation = normalize_grid(grid, background_color)
    print(
        f"Rotation:{rotation} Center of mass: {calculate_center_of_mass(grid, background_color)}"
    )
    unrotated_grid = np.rot90(grid, rotation)
    print(f"Unrotated grid {i}:\n{unrotated_grid}")
    shift = calculate_shift(grid, background_color)
    if rotation == 0:
        is_flipped_y = shift[1] < 0
    elif rotation == 1:
        is_flipped_y = shift[0] < 0
    else:
        assert False, "Invalid rotation"
    # if rotation == 1 or rotation == 3:
    #     is_flipped_y = not is_flipped_y
    # if rotation == 0 or rotation == 2:
    #     is_flipped_x = not is_flipped_x
    print(f"Is flipped Y: {is_flipped_y}")
    if is_flipped_y:
        print("Flipping Y")
        unrotated_grid = np.flipud(unrotated_grid)
        # unrotated_grid = np.rot90(unrotated_grid, rotation)
    # if is_flipped_x:
    #     print("Flipping X")
    #     unrotated_grid = np.fliplr(unrotated_grid)
    print(f"Final grid {i}:\n{unrotated_grid}")
