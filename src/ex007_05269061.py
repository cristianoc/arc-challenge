from test_runner import puzzle
from grid_data import Object

"""
This transformation function takes a grid as input, detects exactly three objects within it, 
and colors the grid such that each cell's color is determined by its diagonal index. 
The grid is divided into three diagonal groups, where the sum of the row and column indices 
modulo three determines the group. Each group is assigned a color based on the detected objects, 
ensuring that all cells within the same diagonal group have the same color as the corresponding object. 
The output grid retains the dimensions of the input grid, with colors consistently applied across 
diagonal groups.
"""


def transform(input: Object) -> Object:
    # Detect objects in the grid, assuming each object is a diagonal
    objects = input.detect_objects()

    # Ensure there are exactly 3 objects as stated in the problem
    assert len(objects) == 3

    # Diagonal index between 0 and 2
    def diagonal_index(row: int, col: int) -> int:
        return (row + col) % 3

    object_colors = [0 for _ in range(3)]
    for obj in objects:
        point_on_diagonal = (obj.origin[0] + obj.height - 1, obj.origin[1])
        index = diagonal_index(point_on_diagonal[0], point_on_diagonal[1])
        object_colors[index] = obj.first_color

    # Create an empty grid with the same dimensions as input
    output_grid = Object.empty(input.height, input.width)

    for row in range(input.height):
        for col in range(input.width):
            # Calculate the diagonal index of the current cell
            diag = diagonal_index(row, col)
            # Set the color of the cell to the corresponding object color
            color = object_colors[diag]
            output_grid.data[row][col] = color

    return output_grid


def test():
    puzzle(name="05269061.json", transform=transform)
