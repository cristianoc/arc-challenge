from test_runner import puzzle
from grid import Grid

"""
"""


def transform(input: Grid) -> Grid:
    # Detect objects in the grid, assuming each object is a diagonal
    objects = input.detect_objects()
    
    # Ensure there are exactly 3 objects as stated in the problem
    assert len(objects) == 3

    # Determine colors of the three detected objects
    object_colors = [obj.color for obj in objects]

    # Create an empty grid with the same dimensions as input
    output_grid = Grid.empty(input.height, input.width)

    # Iterate over diagonals from bottom-left to top-right
    for diagonal_index in range(input.height + input.width - 1):
        # Determine the color index for the current diagonal
        color_index = diagonal_index % 3
        color = object_colors[color_index]

        # Calculate start positions for the diagonal
        if diagonal_index < input.height:
            x_start = diagonal_index
            y_start = 0
        else:
            x_start = input.height - 1
            y_start = diagonal_index - input.height + 1

        # Fill the diagonal with the selected color
        while x_start >= 0 and y_start < input.width:
            output_grid.data[x_start][y_start] = color
            x_start -= 1
            y_start += 1

    return output_grid

def test():
    puzzle(name="05269061.json", transform=transform)
