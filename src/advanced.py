from grid_data import FUSCHIA, ORANGE, RED, YELLOW, display
from grid import Grid

def transform(input_grid: Grid) -> Grid:
    def map_func(x: int, y: int) -> Grid:
        color = input_grid.data[x][y]
        assert isinstance(color, int)
        return Grid.empty(input_grid.size()) if color == 0 else input_grid.copy()
    return input_grid.map_nested(map_func)

def test_example():
    # Define one-letter color variables for clarity
    o = ORANGE
    y = YELLOW
    r = RED
    f = FUSCHIA

    # List of example grids and titles
    examples = [
        ([
            [0, o, o],
            [o, o, o],
            [0, o, o]
        ], "Example 1:"),
        ([
            [y, 0, y],
            [0, 0, 0],
            [0, y, 0]
        ], "Example 2:"),
        ([
            [0, 0, 0],
            [0, 0, r],
            [r, 0, r]
        ], "Example 3:"),
        ([
            [f, f, 0],
            [f, 0, 0],
            [0, f, f]
        ], "Example 4:"),
        ([
            [r, r, r],
            [0, 0, 0],
            [0, r, r]
        ], "Example 5:")
    ]

    # Run each example
    for grid_data, title in examples:
        input_grid = Grid(grid_data)
        output_grid = transform(input_grid)
        display(title=title, input=input_grid.data, output=output_grid.data)

    # Additional test case
    test_grid = Grid([
        [0, o, o],
        [o, o, o],
        [0, o, o],
    ])
    test_output = transform(test_grid)
    display(title="Test:", input=test_grid.data, output=test_output.data)

