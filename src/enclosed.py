from grid_data import GREEN, YELLOW, display
from grid import Grid


def transform(input_grid: Grid) -> Grid:
    # if the square is enclosed, make it yellow, otherwise keep the original color
    def map_func(x: int, y: int) -> int:
        color = input_grid.data[x][y]
        assert isinstance(color, int)
        new_color = YELLOW if input_grid.is_enclosed(x, y) else color
        return new_color
    return input_grid.map(map_func)


def test_example():
    # Define one-letter color variables for clarity
    g = GREEN

    # List of example grids and titles
    examples = [
        (
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, g, 0, 0, 0],
                [0, g, 0, g, 0, 0],
                [0, 0, g, 0, g, 0],
                [0, 0, 0, g, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ], "Example 1:"),
    ]

    # Run each example
    for grid_data, title in examples:
        input_grid = Grid(grid_data)
        output_grid = transform(input_grid)
        display(title=title, input=input_grid.data, output=output_grid.data)
