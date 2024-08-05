from example_tester import example
from grid import Grid
from grid_data import RED


def transform(input_grid: Grid) -> Grid:
    # Assume input grid is 7x3
    # Find leftmost and rightmost 3x3 grid
    leftmost_grid = Grid([row[:3] for row in input_grid.data])
    rightmost_grid = Grid([row[-3:] for row in input_grid.data])
    output_grid = Grid.empty(3, 3)
    # set the color to red if both grids have non-zero color at the same position
    for i in range(3):
        for j in range(3):
            if leftmost_grid.data[i][j] != 0 and rightmost_grid.data[i][j] != 0:
                output_grid.data[i][j] = RED
    return output_grid


def test():
    example(name="0520fde7.json", transform=transform)
