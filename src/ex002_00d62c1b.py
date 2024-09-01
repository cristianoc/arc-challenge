from test_runner import puzzle
from grid_types import YELLOW
from grid import Grid

"""
This example demonstrates transforming a grid by changing the color of enclosed cells. 
A cell is considered enclosed if it is surrounded by cells of non-zero value. 
The transformation changes the color of enclosed cells to yellow while leaving others unchanged. 
This process highlights the use of conditional logic to manipulate grid data based on spatial relationships.
"""


def transform(input: Grid) -> Grid:
    # if the square is enclosed, make it yellow, otherwise keep the original color
    def map_func(x: int, y: int) -> int:
        color = input.data[x][y]
        new_color = YELLOW if input.is_enclosed(x, y) else color
        return new_color

    return input.map(map_func)


def test():
    puzzle(name="00d62c1b.json", transform=transform)
