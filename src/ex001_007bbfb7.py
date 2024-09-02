from test_runner import puzzle
from objects import Object

"""
This example demonstrates transforming a grid by replacing specific cells with nested grids.
Each cell in the original grid is transformed based on its value: cells with a value of 0 are 
replaced by an empty grid of the same size, while non-zero cells are replaced by a copy of the 
original grid. This process showcases the use of nested mapping to create grids within grids, 
illustrating complex data transformations. Various color combinations are used to highlight 
different nested grid patterns, with results visualized for clarity.
"""


def transform(input: Object) -> Object:

    def map_func(x: int, y: int) -> Object:
        color = input.data[x][y]
        assert isinstance(color, int)
        return Object.empty(input.height, input.width) if color == 0 else input.copy()

    return input.map_nested(map_func)


def test():
    puzzle(name="007bbfb7.json", transform=transform)
