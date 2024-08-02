from example_tester import example
from grid import Grid

"""
This example demonstrates transforming a grid of colors by replacing certain cells with nested grids.
In this context, a nested grid means that each cell in the original grid is transformed into another grid.

The transformation is defined such that any cell with a value of 0 is replaced by an empty grid of the same 
size, while cells with non-zero values are replaced by a copy of the original grid. This approach highlights 
the use of nested mapping to create grids within grids, illustrating complex transformations and data 
structure manipulations.

The example showcases various color combinations to demonstrate how different patterns result in different 
nested grid structures. The results are visualized to provide a clear understanding of the transformations 
applied to the input data.
"""


def transform(input_grid: Grid) -> Grid:
    def map_func(x: int, y: int) -> Grid:
        color = input_grid.data[x][y]
        assert isinstance(color, int)
        return Grid.empty(input_grid.size()) if color == 0 else input_grid.copy()
    return input_grid.map_nested(map_func)


def test():
    example(name="007bbfb7.json", transform=transform)
