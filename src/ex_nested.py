from grid_data import display
from grid import Grid
import load_data

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


def test_example():
    name = "007bbfb7.json"
    task = load_data.training_data[name]
    train_set = task['train']
    test_set = task['test']
    for i, example in enumerate(train_set):
        input_grid = Grid(example['input'])
        output_grid = transform(input_grid)
        display(title=f"Train Example {i+1}:", input=input_grid.data, output=output_grid.data)
    for i, example in enumerate(test_set):
        input_grid = Grid(example['input'])
        output_grid = transform(input_grid)
        display(title=f"Test Example {i+1}:", input=input_grid.data, output=output_grid.data)
