from grid_data import YELLOW, display
from grid import Grid
import load_data

"""
This example demonstrates transforming a grid of colors by identifying and changing enclosed areas. 
Using a grid of integers to represent different colors, the example highlights the transformation of 
enclosed areas to a specific color (YELLOW), while non-enclosed areas retain their original colors.

The `transform` function determines if a grid cell is enclosed by checking its surrounding cells. 
The example grid showcases this logic using the color GREEN to illustrate areas that should change to YELLOW.
"""


def transform(input_grid: Grid) -> Grid:
    # if the square is enclosed, make it yellow, otherwise keep the original color
    def map_func(x: int, y: int) -> int:
        color = input_grid.data[x][y]
        assert isinstance(color, int)
        new_color = YELLOW if input_grid.is_enclosed(x, y) else color
        return new_color
    return input_grid.map(map_func)


def test_example():
    name = "00d62c1b.json"
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
