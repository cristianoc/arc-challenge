from detect_objects import detect_objects
from example_tester import example
from grid import Grid
from grid_data import display

"""
TODO: describe this example
"""

def transform(input_grid: Grid) -> Grid:
    objects = detect_objects(input_grid)
    for obj in objects:
        print(f"Object: {obj}")
        rows = len(input_grid.data)
        cols = len(input_grid.data[0])
        new_grid: Grid = Grid.empty(rows, cols)
        new_grid.add_object(obj)
        display(title=f"Object {obj}:", input=new_grid.data)

    return input_grid


def test():
    example(name="025d127b.json", transform=transform)
