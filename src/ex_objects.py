from detect_objects import detect_objects
from example_tester import example
from grid import Grid

"""
TODO: describe this example
"""


def transform(input_grid: Grid) -> Grid:
    objects = detect_objects(input_grid)
    rows = len(input_grid.data)
    cols = len(input_grid.data[0])
    new_grid: Grid = Grid.empty(rows, cols)
    for obj in objects:
        new_grid.add_object(obj.compact_left().move(0, 1))
    return new_grid


def test():
    example(name="025d127b.json", transform=transform)
