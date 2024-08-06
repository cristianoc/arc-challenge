from detect_objects import detect_objects
from test_runner import puzzle
from grid import Grid

"""
# Problem:
# Given a grid, detect all objects and return a new grid with each object
# compacted to the left by one cell. The width of the grid is reduced by one
# cell for each object.
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
    puzzle(name="025d127b.json", transform=transform)
