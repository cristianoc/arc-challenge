from test_runner import puzzle
from objects import Object

"""
# Problem:
# Given a grid, detect all objects and return a new grid with each object
# compacted to the left by one cell. The width of the grid is reduced by one
# cell for each object.
"""


def transform(input: Object) -> Object:
    objects = input.detect_objects()
    rows = len(input.data)
    cols = len(input.data[0])
    new_grid = Object.empty(rows, cols)
    for obj in objects:
        new_grid.add_object(obj.compact_left().move(0, 1))
    return new_grid


def test():
    puzzle(name="025d127b.json", transform=transform)
