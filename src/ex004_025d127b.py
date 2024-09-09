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
    new_grid = Object.empty(input.size)
    for obj in objects:
        new_grid.add_object_in_place(obj.compact_left().move(1, 0))
    return new_grid


def test():
    puzzle(name="025d127b.json", transform=transform)
