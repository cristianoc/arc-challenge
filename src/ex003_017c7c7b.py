from typing import List, TypeVar
from test_runner import puzzle
from grid import Grid
from grid_data import BLUE, RED
from shortest_period import find_shortest_period

"""
This code defines a grid transformation process that involves identifying the shortest repeating vertical pattern,
extending the pattern to a specified length, and then changing all occurrences of the color BLUE to RED in the grid.
"""

T = TypeVar('T')


def extend_list(lst: List[T], length: int):
    '''extend a list by repeating it until it reaches a certain length'''
    return lst * (length // len(lst)) + lst[:length % len(lst)]


def transform(input_grid: Grid) -> Grid:
    vertical_period = find_shortest_period(input_grid.data)
    pattern = input_grid.data[:vertical_period]
    assert len(input_grid.data) == 6
    extended_pattern = extend_list(pattern, 9)
    grid = Grid(extended_pattern)
    return grid.color_change(BLUE, RED)


def test():
    puzzle(name="017c7c7b.json", transform=transform)
