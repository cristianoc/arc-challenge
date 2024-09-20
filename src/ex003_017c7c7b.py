from typing import Any, List

import numpy as np

from grid_types import BLUE, RED
from objects import Object
from shortest_period import find_shortest_period  # type: ignore
from test_runner import puzzle

"""
This code defines a grid transformation process that involves identifying the shortest repeating vertical pattern,
extending the pattern to a specified length, and then changing all occurrences of the color BLUE to RED in the grid.
"""


def extend_list(lst: List[Any], length: int):
    """extend a list by repeating it until it reaches a certain length"""
    return lst * (length // len(lst)) + lst[: length % len(lst)]


def transform(input: Object) -> Object:
    data = input._data.tolist()
    vertical_period: int = find_shortest_period(data)
    pattern = data[:vertical_period]
    assert len(data) == 6
    extended_pattern = extend_list(pattern, 9)
    grid = Object(np.array(extended_pattern))
    return grid.color_change(BLUE, RED)


def test():
    puzzle(name="017c7c7b.json", transform=transform)
