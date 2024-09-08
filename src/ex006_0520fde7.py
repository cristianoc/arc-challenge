import numpy as np
from test_runner import puzzle
from grid_types import RED
from objects import Object


"""
This example extracts the leftmost and rightmost 3x3 sections of a grid 
and compares them. It creates a 3x3 output grid where each cell is colored 
red if both sections have non-zero values at the corresponding position, 
highlighting shared non-zero positions between the sections.
"""


def transform(input: Object) -> Object:
    # Extract leftmost and rightmost 3x3 sub-grids
    leftmost_grid = Object(input._data[:, :3])
    rightmost_grid = Object(input._data[:, -3:])

    output_grid = Object(np.zeros((3, 3), dtype=np.int64))

    # Set cell color to red if both sub-grids have non-zero value at the same position
    # Create boolean masks for non-zero elements in both grids
    left_mask = leftmost_grid._data != 0
    right_mask = rightmost_grid._data != 0
    
    # Combine masks to find shared non-zero positions
    shared_mask = np.logical_and(left_mask, right_mask)
    
    # Set RED color where shared_mask is True
    output_grid._data[shared_mask] = RED

    return output_grid


def test():
    puzzle(name="0520fde7.json", transform=transform)
