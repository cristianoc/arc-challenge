from test_runner import puzzle
from grid_types import RED
from grid_data import Object


"""
This example extracts the leftmost and rightmost 3x3 sections of a grid 
and compares them. It creates a 3x3 output grid where each cell is colored 
red if both sections have non-zero values at the corresponding position, 
highlighting shared non-zero positions between the sections.
"""


def transform(input: Object) -> Object:
    # Extract leftmost and rightmost 3x3 sub-grids
    leftmost_grid = Object([row[:3] for row in input.data])
    rightmost_grid = Object([row[-3:] for row in input.data])

    output_grid = Object([[0 for _ in range(3)] for _ in range(3)])

    # Set cell color to red if both sub-grids have non-zero value at the same position
    for i in range(3):
        for j in range(3):
            if leftmost_grid.data[i][j] != 0 and rightmost_grid.data[i][j] != 0:
                output_grid.data[i][j] = RED

    return output_grid


def test():
    puzzle(name="0520fde7.json", transform=transform)
