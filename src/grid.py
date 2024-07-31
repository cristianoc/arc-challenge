import copy
from typing import Any
from shape import Shape


class Direction:
    Clockwise = 'Clockwise'
    CounterClockwise = 'CounterClockwise'
    Left = 'Left'
    Right = 'Right'
    Up = 'Up'
    Down = 'Down'


class Axis:
    Horizontal = 'Horizontal'
    Vertical = 'Vertical'


class Grid:
    def __init__(self, grid, shape=None):
        self.grid = grid
        if shape:
            self.shape = Shape(*shape)
        else:
            self.shape = self.infer_shape(grid)
            print(f"Grid: {self.grid}")
            print(f"Inferred shape: {self.shape}")

    # infer a shape
    # example: [[1,2], [3,4], [5,6], [7,8]] infers Shape(2), i.e. 2x2
    # nested example (of shape Shape(2,2) i.e. 2x2x2x2):

    def infer_shape(self, grid):
        # From a list of lists, determine the size and check it is a square
        # Check that everty element is a list of the same lenght as the outer list.
        def get_size(grid: list[list[Any]]) -> int:
            # check that it's a list
            if not isinstance(grid, list):
                raise ValueError("Grid is not a list")
            size = len(grid)
            # check that it's a list of lists
            if not isinstance(grid[0], list):
                raise ValueError("Grid is not a list of lists")
            for row in grid:
                if len(row) != size:
                    raise ValueError("Grid is not square")
            return size

        def nested_shape(g: list[list[Any]]) -> list[int]:
            outer_size: int = get_size(g)
            if isinstance(g[0][0], list):
                inner_shape = nested_shape(g[0][0])
                return [outer_size] + inner_shape
            else:
                return [outer_size]

        return Shape(*nested_shape(grid))

    def _rotate_grid(self, grid):
        return [list(reversed(col)) for col in zip(*grid)]

    def Rotate(self, direction):
        rotated_grid = self._rotate_grid(self.grid) if direction == Direction.Clockwise else self._rotate_grid(
            self._rotate_grid(self._rotate_grid(self.grid)))
        return Grid(rotated_grid, shape=self.shape.dims)

    def Flip(self, axis):
        flipped_grid = [
            row[::-1] for row in self.grid] if axis == Axis.Horizontal else self.grid[::-1]
        return Grid(flipped_grid, shape=self.shape.dims)

    def Translate(self, dx, dy):
        new_grid = [[None]*len(self.grid[0]) for _ in range(len(self.grid))]
        for y, row in enumerate(self.grid):
            for x, val in enumerate(row):
                new_x = (x + dx) % len(self.grid[0])
                new_y = (y + dy) % len(self.grid)
                new_grid[new_y][new_x] = val
        return Grid(new_grid, shape=self.shape.dims)

    def ColorChange(self, from_color, to_color):
        new_grid = [[to_color if cell == from_color else cell for cell in row]
                    for row in self.grid]
        return Grid(new_grid, shape=self.shape.dims)

    def Copy(self):
        return Grid(copy.deepcopy(self.grid), shape=self.shape.dims)

    def __getitem__(self, idx):
        elem = self.grid
        for i in idx:
            elem = elem[i]
        return elem

    def __setitem__(self, idx, value):
        elem = self.grid
        for i in idx[:-1]:
            elem = elem[i]
        elem[idx[-1]] = value

    def __eq__(self, other):
        return self.grid == other.grid and self.shape == other.shape

    def __str__(self):
        def to_string(grid):
            if isinstance(grid[0], list):
                return '\n'.join([to_string(row) for row in grid])
            else:
                return ' '.join(map(str, grid))
        return to_string(self.grid)

# Test functions


def test_rotate():
    grid = Grid([[1, 2], [3, 4]])
    rotated_grid = grid.Rotate(Direction.Clockwise)
    assert rotated_grid == Grid(
        [[3, 1], [4, 2]]), f"Expected [[3, 1], [4, 2]], but got {rotated_grid}"

    rotated_grid = grid.Rotate(Direction.CounterClockwise)
    assert rotated_grid == Grid(
        [[2, 4], [1, 3]]), f"Expected [[2, 4], [1, 3]], but got {rotated_grid}"


def test_flip():
    grid = Grid([[1, 2], [3, 4]])
    flipped_grid = grid.Flip(Axis.Horizontal)
    assert flipped_grid == Grid(
        [[2, 1], [4, 3]]), f"Expected [[2, 1], [4, 3]], but got {flipped_grid}"

    flipped_grid = grid.Flip(Axis.Vertical)
    assert flipped_grid == Grid(
        [[3, 4], [1, 2]]), f"Expected [[3, 4], [1, 2]], but got {flipped_grid}"


def test_translate():
    grid = Grid([[1, 2], [3, 4]])
    translated_grid = grid.Translate(1, 1)
    assert translated_grid == Grid(
        [[4, 3], [2, 1]]), f"Expected [[4, 3], [2, 1]], but got {translated_grid}"


def test_color_change():
    grid = Grid([['red', 'blue'], ['green', 'red']])
    color_changed_grid = grid.ColorChange('red', 'yellow')
    assert color_changed_grid == Grid([['yellow', 'blue'], [
                                      'green', 'yellow']]), f"Expected [['yellow', 'blue'], ['green', 'yellow']], but got {color_changed_grid}"


def test_copy():
    grid = Grid([[1, 2], [3, 4]])
    copied_grid = grid.Copy()
    assert copied_grid == grid, f"Expected {grid}, but got {copied_grid}"
    assert copied_grid is not grid, "Copy should create a new instance"


def test_infer_shape():
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    shape = Grid(grid).shape
    assert shape == Shape(3), f"Expected Shape(3), but got {shape}"

    grid = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    shape = Grid(grid).shape
    assert shape == Shape(2, 3), f"Expected Shape(2, 3), but got {shape}"
