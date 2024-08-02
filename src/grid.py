import copy
from enum import Enum
from typing import Callable, List

from flood_fill import find_enclosed_areas
from grid_data import BLACK, BLUE, GREEN, RED, YELLOW, Color, GridData


class Direction(str, Enum):
    CLOCKWISE = 'Clockwise'
    COUNTERCLOCKWISE = 'CounterClockwise'


class Axis(str, Enum):
    HORIZONTAL = 'Horizontal'
    VERTICAL = 'Vertical'


class Grid:
    def __init__(self, data: GridData):
        self.data = data
        print(f"Data: {self.data}")

    def _rotate_grid(self, data: GridData) -> GridData:
        return [list(reversed(col)) for col in zip(*data)]

    def rotate(self, direction: Direction) -> 'Grid':
        if direction == Direction.CLOCKWISE:
            rotated_grid = self._rotate_grid(self.data)
        else:
            rotated_grid = self._rotate_grid(
                self._rotate_grid(self._rotate_grid(self.data)))
        return Grid(rotated_grid)

    def flip(self, axis: Axis) -> 'Grid':
        if axis == Axis.HORIZONTAL:
            flipped_grid: GridData = [row[::-1] for row in self.data]
        else:
            flipped_grid: GridData = self.data[::-1]
        return Grid(flipped_grid)

    def translate(self, dx: int, dy: int) -> 'Grid':
        new_grid: GridData = [[BLACK] * len(self.data[0])
                         for _ in range(len(self.data))]
        for y, row in enumerate(self.data):
            for x, val in enumerate(row):
                new_x = (x + dx) % len(self.data[0])
                new_y = (y + dy) % len(self.data)
                new_grid[new_y][new_x] = val
        return Grid(new_grid)

    def color_change(self, from_color: Color, to_color: Color) -> 'Grid':
        new_grid = [[to_color if cell == from_color else cell for cell in row]
                    for row in self.data]
        return Grid(new_grid)
    
    def is_enclosed(self, x: int, y: int) -> bool:
        if not hasattr(self, 'enclosed'):
            self.enclosed = find_enclosed_areas(self.data)
        return self.enclosed[x][y]

    @staticmethod
    def empty(size: int) -> 'Grid':
        data: GridData = [[BLACK for _ in range(size)] for _ in range(size)]
        return Grid(data)

    def copy(self) -> 'Grid':
        return Grid(copy.deepcopy(self.data))

    def size(self) -> int:
        return len(self.data)

    def map(self, func: Callable[[int, int], int]) -> 'Grid':
        new_grid = [[func(x, y) for y in range(len(self.data[0]))]
                    for x in range(len(self.data))]
        return Grid(new_grid)

    def map_nested(self, func: Callable[[int, int], 'Grid']) -> 'Grid':
        def transform_data(data: List[List[GridData]]) -> GridData:
            n = len(data)
            n2 = n * n
            new_grid = [[0 for _ in range(n2)] for _ in range(n2)]

            for i in range(n):
                for j in range(n):
                    sub_grid = data[i][j]
                    for sub_i in range(n):
                        for sub_j in range(n):
                            new_grid[i * n + sub_i][j * n +
                                                    sub_j] = sub_grid[sub_i][sub_j]

            return new_grid

        size = self.size()
        new_grid: List[List[GridData]] = [
            [func(i, j).data for j in range(size)] for i in range(size)]
        return Grid(transform_data(new_grid))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Grid):
            return self.data == other.data
        return False


# Test functions
def test_rotate():
    grid = Grid([[1, 2], [3, 4]])
    rotated_grid = grid.rotate(Direction.CLOCKWISE)
    assert rotated_grid == Grid(
        [[3, 1], [4, 2]]), f"Expected [[3, 1], [4, 2]], but got {rotated_grid}"

    rotated_grid = grid.rotate(Direction.COUNTERCLOCKWISE)
    assert rotated_grid == Grid(
        [[2, 4], [1, 3]]), f"Expected [[2, 4], [1, 3]], but got {rotated_grid}"


def test_flip():
    grid = Grid([[1, 2], [3, 4]])
    flipped_grid = grid.flip(Axis.HORIZONTAL)
    assert flipped_grid == Grid(
        [[2, 1], [4, 3]]), f"Expected [[2, 1], [4, 3]], but got {flipped_grid}"

    flipped_grid = grid.flip(Axis.VERTICAL)
    assert flipped_grid == Grid(
        [[3, 4], [1, 2]]), f"Expected [[3, 4], [1, 2]], but got {flipped_grid}"


def test_translate():
    grid = Grid([[1, 2], [3, 4]])
    translated_grid = grid.translate(1, 1)
    assert translated_grid == Grid(
        [[4, 3], [2, 1]]), f"Expected [[4, 3], [2, 1]], but got {translated_grid}"


def test_color_change():
    grid = Grid([[RED, BLUE], [GREEN, RED]])
    color_changed_grid = grid.color_change(RED, YELLOW)
    assert color_changed_grid == Grid([[YELLOW, BLUE], [
                                      GREEN, YELLOW]]), f"Expected [[YELLOW, BLUE], [GREEN, YELLOW]], but got {color_changed_grid}"


def test_copy():
    grid = Grid([[1, 2], [3, 4]])
    copied_grid = grid.copy()
    assert copied_grid == grid, f"Expected {grid}, but got {copied_grid}"
    assert copied_grid is not grid, "Copy should create a new instance"
