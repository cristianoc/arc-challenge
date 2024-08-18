from abc import ABC, abstractmethod
import copy
from enum import Enum
from typing import Callable, List, Tuple
from detect_objects import ConnectedComponent, find_connected_components
from flood_fill import find_enclosed_cells
from grid_data import BLACK, BLUE, GREEN, RED, YELLOW, Color, GridData, Object


class Direction(str, Enum):
    CLOCKWISE = 'Clockwise'
    COUNTERCLOCKWISE = 'CounterClockwise'


class Axis(str, Enum):
    HORIZONTAL = 'Horizontal'
    VERTICAL = 'Vertical'


class GridA(ABC):
    def __init__(self, data: GridData):
        self.data = data

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0])
    
    @property
    def size(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @abstractmethod
    def detect_objects(self) -> List[Object]:
        pass

    @staticmethod
    @abstractmethod
    def empty(height: int, width: int) -> 'GridA':
        pass

    @abstractmethod
    def rotate(self, direction: Direction) -> 'GridA':
        pass

    @abstractmethod
    def flip(self, axis: Axis) -> 'GridA':
        pass

    @abstractmethod
    def translate(self, dx: int, dy: int) -> 'GridA':
        pass

    @abstractmethod
    def color_change(self, from_color: Color, to_color: Color) -> 'GridA':
        pass

    def is_enclosed(self, x: int, y: int) -> bool:
        if not hasattr(self, 'enclosed'):
            self.enclosed = find_enclosed_cells(self.data)
        return self.enclosed[x][y]

    @abstractmethod
    def add_object(self, obj: Object) -> None:
        pass


class Grid(GridA):
    def __init__(self, data: GridData):
        super().__init__(data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Grid):
            return self.data == other.data
        return False

    def _rotate_grid(self, data: GridData) -> GridData:
        return [list(reversed(col)) for col in zip(*data)]

    def add_object(self, obj: Object) -> None:
        [r_off, c_off] = obj.origin
        for r in range(obj.height):
            for c in range(obj.width):
                color = obj.data[r][c]
                if color != BLACK:
                    # only add the color if it's in bounds
                    if 0 <= r + r_off < len(self.data) and 0 <= c + c_off < len(self.data[0]):
                        self.data[r + r_off][c + c_off] = color

    def color_change(self, from_color: Color, to_color: Color) -> 'Grid':
        new_grid = [[to_color if cell == from_color else cell for cell in row]
                    for row in self.data]
        return Grid(new_grid)

    def copy(self) -> 'Grid':
        return Grid(copy.deepcopy(self.data))

    def detect_objects(self: 'Grid') -> List[Object]:
        def create_object(grid: Grid, component: ConnectedComponent) -> Object:
            """
            Create an object from a connected component in a grid
            """
            min_row = min(r for r, _ in component)
            min_col = min(c for _, c in component)
            rows = max(r for r, _ in component) - min_row + 1
            columns = max(c for _, c in component) - min_col + 1
            data = Grid.empty(height=rows, width=columns).data
            for r, c in component:
                data[r - min_row][c - min_col] = grid.data[r][c]
            return Object((min_row, min_col), data)
        connected_components = find_connected_components(self.data)
        detected_objects = [create_object(self, component)
                            for component in connected_components]
        return detected_objects

    @staticmethod
    def empty(height: int, width: int) -> 'Grid':
        data: GridData = [[BLACK for _ in range(width)] for _ in range(height)]
        return Grid(data)

    def flip(self, axis: Axis) -> 'Grid':
        if axis == Axis.HORIZONTAL:
            flipped_grid: GridData = [row[::-1] for row in self.data]
        else:
            flipped_grid: GridData = self.data[::-1]
        return Grid(flipped_grid)

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

        new_grid: List[List[GridData]] = [
            [func(i, j).data for j in range(self.width)] for i in range(self.height)]
        return Grid(transform_data(new_grid))

    def rotate(self, direction: Direction) -> 'Grid':
        if direction == Direction.CLOCKWISE:
            rotated_grid = self._rotate_grid(self.data)
        else:
            rotated_grid = self._rotate_grid(
                self._rotate_grid(self._rotate_grid(self.data)))
        return Grid(rotated_grid)

    def translate(self, dx: int, dy: int) -> 'Grid':
        new_grid: GridData = [[BLACK] * len(self.data[0])
                              for _ in range(len(self.data))]
        for y, row in enumerate(self.data):
            for x, val in enumerate(row):
                new_x = (x + dx) % len(self.data[0])
                new_y = (y + dy) % len(self.data)
                new_grid[new_y][new_x] = val
        return Grid(new_grid)


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


def test_detect_objects():
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    objects = Grid(grid).detect_objects()
    for obj in objects:
        print(f"Detected object: {obj}")
