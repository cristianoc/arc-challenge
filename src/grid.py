import copy
from curses import raw
from enum import Enum
from typing import Callable, Optional

from matplotlib import colors, pyplot as plt

from shape import RawGrid, Shape


from typing import NewType

Color = NewType('Color', int)

Black: Color = Color(0)
Red: Color = Color(1)
Green: Color = Color(2)
Blue: Color = Color(3)
Yellow: Color = Color(4)
White: Color = Color(5)


class Direction(str, Enum):
    Clockwise = 'Clockwise'
    CounterClockwise = 'CounterClockwise'


class Axis(str, Enum):
    Horizontal = 'Horizontal'
    Vertical = 'Vertical'


class Grid:
    def __init__(self, raw: RawGrid, shape: Optional[Shape] = None):
        self.raw = raw
        if shape:
            self.shape = shape
        else:
            self.shape = Shape.infer(raw)
            print(f"Raw: {self.raw} Inferred shape: {self.shape}")

    def _rotate_grid(self, raw: RawGrid):
        return [list(reversed(col)) for col in zip(*raw)]

    def Rotate(self, direction: Direction):
        rotated_grid = self._rotate_grid(self.raw) if direction == Direction.Clockwise else self._rotate_grid(
            self._rotate_grid(self._rotate_grid(self.raw)))
        return Grid(rotated_grid, shape=self.shape)

    def Flip(self, axis: Axis):
        if axis == Axis.Horizontal:
            flipped_grid: RawGrid = [row[::-1]
                                     for row in self.raw]  # type: ignore
        else:
            flipped_grid: RawGrid = self.raw[::-1]
        return Grid(flipped_grid, shape=self.shape)

    def Translate(self, dx: int, dy: int):
        new_grid: list[list[RawGrid | int]] = [[Black]
                                               * len(self.raw[0]) for _ in range(len(self.raw))]
        for y, row in enumerate(self.raw):
            for x, val in enumerate(row):
                new_x = (x + dx) % len(self.raw[0])
                new_y = (y + dy) % len(self.raw)
                new_grid[new_y][new_x] = val
        ng: RawGrid = new_grid  # type: ignore
        return Grid(ng, shape=self.shape)

    def ColorChange(self, from_color: Color, to_color: Color):
        new_grid = [[to_color if cell == from_color else cell for cell in row]
                    for row in self.raw]
        ng: RawGrid = new_grid  # type: ignore
        return Grid(ng, shape=self.shape)

    @staticmethod
    def empty(shape: Shape) -> 'Grid':
        size = shape.dims[0]
        raw: RawGrid = [
            [Black for _ in range(size)] for _ in range(size)]
        return Grid(raw, shape=shape)

    def Copy(self):
        return Grid(copy.deepcopy(self.raw), shape=self.shape)

    def map(self, func: Callable[[int, int], 'Grid']) -> 'Grid':
        new_grid = [[func(i, j).raw for j in range(self.shape.dims[0])]
                    for i in range(self.shape.dims[0])]
        return Grid(new_grid, shape=self.shape)

    def Display(self) -> None:
        data = self.raw
        cmap = colors.ListedColormap(
            ['black', 'orange', 'green', 'blue', 'yellow', 'white'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        if isinstance(data[0][0], int):
            fig, ax = plt.subplots()  # type: ignore
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            _im = ax.imshow(data, cmap=cmap, norm=norm)

            # Add borders to each square
            for i in range(len(data)):
                for j in range(len(data[0])):
                    rect = plt.Rectangle(  # type: ignore
                        (j - 0.5, i - 0.5), 1, 1, edgecolor='grey', facecolor='none', linewidth=1)
                    ax.add_patch(rect)
            plt.axis('off')  # type: ignore
        else:
            assert False, "Displaying nested grids is not supported"
        plt.show()  # type: ignore

    # flatten a nested grid
    def flatten(self) -> 'Grid':
        if isinstance(self.raw[0][0], int):
            return self

        def transform_raw(raw: RawGrid):
            n = len(raw)  # This is the size of the outer grid
            n2 = n * n     # This is the size of the resulting grid
            new_grid = [[0 for _ in range(n2)] for _ in range(n2)]  # Initialize the new grid
            
            for i in range(n):
                for j in range(n):
                    sub_grid = raw[i][j]
                    for sub_i in range(n):
                        for sub_j in range(n):
                            new_grid[i * n + sub_i][j * n + sub_j] = sub_grid[sub_i][sub_j]
            
            return new_grid

        raw = transform_raw(self.raw)        
        return Grid(raw, shape=Shape(len(self.raw) * len(self.raw[0])))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Grid):
            return self.raw == other.raw and self.shape == other.shape
        return False


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
    grid = Grid([[Red, Blue], [Green, Red]])
    color_changed_grid = grid.ColorChange(Red, Yellow)
    assert color_changed_grid == Grid([[Yellow, Blue], [
                                      Green, Yellow]]), f"Expected [['yellow', 'blue'], ['green', 'yellow']], but got {color_changed_grid}"


def test_copy():
    grid = Grid([[1, 2], [3, 4]])
    copied_grid = grid.Copy()
    assert copied_grid == grid, f"Expected {grid}, but got {copied_grid}"
    assert copied_grid is not grid, "Copy should create a new instance"


def test_infer_shape():
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    shape = Grid(grid).shape
    assert shape == Shape(3), f"Expected Shape(3), but got {shape}"

    grid: RawGrid = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    shape = Grid(grid).shape
    assert shape == Shape(2, 3), f"Expected Shape(2, 3), but got {shape}"
