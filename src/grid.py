import copy
from enum import Enum
from typing import Callable

from matplotlib import colors, pyplot as plt



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


Raw = list[list[int]]


class Grid:
    def __init__(self, raw: Raw):
        self.raw = raw
        print(f"Raw: {self.raw}")

    def _rotate_grid(self, raw: Raw):
        return [list(reversed(col)) for col in zip(*raw)]

    def Rotate(self, direction: Direction):
        rotated_grid = self._rotate_grid(self.raw) if direction == Direction.Clockwise else self._rotate_grid(
            self._rotate_grid(self._rotate_grid(self.raw)))
        return Grid(rotated_grid)

    def Flip(self, axis: Axis):
        if axis == Axis.Horizontal:
            flipped_grid: Raw = [row[::-1] for row in self.raw]
        else:
            flipped_grid: Raw = self.raw[::-1]
        return Grid(flipped_grid)

    def Translate(self, dx: int, dy: int):
        new_grid: Raw = [[Black] * len(self.raw[0])
                         for _ in range(len(self.raw))]
        for y, row in enumerate(self.raw):
            for x, val in enumerate(row):
                new_x = (x + dx) % len(self.raw[0])
                new_y = (y + dy) % len(self.raw)
                new_grid[new_y][new_x] = val
        return Grid(new_grid)

    def ColorChange(self, from_color: Color, to_color: Color):
        new_grid = [[to_color if cell == from_color else cell for cell in row]
                    for row in self.raw]
        return Grid(new_grid)

    @staticmethod
    def empty(size: int) -> 'Grid':
        raw: Raw = [[Black for _ in range(size)] for _ in range(size)]
        return Grid(raw)

    def Copy(self):
        return Grid(copy.deepcopy(self.raw))

    def Size(self) -> int:
        return len(self.raw)

    def map(self, func: Callable[[int, int], 'Grid']) -> 'Grid':
        def transform_raw(raw: list[list[list[list[int]]]]):
            n = len(raw)  # This is the size of the outer grid
            n2 = n * n     # This is the size of the resulting grid
            # Initialize the new grid
            new_grid = [[0 for _ in range(n2)] for _ in range(n2)]

            for i in range(n):
                for j in range(n):
                    sub_grid = raw[i][j]
                    for sub_i in range(n):
                        for sub_j in range(n):
                            new_grid[i * n + sub_i][j * n +
                                                    sub_j] = sub_grid[sub_i][sub_j]

            return new_grid

        size = self.Size()
        new_grid: list[list[Raw]] = [[func(i, j).raw for j in range(size)]
                                     for i in range(size)]
        return Grid(transform_raw(new_grid))

    def Display(self) -> None:
        data = self.raw
        cmap = colors.ListedColormap(
            ['black', 'orange', 'green', 'blue', 'yellow', 'white'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
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
        plt.show()  # type: ignore

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Grid):
            return self.raw == other.raw
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
