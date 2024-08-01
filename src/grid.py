import copy
from enum import Enum
from typing import Callable, Optional, List

from matplotlib import colors, pyplot as plt
from typing import NewType

import numpy as np

Color = NewType('Color', int)

BLACK: Color = Color(0)
RED: Color = Color(1)
GREEN: Color = Color(2)
BLUE: Color = Color(3)
YELLOW: Color = Color(4)
WHITE: Color = Color(5)


class Direction(str, Enum):
    CLOCKWISE = 'Clockwise'
    COUNTERCLOCKWISE = 'CounterClockwise'


class Axis(str, Enum):
    HORIZONTAL = 'Horizontal'
    VERTICAL = 'Vertical'


Raw = List[List[int]]


class Grid:
    def __init__(self, raw: Raw):
        self.raw = raw
        print(f"Raw: {self.raw}")

    def _rotate_grid(self, raw: Raw) -> Raw:
        return [list(reversed(col)) for col in zip(*raw)]

    def rotate(self, direction: Direction) -> 'Grid':
        if direction == Direction.CLOCKWISE:
            rotated_grid = self._rotate_grid(self.raw)
        else:
            rotated_grid = self._rotate_grid(
                self._rotate_grid(self._rotate_grid(self.raw)))
        return Grid(rotated_grid)

    def flip(self, axis: Axis) -> 'Grid':
        if axis == Axis.HORIZONTAL:
            flipped_grid: Raw = [row[::-1] for row in self.raw]
        else:
            flipped_grid: Raw = self.raw[::-1]
        return Grid(flipped_grid)

    def translate(self, dx: int, dy: int) -> 'Grid':
        new_grid: Raw = [[BLACK] * len(self.raw[0])
                         for _ in range(len(self.raw))]
        for y, row in enumerate(self.raw):
            for x, val in enumerate(row):
                new_x = (x + dx) % len(self.raw[0])
                new_y = (y + dy) % len(self.raw)
                new_grid[new_y][new_x] = val
        return Grid(new_grid)

    def color_change(self, from_color: Color, to_color: Color) -> 'Grid':
        new_grid = [[to_color if cell == from_color else cell for cell in row]
                    for row in self.raw]
        return Grid(new_grid)

    @staticmethod
    def empty(size: int) -> 'Grid':
        raw: Raw = [[BLACK for _ in range(size)] for _ in range(size)]
        return Grid(raw)

    def copy(self) -> 'Grid':
        return Grid(copy.deepcopy(self.raw))

    def size(self) -> int:
        return len(self.raw)

    def map(self, func: Callable[[int, int], 'Grid']) -> 'Grid':
        def transform_raw(raw: List[List[Raw]]) -> Raw:
            n = len(raw)
            n2 = n * n
            new_grid = [[0 for _ in range(n2)] for _ in range(n2)]

            for i in range(n):
                for j in range(n):
                    sub_grid = raw[i][j]
                    for sub_i in range(n):
                        for sub_j in range(n):
                            new_grid[i * n + sub_i][j * n +
                                                    sub_j] = sub_grid[sub_i][sub_j]

            return new_grid

        size = self.size()
        new_grid: List[List[Raw]] = [
            [func(i, j).raw for j in range(size)] for i in range(size)]
        return Grid(transform_raw(new_grid))

    def display(self, title: Optional[str] = None, output: Optional['Grid'] = None) -> None:
        data1 = self.raw

        # Define the custom color scheme as a list of colors
        color_scheme = [
            '#000000',  # black
            '#0074D9',  # blue
            '#FF4136',  # red
            '#2ECC40',  # green
            '#FFDC00',  # yellow
            '#AAAAAA',  # grey
            '#F012BE',  # fuschia
            '#FF851B',  # orange
            '#7FDBFF',  # teal
            '#870C25'   # brown
        ]

        # Create a ListedColormap with the specified colors
        cmap = colors.ListedColormap(color_scheme)

        # Adjust the bounds to match the number of colors
        bounds = np.arange(-0.5, len(color_scheme) - 0.5, 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Create a figure with one or two subplots depending on whether data2 is provided
        num_subplots = 2 if output is not None else 1
        fig, axes = plt.subplots(1, num_subplots, figsize=( # type: ignore
            5 * num_subplots, 5))  # type: ignore

        if num_subplots == 1:
            # Make sure axes is iterable if there's only one subplot
            axes = [axes]

        for ax, data, title_suffix in zip(axes, [data1, output.raw if output is not None else data1], ['Input', 'Output']):
            ax.set_facecolor('black')
            for i in range(len(data)):
                for j in range(len(data[0])):
                    rect = plt.Rectangle( # type: ignore
                        (j - 0.5, i - 0.5), 1, 1, edgecolor='grey', facecolor='none', linewidth=1)
                    ax.add_patch(rect)

            im = ax.imshow(data, cmap=cmap, norm=norm)  # type: ignore
            ax.set_title(
                f"{title} - {title_suffix}" if title else title_suffix)

        plt.tight_layout()
        plt.show() # type: ignore

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Grid):
            return self.raw == other.raw
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
