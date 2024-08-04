from dataclasses import dataclass
from typing import List, NewType, Optional, Tuple

from matplotlib import colors, pyplot as plt
import numpy as np

Cell = Tuple[int, int]
GridData = List[List[int]]


@dataclass
class Object:
    origin: Cell  # top-left corner of the bounding box
    data: GridData  # cells w.r.t the origin

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0]) if self.data else 0

    def move(self, dr: int, dc: int) -> 'Object':
        """
        Moves the object by `dr` rows and `dc` columns.
        """
        new_origin = (self.origin[0] + dr, self.origin[1] + dc)
        return Object(new_origin, self.data)

    def squash_left(self) -> 'Object':
        def remove_last_black(lst: List[int]):
            # Iterate over the list in reverse order
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] == BLACK:
                    del lst[i]
                    break
            return lst

        def squash_row(row: List[int]) -> List[int]:
            if BLACK in row:
                new_row = remove_last_black(row)
            else:
                new_row = row[1:]
            return new_row
        new_data = [squash_row(row) for row in self.data]
        return Object(self.origin, new_data)


Color = NewType('Color', int)

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

# Definitions using the indices
BLACK: Color = Color(0)   # #000000
BLUE: Color = Color(1)    # #0074D9
RED: Color = Color(2)     # #FF4136
GREEN: Color = Color(3)   # #2ECC40
YELLOW: Color = Color(4)  # #FFDC00
GREY: Color = Color(5)    # #AAAAAA
FUSCHIA: Color = Color(6)  # F012BE
ORANGE: Color = Color(7)  # #FF851B
TEAL: Color = Color(8)    # #7FDBFF
BROWN: Color = Color(9)   # #870C25


def display(input: GridData, output: Optional[GridData] = None, title: Optional[str] = None) -> None:
    # Create a ListedColormap with the specified colors
    cmap = colors.ListedColormap(color_scheme)

    # Adjust the bounds to match the number of colors
    bounds = np.arange(-0.5, len(color_scheme) - 0.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Create a figure with one or two subplots depending on whether data2 is provided
    num_subplots = 2 if output is not None else 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(  # type: ignore
        5 * num_subplots, 5))  # type: ignore

    if num_subplots == 1:
        # Make sure axes is iterable if there's only one subplot
        axes = [axes]

    for ax, data, title_suffix in zip(axes, [input, output if output is not None else input], ['Input', 'Output']):
        ax.set_facecolor('black')
        for i in range(len(data)):
            for j in range(len(data[0])):
                rect = plt.Rectangle(  # type: ignore
                    (j - 0.5, i - 0.5), 1, 1, edgecolor='grey', facecolor='none', linewidth=1)
                ax.add_patch(rect)

        im = ax.imshow(data, cmap=cmap, norm=norm)  # type: ignore
        ax.set_title(
            f"{title} - {title_suffix}" if title else title_suffix)

    plt.tight_layout()
    plt.show()  # type: ignore


class TestSquashLeft:

    # Removes the last occurrence of BLACK from each row containing BLACK
    def test_removes_last_black_occurrence(self):
        from src.grid_data import Object, BLACK
        data = [
            [1, 2, BLACK, 3],
            [BLACK, 4, 5, BLACK],
            [6, 7, 8, 9]
        ]
        obj = Object(origin=(0, 0), data=data)
        result = obj.squash_left()
        expected_data = [
            [1, 2, 3],
            [BLACK, 4, 5],
            [7, 8, 9]
        ]
        assert result.data == expected_data
