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

    def change_color(self, to_color: int) -> 'Object':
        """
        Changes the color of all cells in the object to `to_color`.
        """
        new_data = [[to_color if cell != 0 else cell for cell in row]
                    for row in self.data]
        return Object(self.origin, new_data)

    def compact_left(self) -> 'Object':
        """
        Compacts each row of the object's grid data by shifting elements 
        to the left, effectively reducing the grid's width by one while 
        maintaining the overall structure of shapes.

        High-Level Operation:
        - **Shift Left**: Each row is adjusted to move elements to the 
          leftmost position possible.
        - **Remove Last BLACK**: If a row contains `BLACK`, the last 
          `BLACK` element is removed to enable the shift.
        - **Preserve Structure**: If a row does not contain `BLACK`, the 
          first element is removed, maintaining the structural integrity 
          of the shape.

        The result is a grid with a consistent width reduction, ensuring 
        all shapes remain compact and visually consistent.

        Returns:
            Object: A new `Object` instance with each row compacted to 
            the left, reducing the width by one.

        Example:
            Given an object with grid data:
            [
                [1, 2, BLACK, 3],
                [BLACK, 4, 5, BLACK],
                [6, 7, 8, 9]
            ]
            Calling `compact_left` will result in:
            [
                [1, 2, 3],     # Last BLACK is removed, row shifts left
                [BLACK, 4, 5], # Last BLACK is removed, row shifts left
                [7, 8, 9]      # No BLACK, first element is removed
            ]

        This operation is ideal for reducing grid size while preserving 
        the meaningful arrangement of elements.
        """

        def remove_last_black(lst: List[int]) -> List[int]:
            """
            Remove the last BLACK cell in the list.
            """
            if BLACK in lst:
                lst.reverse()
                lst.remove(BLACK)
                lst.reverse()
            else:
                lst = lst[1:]
            return lst

        def squash_row(row: List[int]) -> List[int]:
            """
            Process each row by removing the last BLACK cell or the first cell.
            """
            if BLACK in row:
                new_row = remove_last_black(row)
            else:
                new_row = row[1:]
            return new_row

        # Apply squash_row to each row in self.data
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
        data = [
            [1, 2, BLACK, 3],
            [BLACK, 4, 5, BLACK],
            [6, 7, 8, 9]
        ]
        obj = Object(origin=(0, 0), data=data)
        result = obj.compact_left()
        expected_data = [
            [1, 2, 3],
            [BLACK, 4, 5],
            [7, 8, 9]
        ]
        assert result.data == expected_data
