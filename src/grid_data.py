from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Optional, Tuple

from matplotlib import colors, pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

Cell = Tuple[int, int]
GridData = List[List[int]]

# Directions for moving in the grid: right, left, down, up
DIRECTIONS4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Direction vectors for 8 directions (N, NE, E, SE, S, SW, W, NW)
DIRECTIONS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]


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
    
    @property
    def size(self) -> Tuple[int, int]:
        return (self.height, self.width)

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

    @property
    def first_color(self) -> int:
        """
        Returns the first non-0 color detected in the object
        """
        for row in range(self.height):
            for col in range(self.width):
                    color = self.data[row][col]
                    if color != 0:
                        return color
        return 0

    @property
    def main_color(self) -> int:
        """
        Returns the most frequent color in the object.
        Raises a ValueError if there are no non-zero colors.
        """
        color_count: Dict[int, int] = {}
        for row in range(self.height):
            for col in range(self.width):
                color = self.data[row][col]
                if color != 0:
                    color_count[color] = color_count.get(color, 0) + 1   
        if not color_count:
            return self.first_color     
        return max(color_count, key=lambda item: item)

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
    
    def has_frame(self) -> bool:
        """
        Check if the object is a frame, i.e., has a border of 1 cell width and the color is not 0.

        A frame is defined as having non-zero cells in the entire first and last row,
        as well as the first and last column of the object. Additionally, the frame's color
        should be consistent and non-zero.
        """
        if self.height < 2 or self.width < 2:
            return False

        # Determine the object's color
        obj_color = self.main_color
        # if obj_color == 0:
        #     return False

        # Check top and bottom rows
        if not all(cell == obj_color for cell in self.data[0]) or not all(cell == obj_color for cell in self.data[-1]):
            return False

        # Check left and right columns
        for row in self.data:
            if row[0] != obj_color or row[-1] != obj_color:
                return False

        return True
    
    def is_block(self) -> bool:
        obj_color = self.first_color
        # Check if all cells have the same color
        for row in self.data:
            if any(cell != obj_color for cell in row):
                return False

        return True



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
    '#870C25',   # brown
    '#7FDBFF',  # teal
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
BROWN: Color = Color(8)   # #870C25
TEAL: Color = Color(9)    # #7FDBFF


def display(input: GridData, output: Optional[GridData] = None, title: Optional[str] = None) -> None:
    # Create a ListedColormap with the specified colors
    cmap : ListedColormap = colors.ListedColormap(color_scheme)

    # Adjust the bounds to match the number of colors
    bounds = np.arange(-0.5, len(color_scheme) - 0.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N) # type: ignore

    # Create a figure with one or two subplots depending on whether data2 is provided
    num_subplots = 2 if output is not None else 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(  # type: ignore
        5 * num_subplots, 5))  # type: ignore

    if num_subplots == 1:
        # Make sure axes is iterable if there's only one subplot
        axes: Any = [axes]

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

    plt.tight_layout() # type: ignore
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
