from typing import Any, Dict, List, Optional, Tuple

from matplotlib import colors, pyplot as plt  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore
import numpy as np  # type: ignore
from grid_types import Cell, GridData, Rotation, Axis, BLACK, Color

class Object:

    def __init__(self, data: GridData, origin: Cell = (0, 0)):
        self.data = data
        self.origin = origin

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Object):
            return self.data == other.data
        return False

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0]) if self.data else 0

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def size(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def rotate(self, direction: Rotation) -> 'Object':
        data: List[List[int]] = self.data
        height, width = len(data), len(data[0])
        rotated_grid = [[0 for _ in range(height)] for _ in range(width)]
        if direction == Rotation.CLOCKWISE:
            for i in range(height):
                for j in range(width):
                    rotated_grid[j][height - 1 - i] = data[i][j]
        else:  # Rotation.COUNTERCLOCKWISE
            for i in range(height):
                for j in range(width):
                    rotated_grid[width - 1 - j][i] = data[i][j]
        return Object(rotated_grid)

    def translate(self, dy: int, dx: int) -> 'Object':
        height, width = len(self.data), len(self.data[0])
        new_grid: GridData = [[BLACK] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                new_x = x + dx
                new_y = y + dy
                # Ensure the new position is within bounds
                if 0 <= new_x < width and 0 <= new_y < height:
                    new_grid[new_y][new_x] = self.data[y][x]

        return Object(new_grid)

    def flip(self, axis: Axis) -> "Object":
        if axis == Axis.HORIZONTAL:
            flipped_grid: GridData = [row[::-1] for row in self.data]
        else:
            flipped_grid: GridData = self.data[::-1]
        return Object(flipped_grid)

    def num_cells(self, color: Optional[int]) -> int:
        if color is None:
            return sum(cell != 0 for row in self.data for cell in row)
        else:
            return sum(cell == color for row in self.data for cell in row)

    def move(self, dr: int, dc: int) -> "Object":
        """
        Moves the object by `dr` rows and `dc` columns.
        """
        new_origin = (self.origin[0] + dr, self.origin[1] + dc)
        return Object(self.data, new_origin)

    def change_color(self, from_color: Optional[int], to_color: int) -> "Object":
        """
        Changes the color of all cells in the object to `to_color`.
        """
        new_data = [
            [
                to_color
                if color == from_color or (color != 0 and from_color is None)
                else color
                for color in row
            ]
            for row in self.data
        ]
        return Object(new_data, self.origin)

    def contains_cell(self, cell: Cell) -> bool:
        """
        Checks if the cell is within the object's bounding box.
        """
        row, col = cell
        r, c = self.origin
        return r <= row < r + self.height and c <= col < c + self.width

    def get_colors(self, allow_black: bool = True) -> List[int]:
        colors: set[Color] = set()
        for row in self.data:
            for color in row:
                if color == BLACK and not allow_black:
                    continue
                colors.add(color)  # type: ignore
        return sorted(list(colors))

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

    def main_color(self, allow_black: bool = False) -> int:
        """
        Returns the most frequent color in the object.
        Raises a ValueError if there are no non-zero colors.
        """
        color_count: Dict[int, int] = {}
        for row in range(self.height):
            for col in range(self.width):
                color = self.data[row][col]
                if allow_black or color != 0:
                    color_count[color] = color_count.get(color, 0) + 1
        if not color_count:
            return self.first_color
        return max(color_count, key=lambda c: color_count.get(c, 0))

    def compact_left(self) -> "Object":
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
        return Object(new_data,self.origin)

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
        obj_color = self.main_color()

        # Check top and bottom rows
        if not all(cell == obj_color for cell in self.data[0]) or not all(
            cell == obj_color for cell in self.data[-1]
        ):
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




def display(
    input: GridData, output: Optional[GridData] = None, title: Optional[str] = None
) -> None:
    display_multiple([(input, output)], title)


plt: Any = plt


def display_multiple(
    grid_pairs: List[Tuple[GridData, Optional[GridData]]], title: Optional[str] = None
) -> None:
    num_pairs = len(grid_pairs)

    # Create a Matplotlib figure with multiple rows, two subplots per row
    fig, axes = plt.subplots(
        num_pairs, 2, figsize=(4, 2 * num_pairs)
    )  # Adjust figsize as needed

    # If there's only one pair, axes won't be a list, so we wrap it in a list
    if num_pairs == 1:
        axes = [axes]

    for i, (input_data, output_data) in enumerate(grid_pairs):
        ax_input, ax_output = axes[i]

        # Plot the input grid
        cmap: ListedColormap = colors.ListedColormap(color_scheme)  # type: ignore
        # Adjust the bounds to match the number of colors
        bounds = np.arange(-0.5, len(color_scheme) + 0.5, 1)  # type: ignore
        norm = colors.BoundaryNorm(bounds, cmap.N)  # type: ignore
        ax_input.imshow(input_data, cmap=cmap, norm=norm)
        ax_input.set_title(f"Input Grid {i+1}")
        ax_input.axis("off")  # Hide the axes

        if output_data is not None:
            # Plot the output grid if provided
            ax_output.imshow(output_data, cmap=cmap, norm=norm)
            ax_output.set_title(f"Output Grid {i+1}")
        else:
            # If output_data is None, just show a blank plot
            ax_output.imshow(np.zeros_like(input_data), cmap="gray")  # type: ignore
            ax_output.set_title(f"Output Grid {i+1} (None)")

        ax_output.axis("off")  # Hide the axes

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
    plt.show()


class TestSquashLeft:
    # Removes the last occurrence of BLACK from each row containing BLACK
    def test_removes_last_black_occurrence(self):
        data = [[1, 2, BLACK, 3], [BLACK, 4, 5, BLACK], [6, 7, 8, 9]]
        obj = Object(origin=(0, 0), data=data)
        result = obj.compact_left()
        expected_data = [[1, 2, 3], [BLACK, 4, 5], [7, 8, 9]]
        assert result.data == expected_data
