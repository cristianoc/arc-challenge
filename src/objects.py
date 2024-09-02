from typing import Any, Dict, List, Optional, Tuple, Callable, Sequence

from matplotlib import colors, pyplot as plt  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore
import numpy as np
from grid_types import Cell, GridData, Rotation, Axis, BLACK, Color, RigidTransformation, XReflection
from detect_objects import ConnectedComponent, find_connected_components
from flood_fill import EnclosedCells, find_enclosed_cells
from grid_types import (
    BLUE,
    GREEN,
    RED,
    YELLOW,
    logger,
    Axis,
    Rotation,
    color_scheme,
) 

class Object:

    def __init__(self, data: np.ndarray[np.int64], origin: Cell = (0, 0)):
        self._data: np.ndarray[np.int64] = data
        self._data_cached: Optional[GridData] = None
        self._enclosed_cached: Optional[EnclosedCells] = None
        self.origin = origin

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Object):
            return self.datax == other.datax and self.origin == other.origin
        return False

    def format_grid(self, indent: str = '') -> str:
        return '\n'.join(indent + ' '.join(f'{cell:2}' for cell in row) for row in self.datax)

    def __str__(self) -> str:
        return self.format_grid()

    def __repr__(self) -> str:
        return f"Object(origin={self.origin}, data={self._data})"

    def __format__(self, format_spec: str) -> str:
        return f"\n{self.format_grid(' ')}"

    def __getitem__(self, key: Tuple[int, int]) -> int:
        return self.datax[key[1]][key[0]]

    def __setitem__(self, key: Tuple[int, int], value: int) -> None:
        self._data[key[0], key[1]] = value
        self._data_cached = None
        self._enclosed_cached = None

    def copy(self) -> "Object":
        return Object(self._data.copy())

    def apply_rigid_xform(self, xform: RigidTransformation) -> "Object":
        grid = self
        if xform.x_reflection == XReflection.REFLECT:
            grid = self.flip(Axis.HORIZONTAL)
        return grid.rot90_clockwise(xform.rotation.value)

    @property
    def datax(self) -> GridData:
        if self._data_cached is None:
            d: GridData = self._data.tolist()  # type: ignore
            self._data_cached = d
        return self._data_cached

    @property
    def height(self) -> int:
        return len(self.datax)

    @property
    def width(self) -> int:
        return len(self.datax[0]) if self.datax else 0

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    @staticmethod
    def empty(size: Tuple[int, int]) -> "Object":
        data: np.ndarray[np.int64] = np.zeros(size, dtype=np.int64)
        return Object(data)

    def add_object(self, obj: "Object") -> None:
        [r_off, c_off] = obj.origin
        for r in range(obj.height):
            for c in range(obj.width):
                color = obj.datax[r][c]
                if color != BLACK:
                    # only add the color if it's in bounds
                    if 0 <= r + r_off < len(self.datax) and 0 <= c + c_off < len(
                        self.datax[0]
                    ):
                        self.datax[r + r_off][c + c_off] = color

    def map(self, func: Callable[[int, int], int]) -> "Object":
        new_grid = [
            [func(x, y) for y in range(len(self.datax[0]))]
            for x in range(len(self.datax))
        ]
        return Object(np.array(new_grid))

    def map_nested(self, func: Callable[[int, int], "Object"]) -> "Object":
        def transform_data(data: List[List[GridData]]) -> GridData:
            n = len(data)
            n2 = n * n
            new_grid = [[0 for _ in range(n2)] for _ in range(n2)]

            for i in range(n):
                for j in range(n):
                    sub_grid = data[i][j]
                    for sub_i in range(n):
                        for sub_j in range(n):
                            new_grid[i * n + sub_i][j * n + sub_j] = sub_grid[sub_i][
                                sub_j
                            ]

            return new_grid

        new_grid: List[List[GridData]] = [
            [func(i, j).datax for j in range(self.width)] for i in range(self.height)
        ]
        return Object(np.array(transform_data(new_grid)))

    def is_enclosed(self, x: int, y: int) -> bool:
        if self._enclosed_cached is None:
            self._enclosed_cached = find_enclosed_cells(self.datax)
        return self._enclosed_cached[x][y]

    def color_change(self, from_color: Color, to_color: Color) -> "Object":
        new_grid = [
            [to_color if cell == from_color else cell for cell in row]
            for row in self.datax
        ]
        return Object(np.array(new_grid))

    def detect_objects(
        self: "Object", diagonals: bool = True, allow_black: bool = False
    ) -> List["Object"]:
        def create_object(grid: Object, component: ConnectedComponent) -> Object:
            """
            Create an object from a connected component in a grid
            """
            min_row = min(r for r, _ in component)
            min_col = min(c for _, c in component)
            rows = max(r for r, _ in component) - min_row + 1
            columns = max(c for _, c in component) - min_col + 1
            data = Object.empty(size=(rows, columns)).datax  # Fix dimensions here
            for r, c in component:
                data[r - min_row][c - min_col] = grid.datax[r][c]
            return Object(origin=(min_row, min_col), data=np.array(data))

        connected_components = find_connected_components(
            self.datax, diagonals, allow_black
        )
        detected_objects = [
            create_object(self, component) for component in connected_components
        ]
        return detected_objects

    def rot90_clockwise(self, n: int) -> "Object":
        x: np.ndarray = np.rot90(self._data, -n)  # type: ignore
        return Object(x)

    def fliplr(self) -> "Object":
        x: np.ndarray = np.fliplr(self._data)  # type: ignore
        return Object(x)

    def flipud(self) -> "Object":
        x: np.ndarray = np.flipud(self._data)  # type: ignore
        return Object(x)

    def invert(self) -> "Object":
        x: np.ndarray = 1 - self._data  # type: ignore
        return Object(x)

    def rotate(self, direction: Rotation) -> "Object":
        data: List[List[int]] = self.datax
        height, width = len(data), len(data[0])
        rotated_grid = [[0 for _ in range(height)] for _ in range(width)]
        if direction == Rotation.CLOCKWISE:
            return self.rot90_clockwise(1)
        else:  # Rotation.COUNTERCLOCKWISE
            return self.rot90_clockwise(-1)
        
    def translate(self, dy: int, dx: int) -> "Object":
        height, width = len(self.datax), len(self.datax[0])
        new_grid: GridData = [[BLACK] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                new_x = x + dx
                new_y = y + dy
                # Ensure the new position is within bounds
                if 0 <= new_x < width and 0 <= new_y < height:
                    new_grid[new_y][new_x] = self.datax[y][x]

        return Object(np.array(new_grid))

    def flip(self, axis: Axis) -> "Object":
        if axis == Axis.HORIZONTAL:
            return self.fliplr()
        else:
            return self.flipud()

    def num_cells(self, color: Optional[int]) -> int:
        if color is None:
            return sum(cell != 0 for row in self.datax for cell in row)
        else:
            return sum(cell == color for row in self.datax for cell in row)

    def move(self, dr: int, dc: int) -> "Object":
        """
        Moves the object by `dr` rows and `dc` columns.
        """
        new_origin = (self.origin[0] + dr, self.origin[1] + dc)
        return Object(self._data, new_origin)

    def change_color(self, from_color: Optional[int], to_color: int) -> "Object":
        """
        Changes the color of all cells in the object to `to_color`.
        """
        new_data = [
            [
                (
                    to_color
                    if color == from_color or (color != 0 and from_color is None)
                    else color
                )
                for color in row
            ]
            for row in self.datax
        ]
        return Object(np.array(new_data), self.origin)

    def contains_cell(self, cell: Cell) -> bool:
        """
        Checks if the cell is within the object's bounding box.
        """
        row, col = cell
        r, c = self.origin
        return r <= row < r + self.height and c <= col < c + self.width

    def get_colors(self, allow_black: bool = True) -> List[int]:
        colors: set[Color] = set()
        for row in self.datax:
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
                color = self.datax[row][col]
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
                color = self.datax[row][col]
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
        new_data = [squash_row(row) for row in self.datax]
        return Object(np.array(new_data), self.origin)

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
        if not all(cell == obj_color for cell in self.datax[0]) or not all(
            cell == obj_color for cell in self.datax[-1]
        ):
            return False

        # Check left and right columns
        for row in self.datax:
            if row[0] != obj_color or row[-1] != obj_color:
                return False

        return True

    def is_block(self) -> bool:
        obj_color = self.first_color
        # Check if all cells have the same color
        for row in self.datax:
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
            ax_output.imshow(input_data, cmap="gray")  # type: ignore
            ax_output.set_title(f"Output Grid {i+1} (None)")

        ax_output.axis("off")  # Hide the axes

    if title:
        fig.suptitle(title, fontsize=16)

    # Adjust layout to make room for title
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


class TestSquashLeft:
    # Removes the last occurrence of BLACK from each row containing BLACK
    def test_removes_last_black_occurrence(self):
        data = [[1, 2, BLACK, 3], [BLACK, 4, 5, BLACK], [6, 7, 8, 9]]
        obj = Object(np.array(data))
        result = obj.compact_left()
        expected_data = [[1, 2, 3], [BLACK, 4, 5], [7, 8, 9]]
        assert result.datax == expected_data


def test_rotate():
    grid = Object(np.array([[1, 2], [3, 4]]))
    rotated_grid = grid.rotate(Rotation.CLOCKWISE)
    assert rotated_grid == Object(
        np.array([[3, 1], [4, 2]])
    ), f"Expected [[3, 1], [4, 2]], but got {rotated_grid}"

    rotated_grid = grid.rotate(Rotation.COUNTERCLOCKWISE)
    assert rotated_grid == Object(
        np.array([[2, 4], [1, 3]])
    ), f"Expected [[2, 4], [1, 3]], but got {rotated_grid}"


def test_flip():
    grid = Object(np.array([[1, 2], [3, 4]]))
    flipped_grid = grid.flip(Axis.HORIZONTAL)
    assert flipped_grid == Object(
        np.array([[2, 1], [4, 3]])
    ), f"Expected [[2, 1], [4, 3]], but got {flipped_grid}"

    flipped_grid = grid.flip(Axis.VERTICAL)
    assert flipped_grid == Object(
        np.array([[3, 4], [1, 2]])
    ), f"Expected [[3, 4], [1, 2]], but got {flipped_grid}"


def test_translate():
    grid = Object(np.array([[1, 2], [3, 4]]))
    translated_grid = grid.translate(1, 1)
    assert np.array_equal(translated_grid._data, np.array([[0, 0], [0, 1]]))  # Use np.array_equal


def test_color_change():
    grid = Object(np.array([[RED, BLUE], [GREEN, RED]]))
    color_changed_grid = grid.color_change(RED, YELLOW)
    assert color_changed_grid == Object(
        np.array([[YELLOW, BLUE], [GREEN, YELLOW]])
    ), f"Expected [[YELLOW, BLUE], [GREEN, YELLOW]], but got {color_changed_grid}"


def test_copy():
    grid = Object(np.array([[1, 2], [3, 4]]))
    copied_grid = grid.copy()
    assert copied_grid == grid, f"Expected {grid}, but got {copied_grid}"
    assert copied_grid is not grid, "Copy should create a new instance"


def test_detect_objects():
    grid = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    objects = Object(grid).detect_objects()
    for obj in objects:
        logger.info(f"Detected object: {obj}")
