from typing import Any, Dict, List, Optional, Tuple, Callable, Sequence

plt: Any
from matplotlib import colors, pyplot as plt  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore
import numpy as np
from grid_types import (
    Cell,
    GridData,
    Rotation,
    Axis,
    BLACK,
    Color,
    RigidTransformation,
    XReflection,
)
from detect_objects import ConnectedComponent, find_connected_components
from flood_fill import EnclosedCells, find_enclosed_cells
from logger import logger
from grid_types import (
    BLUE,
    GREEN,
    RED,
    YELLOW,
    Axis,
    Rotation,
    color_scheme,
)
import visual_cortex


class Object:

    def __init__(
        self,
        data: np.ndarray,
        origin: Cell = (0, 0),
    ):
        self._data: np.ndarray = data
        self._enclosed_cached: Optional[EnclosedCells] = None
        self.origin = origin

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Object):
            return (
                np.array_equal(self._data, other._data) and self.origin == other.origin
            )
        return False

    def format_grid(self) -> str:
        return f"{self._data}"

    def __str__(self) -> str:
        return self.format_grid()

    def __repr__(self) -> str:
        return f"Object(origin={self.origin}, data={self._data})"

    def __format__(self, format_spec: str) -> str:
        return f"\n{self.format_grid()}"

    def __getitem__(self, key: Tuple[int, int]) -> int:
        return self._data[key[1], key[0]]

    def __setitem__(self, key: Tuple[int, int], value: int) -> None:
        self._data[key[1], key[0]] = value  # Corrected the indices
        self.clear_caches()

    def clear_caches(self) -> None:
        """
        Clear the caches for the object whenever the data is modified.
        """
        self._enclosed_cached = None

    def copy(self) -> "Object":
        return Object(origin=self.origin, data=self._data.copy())

    def apply_rigid_xform(self, xform: RigidTransformation) -> "Object":
        grid = self.rot90_clockwise(xform.rotation.value)
        if xform.x_reflection == XReflection.REFLECT:
            grid = grid.fliplr()
        return grid

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    @staticmethod
    def empty(size: Tuple[int, int], background_color: int = 0) -> "Object":
        width, height = size
        data = np.full((height, width), background_color, dtype=np.int64)
        return Object(data)

    def add_object_in_place(self, obj: "Object", background_color: int = 0) -> None:
        """
        Add the object to the grid.
        """
        self.clear_caches()
        x_off, y_off = obj.origin
        for x in range(obj.width):
            for y in range(obj.height):
                color = obj[x, y]
                if color != background_color:
                    # Only add the color if it's in bounds
                    new_x, new_y = x + x_off, y + y_off
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        self[new_x, new_y] = color

    def remove_object_in_place(self, obj: "Object") -> None:
        """
        Remove the object from the grid.
        """
        self.clear_caches()
        x_off, y_off = obj.origin
        for x in range(obj.width):
            for y in range(obj.height):
                color = obj[x, y]
                if color != 0:
                    self[x + x_off, y + y_off] = 0

    def map(self, func: Callable[[int, int], int]) -> "Object":
        new_grid = [[func(x, y) for y in range(self.width)] for x in range(self.height)]
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
            [func(i, j)._data.tolist() for j in range(self.width)]
            for i in range(self.height)
        ]
        return Object(np.array(transform_data(new_grid)))

    def is_enclosed(self, x: int, y: int) -> bool:
        if self._enclosed_cached is None:
            self._enclosed_cached = find_enclosed_cells(self)
        return self._enclosed_cached[x][y]

    def color_change(self, from_color: Color, to_color: Color) -> "Object":
        new_data = np.where(self._data == from_color, to_color, self._data)
        return Object(new_data)

    def detect_objects(
        self: "Object",
        diagonals: bool = True,
        allow_black: bool = False,
        background_color: int = 0,
        multicolor: bool = False,
    ) -> List["Object"]:
        def create_object(
            grid: Object, component: ConnectedComponent, background_color: int
        ) -> Object:
            """
            Create an object from a connected component in a grid
            """
            min_x = min(x for x, _ in component)
            min_y = min(y for _, y in component)
            max_x = max(x for x, _ in component)
            max_y = max(y for _, y in component)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            data = Object.empty(size=(width, height), background_color=background_color)
            for x, y in component:
                q = grid[x, y]
                data[x - min_x, y - min_y] = q
            return Object(origin=(min_x, min_y), data=data._data)

        connected_components = find_connected_components(
            grid=self, diagonals=diagonals, allow_black=allow_black, multicolor=multicolor
        )
        detected_objects = [
            create_object(self, component, background_color)
            for component in connected_components
        ]
        return detected_objects

    def rot90_clockwise(self, n: int) -> "Object":
        x: np.ndarray = np.rot90(self._data.copy(), -n)
        return Object(origin=self.origin, data=x)

    def fliplr(self) -> "Object":
        x: np.ndarray = np.fliplr(self._data.copy())
        return Object(origin=self.origin, data=x)

    def flipud(self) -> "Object":
        x: np.ndarray = np.flipud(self._data.copy())
        return Object(origin=self.origin, data=x)

    def invert(self) -> "Object":
        x: np.ndarray = 1 - self._data.copy()
        return Object(origin=self.origin, data=x)

    def rotate(self, direction: Rotation) -> "Object":
        if direction == Rotation.CLOCKWISE:
            return self.rot90_clockwise(1)
        else:  # Rotation.COUNTERCLOCKWISE
            return self.rot90_clockwise(-1)

    def translate_in_place(self, dx: int, dy: int) -> None:
        self.clear_caches()
        new_origin = (self.origin[0] + dx, self.origin[1] + dy)
        self.origin = new_origin

    def flip(self, axis: Axis) -> "Object":
        if axis == Axis.HORIZONTAL:
            return self.fliplr()
        else:
            return self.flipud()

    def num_cells(self, color: Optional[int]) -> int:
        if color is None:
            # Count all non-zero cells
            return np.count_nonzero(self._data != 0)
        else:
            # Count cells equal to the specified color
            return np.count_nonzero(self._data == color)

    def move(self, dx: int, dy: int) -> "Object":
        """
        Moves the object by `dr` rows and `dc` columns.
        """
        new_origin = (self.origin[0] + dx, self.origin[1] + dy)
        return Object(self._data.copy(), new_origin)

    def change_color(self, from_color: Optional[int], to_color: int) -> "Object":
        """
        Changes the color of all cells in the object to `to_color`.
        """
        old_data = self._data

        if from_color is None:
            # Change all non-zero colors to `to_color` when `from_color` is None
            new_data = np.where(old_data != 0, to_color, old_data)
        else:
            # Change colors that match `from_color` to `to_color`
            new_data = np.where(old_data == from_color, to_color, old_data)

        return Object(new_data.copy(), self.origin)

    def contains_cell(self, cell: Cell) -> bool:
        """
        Checks if the cell is within the object's bounding box.
        """
        row, col = cell
        r, c = self.origin
        return r <= row < r + self.height and c <= col < c + self.width

    def get_colors(self, allow_black: bool = True) -> List[int]:
        colors = np.unique(self._data)
        if not allow_black:
            colors = colors[colors != BLACK]
        return sorted(colors.tolist())

    def get_shape(self, background_color: int = 0) -> "Object":
        """
        Make the background color 0, and the rest 1.
        """
        return Object(np.where(self._data == background_color, 0, 1).copy())

    @property
    def first_color(self) -> int:
        """
        Returns the first non-0 color detected in the object
        """
        for row in range(self.height):
            for col in range(self.width):
                color = self._data[row, col]
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
                color = self._data[row, col]
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

        def remove_last_black(row: np.ndarray) -> np.ndarray:
            """
            Remove the last BLACK cell in the row (NumPy array).
            If no BLACK is found, remove the first element of the row.
            """
            if BLACK in row:
                # Get index of the last BLACK and remove it
                last_black_index = np.max(np.where(row == BLACK))
                row = np.delete(row, last_black_index)
            else:
                # Remove the first element if no BLACK is found
                row = row[1:]
            return row

        def squash_row(row: np.ndarray) -> np.ndarray:
            """
            Process each row by removing the last BLACK cell or the first cell.
            """
            return remove_last_black(row)

        # Apply squash_row to each row of the NumPy array
        new_data = np.array([squash_row(row) for row in self._data])

        return Object(new_data.copy(), self.origin)

    def has_frame(self) -> bool:
        """
        Check if the object has a frame (border of 1 cell width) and the color is not 0.

        A frame is defined as having non-zero cells in the entire first and last row,
        as well as the first and last column of the object. Additionally, the frame's color
        should be consistent and non-zero.
        """
        if self.height < 2 or self.width < 2:
            return False

        # Determine the object's color
        obj_color = self.main_color()

        # Check top and bottom rows
        if not np.all(self._data[0, :] == obj_color) or not np.all(
            self._data[-1, :] == obj_color
        ):
            return False

        # Check left and right columns
        if not np.all(self._data[:, 0] == obj_color) or not np.all(
            self._data[:, -1] == obj_color
        ):
            return False

        return True

    def is_block(self) -> bool:
        obj_color = self.first_color
        # Check if all cells have the same color
        return bool(np.all(self._data == obj_color))

    def detect_colored_objects(
        self, background_color: Optional[int] = None
    ) -> List["Object"]:
        """
        Detects and returns a list of all distinct objects within the grid based on color.

        This method uses the implementation from visual_cortex.py.
        """
        return visual_cortex.find_colored_objects(self, background_color)


def display(
    input: Object, output: Object = Object(np.array([[0]])), title: Optional[str] = None
) -> None:
    display_multiple([(input, output)], title)


def display_multiple(
    grid_pairs: List[Tuple[Object, Object]], title: Optional[str] = None
) -> None:
    num_pairs = len(grid_pairs)

    # Create a Matplotlib figure with multiple rows, two subplots per row
    fig, axes = plt.subplots(
        num_pairs, 2, figsize=(4, 2 * num_pairs)
    )  # Adjust figsize as needed

    # If there's only one pair, axes won't be a list, so we wrap it in a list
    if num_pairs == 1:
        axes = [axes]  # type: ignore

    for i, (input, output) in enumerate(grid_pairs):
        input_data = input._data
        output_data = output._data
        ax_input: Any
        ax_output: Any
        ax_input, ax_output = axes[i]

        # Custom rectangle-based plot for input grid with provided color scheme
        for row in range(input_data.shape[0]):
            for col in range(input_data.shape[1]):
                color = color_scheme[input_data[row, col] % len(color_scheme)]
                rect = plt.Rectangle(
                    [col, row], 1, 1, facecolor=color, edgecolor="grey", linewidth=0.5
                )
                ax_input.add_patch(rect)

        # Set limits, aspect ratio, and grid appearance for the input grid
        ax_input.set_xlim(0, input_data.shape[1])
        ax_input.set_ylim(0, input_data.shape[0])
        ax_input.set_aspect("equal")
        ax_input.invert_yaxis()  # Invert to match `imshow`
        ax_input.set_title(f"Input Grid {i+1}")
        ax_input.axis("off")  # Hide the axes

        if output_data is not None:
            # Custom rectangle-based plot for output grid with provided color scheme
            for row in range(output_data.shape[0]):
                for col in range(output_data.shape[1]):
                    color = color_scheme[output_data[row, col] % len(color_scheme)]
                    rect = plt.Rectangle(
                        [col, row],
                        1,
                        1,
                        facecolor=color,
                        edgecolor="grey",
                        linewidth=0.5,
                    )
                    ax_output.add_patch(rect)

            # Set limits, aspect ratio, and grid appearance for the output grid
            ax_output.set_xlim(0, output_data.shape[1])
            ax_output.set_ylim(0, output_data.shape[0])
            ax_output.set_aspect("equal")
            ax_output.invert_yaxis()  # Invert to match `imshow`
            ax_output.set_title(f"Output Grid {i+1}")
        else:
            # If output_data is None, just show a blank plot
            ax_output.set_title(f"Output Grid {i+1} (None)")
            ax_output.axis("off")  # Hide the axes for a blank plot

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
        assert result == Object(np.array(expected_data))


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
    obj = grid.copy()
    obj.translate_in_place(1, 1)
    new_grid = Object.empty(obj.size)
    new_grid.add_object_in_place(obj)
    assert new_grid == Object(np.array([[0, 0], [0, 1]]))


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
    grid = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    objects = Object(grid).detect_objects()
    for obj in objects:
        logger.info(f"Detected object: {obj}")
