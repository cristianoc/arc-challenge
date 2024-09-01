import copy
from enum import Enum
from typing import Callable, List, Tuple
from detect_objects import (
    ConnectedComponent,
    find_connected_components,
    find_rectangular_objects,
)
from flood_fill import find_enclosed_cells
from grid_data import BLACK, BLUE, GREEN, RED, YELLOW, Color, GridData, Object, logger, Cell  # Added Cell import


class Rotation(str, Enum):
    CLOCKWISE = "Clockwise"
    COUNTERCLOCKWISE = "CounterClockwise"


class Axis(str, Enum):
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"


class Grid:
    def __init__(self, data: GridData, origin:Cell = (0, 0)):
        self.data = data
        self.origin = origin

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Grid):
            return self.data == other.data
        return False

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0])

    @property
    def size(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def is_enclosed(self, x: int, y: int) -> bool:
        if not hasattr(self, "enclosed"):
            self.enclosed = find_enclosed_cells(self.data)
        return self.enclosed[x][y]

    def add_object(self, obj: Object) -> None:
        [r_off, c_off] = obj.origin
        for r in range(obj.height):
            for c in range(obj.width):
                color = obj.data[r][c]
                if color != BLACK:
                    # only add the color if it's in bounds
                    if 0 <= r + r_off < len(self.data) and 0 <= c + c_off < len(
                        self.data[0]
                    ):
                        self.data[r + r_off][c + c_off] = color

    def color_change(self, from_color: Color, to_color: Color) -> "Grid":
        new_grid = [
            [to_color if cell == from_color else cell for cell in row]
            for row in self.data
        ]
        return Grid(new_grid)

    def copy(self) -> "Grid":
        return Grid(copy.deepcopy(self.data))

    def get_colors(self, allow_black: bool = True) -> List[int]:
        colors: set[Color] = set()
        for row in self.data:
            for color in row:
                if color == BLACK and not allow_black:
                    continue
                colors.add(color)  # type: ignore
        return sorted(list(colors))

    def detect_objects(
        self: "Grid", diagonals: bool = True, allow_black: bool = False
    ) -> List[Object]:
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
            return Object(data, (min_row, min_col))

        connected_components = find_connected_components(
            self.data, diagonals, allow_black
        )
        detected_objects = [
            create_object(self, component) for component in connected_components
        ]
        return detected_objects

    def detect_rectangular_objects(
        self, allow_multicolor: bool = False
    ) -> List[Object]:
        return find_rectangular_objects(self.data, allow_multicolor=allow_multicolor)

    @staticmethod
    def empty(height: int, width: int) -> "Grid":
        data: GridData = [[BLACK for _ in range(width)] for _ in range(height)]
        return Grid(data)

    def flip(self, axis: Axis) -> "Grid":
        if axis == Axis.HORIZONTAL:
            flipped_grid: GridData = [row[::-1] for row in self.data]
        else:
            flipped_grid: GridData = self.data[::-1]
        return Grid(flipped_grid)

    def map(self, func: Callable[[int, int], int]) -> "Grid":
        new_grid = [
            [func(x, y) for y in range(len(self.data[0]))]
            for x in range(len(self.data))
        ]
        return Grid(new_grid)

    def map_nested(self, func: Callable[[int, int], "Grid"]) -> "Grid":
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
            [func(i, j).data for j in range(self.width)] for i in range(self.height)
        ]
        return Grid(transform_data(new_grid))

    def rotate(self, direction: Rotation) -> "Grid":
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
        return Grid(rotated_grid)

    def translate(self, dy: int, dx: int) -> "Grid":
        height, width = len(self.data), len(self.data[0])
        new_grid: GridData = [[BLACK] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                new_x = x + dx
                new_y = y + dy
                # Ensure the new position is within bounds
                if 0 <= new_x < width and 0 <= new_y < height:
                    new_grid[new_y][new_x] = self.data[y][x]

        return Grid(new_grid)


# Test functions


def test_rotate():
    grid = Grid([[1, 2], [3, 4]])
    rotated_grid = grid.rotate(Rotation.CLOCKWISE)
    assert rotated_grid == Grid(
        [[3, 1], [4, 2]]
    ), f"Expected [[3, 1], [4, 2]], but got {rotated_grid}"

    rotated_grid = grid.rotate(Rotation.COUNTERCLOCKWISE)
    assert rotated_grid == Grid(
        [[2, 4], [1, 3]]
    ), f"Expected [[2, 4], [1, 3]], but got {rotated_grid}"


def test_flip():
    grid = Grid([[1, 2], [3, 4]])
    flipped_grid = grid.flip(Axis.HORIZONTAL)
    assert flipped_grid == Grid(
        [[2, 1], [4, 3]]
    ), f"Expected [[2, 1], [4, 3]], but got {flipped_grid}"

    flipped_grid = grid.flip(Axis.VERTICAL)
    assert flipped_grid == Grid(
        [[3, 4], [1, 2]]
    ), f"Expected [[3, 4], [1, 2]], but got {flipped_grid}"


def test_translate():
    grid = Grid([[1, 2], [3, 4]])
    translated_grid = grid.translate(1, 1)
    assert translated_grid.data == [[0, 0], [0, 1]]


def test_color_change():
    grid = Grid([[RED, BLUE], [GREEN, RED]])
    color_changed_grid = grid.color_change(RED, YELLOW)
    assert color_changed_grid == Grid(
        [[YELLOW, BLUE], [GREEN, YELLOW]]
    ), f"Expected [[YELLOW, BLUE], [GREEN, YELLOW]], but got {color_changed_grid}"


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
        logger.info(f"Detected object: {obj}")


def test_detect_rectangular_objects():
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    objects: List[Object] = Grid(grid).detect_rectangular_objects()
    for obj in objects:
        logger.info(f"Detected rectangular object: {obj}")
    object_dims = [(obj.origin, obj.size) for obj in objects]
    assert object_dims == [((1, 1), (4, 4))]


def test_several_rectangular_objects_of_different_color():
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 2, 0],
        [0, 0, 1, 0, 2, 2],
        [0, 0, 0, 1, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    objects = Grid(grid).detect_rectangular_objects()
    for obj in objects:
        logger.info(f"Detected rectangular object: {obj}")
    object_dims = [(obj.origin, obj.size) for obj in objects]
    assert object_dims == [((1, 1), (4, 3)), ((2, 4), (3, 2))]
