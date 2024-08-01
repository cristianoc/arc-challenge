from typing import Union

# Forward declaration of the recursive type RawGrid
RawGrid = Union[list[list[int]], list[list['RawGrid']]]


class Shape:
    def __init__(self, *dims: int):
        self.dims = list(dims)

    def total_elements(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim ** 2  # Each dimension is squared for a square shape
        return total

    def can_reshape(self, *new_dims: int):
        new_shape = Shape(*new_dims)
        return self.total_elements() == new_shape.total_elements()

    def view(self, *new_dims: int):
        if self.can_reshape(*new_dims):
            return Shape(*new_dims)
        else:
            raise ValueError(
                "New dimensions do not match total number of elements")

    def __eq__(self, other: object):
        if not isinstance(other, Shape):
            return False
        return self.dims == other.dims

    def __str__(self):
        return f"Shape(dims={self.dims})"

    @staticmethod
    def infer(grid: RawGrid) -> "Shape":
        """
        Infer the shape of a grid.
        """
        def get_size(grid: RawGrid) -> int:
            size = len(grid)
            for row in grid:
                if len(row) != size:
                    raise ValueError("Grid is not square")
            return size

        def nested_shape(g: RawGrid) -> list[int]:
            outer_size: int = get_size(g)
            if isinstance(g[0][0], list):
                inner_shape = nested_shape(g[0][0])
                return [outer_size] + inner_shape
            else:
                return [outer_size]

        return Shape(*nested_shape(grid))


def test_square_grid_initialization():
    shape1 = Shape(3)
    shape2 = Shape(3, 3)

    assert shape1.dims == [3], f"Expected [3], but got {shape1.dims}"
    assert shape2.dims == [3, 3], f"Expected [3, 3], but got {shape2.dims}"

    print("Shape Initialization Test Passed")


def test_total_elements():
    shape1 = Shape(3)
    shape2 = Shape(3, 3)

    assert shape1.total_elements(
    ) == 9, f"Expected 9, but got {shape1.total_elements()}"
    assert shape2.total_elements(
    ) == 81, f"Expected 81, but got {shape2.total_elements()}"

    print("Total Elements Test Passed")


def test_view():
    shape1 = Shape(3, 3)

    # Reshape from [3, 3] to [9]
    shape2 = shape1.view(9)
    assert shape2 == Shape(9), f"Expected Shape(9), but got {shape2}"

    # Reshape back from [9] to [3, 3]
    shape3 = shape2.view(3, 3)
    assert shape3 == Shape(3, 3), f"Expected Shape(3, 3), but got {shape3}"

    try:
        _invalid_shape = shape1.view(4)
        assert False, "Expected ValueError for invalid reshape"
    except ValueError:
        pass  # Expected behavior

    print("View Test Passed")
