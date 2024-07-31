class SquareGrid:
    def __init__(self, *dims):
        self.dims = list(dims)

    def total_elements(self):
        total = 1
        for dim in self.dims:
            total *= dim
        return total

    def can_reshape(self, *new_dims):
        new_shape = SquareGrid(*new_dims)
        return self.total_elements() == new_shape.total_elements()

    def view(self, *new_dims):
        if self.can_reshape(*new_dims):
            return SquareGrid(*new_dims)
        else:
            raise ValueError("New dimensions do not match total number of elements")

    def __eq__(self, other):
        return self.dims == other.dims

    def __str__(self):
        return f"SquareGrid(dims={self.dims})"


def test_square_grid_initialization():
    shape1 = SquareGrid(3)
    shape2 = SquareGrid(3, 3)

    assert shape1.dims == [3], f"Expected [3], but got {shape1.dims}"
    assert shape2.dims == [3, 3], f"Expected [3, 3], but got {shape2.dims}"

    print("SquareGrid Initialization Test Passed")

def test_total_elements():
    shape1 = SquareGrid(3)
    shape2 = SquareGrid(3, 3)

    assert shape1.total_elements() == 3, f"Expected 3, but got {shape1.total_elements()}"
    assert shape2.total_elements() == 9, f"Expected 9, but got {shape2.total_elements()}"

    print("Total Elements Test Passed")

def test_view():
    shape1 = SquareGrid(3, 3)
    
    # Reshape from [3, 3] to [9]
    shape2 = shape1.view(9)
    assert shape2 == SquareGrid(9), f"Expected SquareGrid(9), but got {shape2}"

    # Reshape back from [9] to [3, 3]
    shape3 = shape2.view(3, 3)
    assert shape3 == SquareGrid(3, 3), f"Expected SquareGrid(3, 3), but got {shape3}"

    try:
        invalid_shape = shape1.view(4)
        assert False, "Expected ValueError for invalid reshape"
    except ValueError:
        pass  # Expected behavior

    print("View Test Passed")

