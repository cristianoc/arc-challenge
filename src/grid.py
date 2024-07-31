import copy

class Direction:
    Clockwise = 'Clockwise'
    CounterClockwise = 'CounterClockwise'
    Left = 'Left'
    Right = 'Right'
    Up = 'Up'
    Down = 'Down'

class Axis:
    Horizontal = 'Horizontal'
    Vertical = 'Vertical'

class Grid:
    def __init__(self, grid):
        self.grid = grid

    def Rotate(self, direction):
        if direction == Direction.Clockwise:
            return Grid([list(reversed(col)) for col in zip(*self.grid)])
        elif direction == Direction.CounterClockwise:
            return Grid([list(col) for col in reversed(list(zip(*self.grid)))])
        else:
            raise ValueError("Invalid rotation direction")

    def Flip(self, axis):
        if axis == Axis.Horizontal:
            return Grid([row[::-1] for row in self.grid])
        elif axis == Axis.Vertical:
            return Grid(self.grid[::-1])
        else:
            raise ValueError("Invalid flip axis")

    def Translate(self, dx, dy):
        new_grid = [[None]*len(self.grid[0]) for _ in range(len(self.grid))]
        for y, row in enumerate(self.grid):
            for x, val in enumerate(row):
                new_x = (x + dx) % len(self.grid[0])
                new_y = (y + dy) % len(self.grid)
                new_grid[new_y][new_x] = val
        return Grid(new_grid)

    def ColorChange(self, from_color, to_color):
        new_grid = [[to_color if cell == from_color else cell for cell in row] for row in self.grid]
        return Grid(new_grid)

    def Copy(self):
        return Grid(copy.deepcopy(self.grid))

    def __eq__(self, other):
        return self.grid == other.grid

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.grid])

# Test functions

def test_rotate():
    grid = Grid([[1, 2], [3, 4]])
    rotated_grid = grid.Rotate(Direction.Clockwise)
    assert rotated_grid == Grid([[3, 1], [4, 2]]), f"Expected [[3, 1], [4, 2]], but got {rotated_grid}"

    rotated_grid = grid.Rotate(Direction.CounterClockwise)
    assert rotated_grid == Grid([[2, 4], [1, 3]]), f"Expected [[2, 4], [1, 3]], but got {rotated_grid}"

def test_flip():
    grid = Grid([[1, 2], [3, 4]])
    flipped_grid = grid.Flip(Axis.Horizontal)
    assert flipped_grid == Grid([[2, 1], [4, 3]]), f"Expected [[2, 1], [4, 3]], but got {flipped_grid}"

    flipped_grid = grid.Flip(Axis.Vertical)
    assert flipped_grid == Grid([[3, 4], [1, 2]]), f"Expected [[3, 4], [1, 2]], but got {flipped_grid}"

def test_translate():
    grid = Grid([[1, 2], [3, 4]])
    translated_grid = grid.Translate(1, 1)
    assert translated_grid == Grid([[4, 3], [2, 1]]), f"Expected [[4, 3], [2, 1]], but got {translated_grid}"

def test_color_change():
    grid = Grid([['red', 'blue'], ['green', 'red']])
    color_changed_grid = grid.ColorChange('red', 'yellow')
    assert color_changed_grid == Grid([['yellow', 'blue'], ['green', 'yellow']]), f"Expected [['yellow', 'blue'], ['green', 'yellow']], but got {color_changed_grid}"

def test_copy():
    grid = Grid([[1, 2], [3, 4]])
    copied_grid = grid.Copy()
    assert copied_grid == grid, f"Expected {grid}, but got {copied_grid}"
    assert copied_grid is not grid, "Copy should create a new instance"

def run_tests():
    test_rotate()
    test_flip()
    test_translate()
    test_color_change()
    test_copy()
    return "All tests passed."

run_tests()
