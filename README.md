# Grid Transformation DSL

This DSL provides a structured and efficient way to perform complex grid transformations. It is designed to facilitate collaboration, ensure correctness, and support a wide range of grid manipulation tasks with a simple and intuitive API.

## Key Components

### `Grid` Class

The `Grid` class is the core of this DSL, offering a flexible interface to manipulate grid data. It supports various transformations, such as rotations, flips, translations, color changes, and more.

- **Create**: Initialize grids of specific dimensions with defined patterns or empty states.
- **Transform**: Rotate, flip, and translate grids to achieve desired spatial configurations.
- **Color Manipulation**: Change colors based on conditions, enabling dynamic visual transformations.
- **Object Detection**: Identify and manipulate distinct objects within grids for complex transformations.

**Basic Usage:**

```python
grid = Grid(data)
rotated_grid = grid.rotate(Clockwise)
flipped_grid = rotated_grid.flip(Horizontal)
final_grid = flipped_grid.color_change(from_color="red", to_color="blue")
```

### `Object` Class

The `Object` class encapsulates individual grid entities, allowing for refined operations like movement, color changes, and compacting. It is particularly useful for scenarios where distinct entities within a grid require independent transformations.

- **Movement**: Shift objects within the grid without altering others.
- **Color Changes**: Alter the color of entire objects.
- **Compacting**: Reduce object size or move them in specific directions for optimized layouts.

**Basic Usage:**

```python
objects = grid.detect_objects()
for obj in objects:
    transformed_obj = obj.change_color("yellow").move(0, 1)
```

### Types and Enums

- **`Direction`**: Used for operations like rotation and translation. Values include `Clockwise`, `CounterClockwise`, `Left`, `Right`, `Up`, and `Down`.
- **`Axis`**: Used for flipping operations. Values include `Horizontal` and `Vertical`.
- **`Color`**: Colors are represented as integers, allowing for flexible color manipulation.

## Illustrative Examples

Here are some representative examples of how to use the DSL for various grid transformation tasks:

### Example 1: Nested Grid Transformation

Transform a grid by replacing specific cells with nested grids, showcasing complex data transformations. Cells with a value of 0 are replaced by an empty grid, while non-zero cells are replaced by a copy of the original grid.

```python
from grid import Grid
from test_runner import puzzle

def transform(input: Grid) -> Grid:
    def map_func(x: int, y: int) -> Grid:
        color = input.data[x][y]
        return Grid.empty(input.size(), input.size()) if color == 0 else input.copy()
    return input.map_nested(map_func)

def test():
    puzzle(name="007bbfb7.json", transform=transform)
```

### Example 2: Color Change Based on Enclosure

Change the color of enclosed cells to yellow. A cell is enclosed if surrounded by non-zero cells, demonstrating conditional logic based on spatial relationships.

```python
from grid import Grid
from grid_data import YELLOW
from test_runner import puzzle

def transform(input: Grid) -> Grid:
    def map_func(x: int, y: int) -> int:
        color = input.data[x][y]
        return YELLOW if input.is_enclosed(x, y) else color
    return input.map(map_func)

def test():
    puzzle(name="00d62c1b.json", transform=transform)
```

### Example 3: Pattern Extension and Color Change

Identify the shortest repeating vertical pattern, extend it to a specified length, and change all occurrences of BLUE to RED, demonstrating pattern recognition and transformation.

```python
from grid import Grid
from grid_data import BLUE, RED
from shortest_period import find_shortest_period
from test_runner import puzzle

def transform(input: Grid) -> Grid:
    vertical_period = find_shortest_period(input.data)
    pattern = input.data[:vertical_period]
    extended_pattern = pattern * (9 // len(pattern)) + pattern[:9 % len(pattern)]
    grid = Grid(extended_pattern)
    return grid.color_change(BLUE, RED)

def test():
    puzzle(name="017c7c7b.json", transform=transform)
```

### Example 4: Object Detection and Compaction

Detect objects within a grid, compact them to the left by one cell, and create a new grid with adjusted dimensions.

```python
from grid import Grid
from test_runner import puzzle

def transform(input: Grid) -> Grid:
    objects = input.detect_objects()
    new_grid = Grid.empty(input.height, input.width)
    for obj in objects:
        new_grid.add_object(obj.compact_left().move(0, 1))
    return new_grid

def test():
    puzzle(name="025d127b.json", transform=transform)
```
