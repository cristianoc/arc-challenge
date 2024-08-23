
# SizeARC and Grid Transformation DSL

This repository explores a simplified version of the Abstraction and Reasoning Corpus (ARC) challenge, called SizeARC, along with the symbolic, specification-driven approach developed to solve it. The Grid Transformation DSL, a key component of this approach, is also described in some detail to illustrate the grid manipulation techniques.

## SizeARC: A Simplified ARC Challenge

**SizeARC** is a task derived from the ARC challenge, where the objective is to predict the dimensions of the output grid based on a given input grid. Unlike the full ARC challenge, which requires determining the entire content of the output grid, SizeARC simplifies the problem by focusing solely on predicting the size. This task utilizes the public ARC datasets and is agnostic to the methods used to solve it, making it an interesting case for testing various approaches to abstract reasoning.

### Problem Context

The ARC challenge is designed to test a system's ability to generalize from few examples, mimicking aspects of human cognition. SizeARC isolates a specific facet of the problem—predicting grid dimensions—to create a more focused challenge. While simpler, SizeARC contains a spectrum of difficulties, ranging from trivial cases to those as challenging as the original ARC tasks. This makes it a valuable benchmark for exploring generalization, abstraction, and reasoning in artificial intelligence.

## Solution Approach: Symbolic, Specification-Driven Method

To tackle SizeARC, we developed a purely symbolic, specification-driven approach. This method is based on selecting appropriate transformation specifications based on detected properties, rather than directly constructing programs through exhaustive enumeration or learning.

### Key Properties of the Approach

1. **Specification-Driven Selection**: The approach leverages predefined, high-level transformation rules. Rather than iteratively constructing solutions, it identifies which transformation rules apply to the detected features of the grid, guided by domain-specific priors.

2. **Domain-Specific Priors**: The approach incorporates priors about the domain to inform the selection of transformation rules, integrating knowledge that can guide the reasoning process.

3. **Semantic Reasoning**: By focusing on the "what" rather than the "how," the method emphasizes understanding the necessary transformations at a high level, relying on semantic reasoning rather than procedural construction.

4. **Layered Complexity**: While the initial stages of problem-solving are straightforward, progressively mode advanced techniques are applied only when the earlier stage fails.

### Results Summary

The method was tested across various configurations, showing a progession in tackling the SizeARC challenge:

- **Basic Predefined Transformations Only**: Achieved around 88% accuracy.
- **Adding Decision Rules on Matched Objects**: Improved accuracy to nearly 90%.
- **Also using Linear Programming**: Further increased accuracy to over 93%.

These results illustrate the potential of symbolic approaches in abstract reasoning tasks, particularly when combined with semantic reasoning and domain-specific knowledge. It remains to be seen whether this generalizes to the hidden data set, or whether it can be extended with at least some success to the full ARC problem.

### Key Techniques Used

1. **Basic Predefined Transformations**: The approach starts with predefined transformations that apply to input grids, guided by specific rules about how grid properties relate to the output size.

2. **Decision Rules on Matched Objects**: The method enhances accuracy by detecting objects within the grid, matching them across examples, and applying decision rules that leverage these matches to predict the output size.

3. **Linear Programming**: In more complex cases, linear programming is employed to refine predictions. This involves solving for weights and biases that best fit the observed data, allowing for a more accurate determination of grid dimensions.

## Implementation

For more details on the implementation of the methods used, start with the [predict_size.py](https://github.com/cristianoc/arc-challenge/blob/main/src/predict_size.py) file. This file provides a practical entry point into understanding how the symbolic approach was applied to solve the SizeARC challenge.


## Grid Transformation DSL

For those interested in the underlying mechanisms that support the SizeARC solution, the Grid Transformation DSL provides a robust framework for grid manipulations. The DSL enables various transformations such as rotations, flips, translations, and color changes, all of which are integral to the symbolic approach used.

### Key Components of the DSL

#### `Grid` Class

The `Grid` class serves as the core of the DSL, providing an interface for manipulating grid data. It supports operations that are essential for implementing transformation rules, such as:

- **Create**: Initialize grids with specific dimensions and patterns.
- **Transform**: Perform rotations, flips, and translations.
- **Color Manipulation**: Change colors based on defined conditions.
- **Object Detection**: Identify distinct objects within grids for targeted transformations.

#### `Object` Class

The `Object` class encapsulates grid entities, allowing for operations like movement, color changes, and compacting—crucial for scenarios where individual grid elements need independent handling.

#### Types and Enums

To facilitate grid operations, the DSL includes various types and enums, such as:

- **`Direction`**: Used for operations like rotation and translation.
- **`Axis`**: Used for flipping operations.
- **`Color`**: Colors represented as integers for flexible manipulation.

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
