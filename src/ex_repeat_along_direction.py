from detect_objects import detect_objects
from example_tester import example
from grid import Grid
from typing import List, Tuple

from grid_data import Object


def find_main_object(objects: List[Object]) -> Object:
    """Find and return the main 3x3 object from the detected objects."""
    for obj in objects:
        if obj.height == 3 and obj.width == 3:
            return obj
    assert False, "Main object not found"


def find_subsets(grid: Grid, main_object: 'Object', objects: List['Object']) -> List[Tuple[Tuple[int, int], int]]:
    """Find subset objects around the main object and determine their directions and colors."""
    directions = [(-1, 0), (-1, -1), (0, -1), (1, -1),
                  (1, 0), (1, 1), (0, 1), (-1, 1)]
    subsets: List[Tuple[Tuple[int, int], int]] = []

    for dr, dc in directions:
        # Calculate expected position for a subset
        off_row = main_object.origin[0] + 4 * dr
        off_col = main_object.origin[1] + 4 * dc
        # Find the first color in the subset object (assuming uniufor color)
        color = 0
        for r in range(main_object.height):
            for c in range(main_object.width):
                if color == 0:
                    try:
                        color = grid.data[r+off_row][c+off_col]
                    except IndexError:
                        pass
        if color != 0:
            subsets.append(((dr, dc), color))

    return subsets


def transform(input_grid: Grid) -> Grid:
    # Detect all objects in the grid
    objects = detect_objects(input_grid)

    # Find the main 3x3 object
    main_object = find_main_object(objects)

    if not main_object:
        return input_grid  # Return the original grid if no main object is found

    # Find subset objects and their directions
    subsets = find_subsets(input_grid, main_object, objects)

    # Create a new grid with the same dimensions as the input grid
    new_grid = Grid.empty(len(input_grid.data), len(input_grid.data[0]))

    # Add the main object to the new grid in its original position
    new_grid.add_object(main_object)

    # Extend the main object in the direction of subsets
    for direction, color in subsets:
        dr, dc = direction
        current_object = main_object

        while True:
            # Calculate new origin for the object
            new_origin = (
                current_object.origin[0] + 4 * dr, current_object.origin[1] + 4 * dc)

            # Check if the new object is within grid boundaries
            if not (0 <= new_origin[0] < len(input_grid.data) - 2 and
                    0 <= new_origin[1] < len(input_grid.data[0]) - 2):
                break

            # Create a new object with the specified color and new origin
            new_object = current_object.move(4 * dr, 4 * dc)

            # Change the color of the new object
            for r in range(current_object.height):
                for c in range(current_object.width):
                    if current_object.data[r][c] != 0:
                        new_object.data[r][c] = color

            # Add the new object to the grid
            new_grid.add_object(new_object)

            # Move to the next position
            current_object = new_object

    return new_grid


def test():
    example(name="045e512c.json", transform=transform)
