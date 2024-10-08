from typing import List, Tuple

from grid_types import DIRECTIONS8
from objects import Object
from test_runner import puzzle

"""
This example demonstrates a grid transformation process where a main object
is identified and extended in specified directions based on detected subset objects.
The algorithm first detects all objects in the grid, identifies a primary 3x3 object,
and then searches for related subset objects positioned in specific directions
around the main object. It uses this information to transform the grid by 
extending the main object in the direction of the subsets, replicating and
coloring them based on detected subset properties. The process ultimately
illustrates how to manipulate and extend patterns within a grid-like structure.
"""


def find_max_area_object(objects: List[Object]) -> Object:
    """Find and return the largest object from the detected objects."""
    return max(objects, key=lambda obj: obj.area)


def find_subsets(
    grid: Object, main_object: "Object", objects: List["Object"]
) -> List[Tuple[Tuple[int, int], int]]:
    """Find subset objects around the main object and determine their directions and colors."""
    subsets: List[Tuple[Tuple[int, int], int]] = []

    for dx, dy in DIRECTIONS8:
        # Calculate expected position for a subset
        off_x = main_object.origin[0] + 4 * dx
        off_y = main_object.origin[1] + 4 * dy

        # Find the first color in the subset object (assuming uniform color)
        color = 0
        for x in range(main_object.width):
            for y in range(main_object.height):
                if color == 0:
                    try:
                        color = grid[x + off_x, y + off_y]
                    except IndexError:
                        continue
        if color != 0:
            subsets.append(((dx, dy), color))

    return subsets


def transform(input: Object) -> Object:
    # Detect all objects in the grid
    objects = input.detect_objects()

    # Find the main 3x3 object
    main_object = find_max_area_object(objects)

    if not main_object:
        return input  # Return the original grid if no main object is found

    # Find subset objects and their directions
    subsets = find_subsets(input, main_object, objects)

    # Create a new grid with the same dimensions as the input grid
    new_grid = Object.empty(input.size)

    # Add the main object to the new grid in its original position
    new_grid.add_object_in_place(main_object)

    # Extend the main object in the direction of subsets
    for direction, color in subsets:
        dx, dy = direction
        current_object = main_object

        while True:
            # Calculate new origin for the object
            new_origin = current_object.origin + 4 * (dx, dy)

            # Check if the new object is completely outside the grid boundaries
            if (
                new_origin[0] >= input.width
                or new_origin[1] >= input.height
                or new_origin[0] + current_object.width <= 0
                or new_origin[1] + current_object.height <= 0
            ):
                break

            # Create a new object with the specified color and new origin
            new_object = current_object.move(4 * dx, 4 * dy).change_color(None, color)

            # Add the new object to the grid
            new_grid.add_object_in_place(new_object)

            # Move to the next position
            current_object = new_object

    return new_grid


def test():
    puzzle(name="045e512c.json", transform=transform)
