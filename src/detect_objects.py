from typing import List

from grid_data import DIRECTIONS4, DIRECTIONS8, Cell, GridData, Object, logger

# Type alias for the id of visited objects
VISITED = List[List[bool]]

# Defining a type alias for a connected component
ConnectedComponent = List[Cell]


def dfs_recursive_list(data: GridData, visited: VISITED, r: int, c: int, color: int, component: ConnectedComponent, diagonals: bool):
    # Base conditions to stop recursion
    if r < 0 or r >= len(data) or c < 0 or c >= len(data[0]):
        return
    if visited[r][c] or data[r][c] != color:
        return

    # Mark the cell as visited
    visited[r][c] = True

    # Add the cell to the current component
    component.append((r, c))

    # Recursively visit all 8 neighbors
    directions = DIRECTIONS8 if diagonals else DIRECTIONS4
    for dr, dc in directions:
        dfs_recursive_list(data, visited, r + dr, c + dc,
                           color, component, diagonals)


def find_connected_components(data: GridData, diagonals: bool, allow_black: bool) -> List[ConnectedComponent]:
    rows = len(data)
    cols = len(data[0])
    visited: VISITED = [[False for _ in range(cols)] for _ in range(rows)]
    connected_components: List[ConnectedComponent] = []

    for r in range(rows):
        for c in range(cols):
            # Skip cells with color 0
            if visited[r][c] == False and (allow_black or data[r][c] != 0):
                # Create a new component
                component: ConnectedComponent = []
                dfs_recursive_list(data, visited, r, c,
                                   data[r][c], component, diagonals)
                # Add the component to the list of connected components
                if component:
                    connected_components.append(component)

    return connected_components


def find_rectangular_objects(data: GridData, allow_multicolor: bool) -> List[Object]:
    objects: List[Object] = []
    rows, cols = len(data), len(data[0])

    def cell_contained_in_objects(cell: Cell) -> bool:
        return any(obj.contains_cell(cell) for obj in objects)

    def is_valid_rectangle(origin: Cell, height: int, width: int, color: int) -> bool:
        start_r, start_c = origin
        if start_r < 0 or start_c < 0 or start_r + height > rows or start_c + width > cols:
            return False
        for r in range(start_r, start_r + height):
            for c in range(start_c, start_c + width):
                if not allow_multicolor and data[r][c] != color and data[r][c] != 0:
                    return False
        # check that the first and last rows and columns are not all 0
        if all(data[start_r][c] == 0 for c in range(start_c, start_c + width)):
            return False
        if all(data[start_r + height - 1][c] == 0 for c in range(start_c, start_c + width)):
            return False
        if all(data[r][start_c] == 0 for r in range(start_r, start_r + height)):
            return False
        if all(data[r][start_c + width - 1] == 0 for r in range(start_r, start_r + height)):
            return False
        return True

    for r in range(rows):
        for c in range(cols):
            if not cell_contained_in_objects((r, c)) and data[r][c] != 0:
                main_color = data[r][c]
                origin: Cell = (r, c)
                height, width = 1, 1

                logger.debug(f"\nstarting new object at {origin}")

                while True:
                    expanded = False

                    # Try expanding rightwards
                    if is_valid_rectangle(origin, height, width + 1, main_color):
                        width += 1
                        expanded = True
                        logger.debug(f"expanded rightwards new dimensions: {origin, height, width}")

                    # Try expanding downwards
                    if is_valid_rectangle(origin, height + 1, width, main_color):
                        height += 1
                        expanded = True
                        logger.debug(f"expanded downwards new dimensions: {origin, height, width}")

                    # Try expanding right-downwards
                    if is_valid_rectangle(origin, height + 1, width + 1, main_color):
                        height += 1
                        width += 1
                        expanded = True
                        logger.debug(f"expanded right-downwards new dimensions: {origin, height, width}")

                    # Try expanding leftwards
                    if is_valid_rectangle((origin[0], origin[1] - 1), height, width, main_color):
                        origin = (origin[0], origin[1] - 1)
                        width += 1
                        expanded = True
                        logger.debug(f"expanded leftwards new dimensions: {origin, height, width}")

                    # Try expanding upwards
                    if is_valid_rectangle((origin[0] - 1, origin[1]), height, width, main_color):
                        origin = (origin[0] - 1, origin[1])
                        height += 1
                        expanded = True
                        logger.debug(f"expanded upwards new dimensions: {origin, height, width}")

                    # If no further expansion is possible, break the loop
                    if not expanded:
                        break

                # Once the largest rectangle is found, create the grid data for the object
                object_grid_data = [
                    [data[r][c] for c in range(origin[1], origin[1] + width)]
                    for r in range(origin[0], origin[0] + height)
                ]
                current_object = Object(
                    origin, object_grid_data)
                objects.append(current_object)

    return objects
