from test_runner import puzzle
from grid import Grid

"""
"""



def transform(input: Grid) -> Grid:
    # Detect objects in the grid, assuming each object is a diagonal
    objects = input.detect_objects()
    
    # Ensure there are exactly 3 objects as stated in the problem
    assert len(objects) == 3

    # Diagonal index between 0 and 2
    def diagonal_index(row: int, col:int) -> int:
        return (row + col) % 3

    object_colors = [0 for _ in range(3)]
    for obj in objects:
        object_colors[diagonal_index(obj.origin[0], obj.origin[1])] = obj.color

    # Determine colors of the three detected objects
    object_colors = [obj.color for obj in objects]

    # Create an empty grid with the same dimensions as input
    output_grid = Grid.empty(input.height, input.width)

    for row in range(input.height):
        for col in range(input.width):
            # Calculate the diagonal index of the current cell
            diag = diagonal_index(row, col)
            # Set the color of the cell to the corresponding object color
            print(f"diag: {diag}")
            color = object_colors[diag]
            output_grid.data[row][col] = color

    return output_grid

def test():
    puzzle(name="05269061.json", transform=transform)
