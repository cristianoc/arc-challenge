from grid import Grid


def transform(input_grid: Grid) -> Grid:
    def map_func(i: int, j: int) -> Grid:
        color = input_grid.raw[i][j]
        assert isinstance(color, int)
        if color == 0:
            return Grid.empty(input_grid.shape)
        else:
            return input_grid.Copy()

    # Transform the input grid to the output grid
    output_grid = input_grid.map(map_func)
    return output_grid


# Example usage:
input_grid = Grid([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

output_grid = transform(input_grid)
print(output_grid.raw)
