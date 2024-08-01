from grid import Grid


def transform(input_grid: Grid) -> Grid:
    def map_func(x: int, y: int) -> Grid:
        color = input_grid.raw[x][y]
        assert isinstance(color, int)
        if color == 0:
            return Grid.empty(input_grid.size())
        else:
            return input_grid.copy()
    return input_grid.map(map_func)


def test_example():
    input_grid = Grid([
        [0, 7, 7],
        [7, 7, 7],
        [0, 7, 7]
    ])

    output_grid = transform(input_grid)
    input_grid.display(title="Example 1:", output=output_grid)
