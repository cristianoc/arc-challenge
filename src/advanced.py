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
        [0, 1, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])

    input_grid.display("Input Grid:")
    output_grid = transform(input_grid)
    output_grid.display("Output Grid:")
