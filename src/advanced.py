from grid import FUSCHIA, ORANGE, RED, YELLOW, Grid


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
    o = ORANGE
    y = YELLOW
    r = RED
    f = FUSCHIA

    input1 = Grid([
        [0, o, o],
        [o, o, o],
        [0, o, o]
    ])
    output1 = transform(input1)
    input1.display(title="Example 1:", output=output1)

    input2 = Grid([
        [y, 0, y],
        [0, 0, 0],
        [0, y, 0]
    ])
    output2 = transform(input2)
    input2.display(title="Example 2:", output=output2)

    input3 = Grid([
        [0, 0, 0],
        [0, 0, r],
        [r, 0, r]
    ])
    output3 = transform(input3)
    input3.display(title="Example 3:", output=output3)

    input4 = Grid([
        [f, f, 0],
        [f, 0, 0],
        [0, f, f]
    ])
    output4 = transform(input4)
    input4.display(title="Example 4:", output=output4)

    input5 = Grid([
        [r, r, r],
        [0, 0, 0],
        [0, r, r]
    ])
    output5 = transform(input5)
    input5.display(title="Example 5:", output=output5)

    test = Grid([
        [0, o, o],
        [o, o, o],
        [0, o, o],
    ])
    output = transform(test)
    test.display(title="Test:", output=output)
