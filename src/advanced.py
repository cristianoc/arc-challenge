from grid_data import FUSCHIA, ORANGE, RED, YELLOW, display
from grid import Grid


def transform(input_grid: Grid) -> Grid:
    def map_func(x: int, y: int) -> Grid:
        color = input_grid.data[x][y]
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
    display(title="Example 1:", input=input1.data, output=output1.data)

    input2 = Grid([
        [y, 0, y],
        [0, 0, 0],
        [0, y, 0]
    ])
    output2 = transform(input2)
    display(title="Example 2:", input=input2.data, output=output2.data)

    input3 = Grid([
        [0, 0, 0],
        [0, 0, r],
        [r, 0, r]
    ])
    output3 = transform(input3)
    display(title="Example 3:", input=input3.data, output=output3.data)

    input4 = Grid([
        [f, f, 0],
        [f, 0, 0],
        [0, f, f]
    ])
    output4 = transform(input4)
    display(title="Example 4:", input=input4.data, output=output4.data)

    input5 = Grid([
        [r, r, r],
        [0, 0, 0],
        [0, r, r]
    ])
    output5 = transform(input5)
    display(title="Example 5:", input=input5.data, output=output5.data)

    test = Grid([
        [0, o, o],
        [o, o, o],
        [0, o, o],
    ])
    output = transform(test)
    display(title="Test:", input=test.data, output=output.data)
