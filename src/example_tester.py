from typing import Callable
from grid import Grid
from grid_data import display
import load_data


def example(name: str, transform: Callable[[Grid], Grid]) -> None:
    task = load_data.training_data[name]
    train_set = task['train']
    test_set = task['test']
    for i, example in enumerate(train_set):
        input_grid = Grid(example['input'])
        output_grid = transform(input_grid)
        display(title=f"Train Example {i+1}:",
                input=input_grid.data, output=output_grid.data)
    for i, example in enumerate(test_set):
        input_grid = Grid(example['input'])
        correct_grid = Grid(example['output'])
        output_grid = transform(input_grid)
        display(title=f"Train Test {i+1}:",
                input=input_grid.data, output=output_grid.data)
        if output_grid != correct_grid:
            display(title=f"Correct Output {i+1}:",
                input=output_grid.data, output=correct_grid.data)
            assert False, f"Test {i+1} failed"
