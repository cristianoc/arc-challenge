import os
from typing import Callable
from grid import Grid
from grid_data import display
import load_data

# Get DISPLAY from environment variable or default to True if not set
DISPLAY = os.getenv('DISPLAY', 'True').lower() in ['true', '1', 'yes']

def puzzle(name: str, transform: Callable[[Grid], Grid]) -> None:
    task = load_data.training_data[name]
    train_set = task['train']
    test_set = task['test']
    for i, example in enumerate(train_set):
        input = Grid(example['input'])
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Example {i+1}:",
                    input=input.data, output=output.data)
    for i, example in enumerate(test_set):
        input = Grid(example['input'])
        correct_grid = Grid(example['output'])
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Test {i+1}:",
                    input=input.data, output=output.data)
        if output != correct_grid:
            display(title=f"Correct Output {i+1}:",
                input=output.data, output=correct_grid.data)
            assert False, f"Test {i+1} failed"
