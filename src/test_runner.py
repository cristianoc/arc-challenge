import os
from typing import Callable
from objects import display, Object
import load_data

# Get DISPLAY from environment variable or default to True if not set
DISPLAY = os.getenv("DISPLAY", "True").lower() in ["true", "1", "yes"]


def puzzle(name: str, transform: Callable[[Object], Object]) -> None:
    task = load_data.training_data[name]
    train_set = task["train"]
    test_set = task["test"]
    for i, example in enumerate(train_set):
        input = Object(example[0])
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Example {i+1}:", input=input.data, output=output.data)
    for i, example in enumerate(test_set):
        input = Object(example[0])
        correct_grid = Object(example[1])
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Test {i+1}:", input=input.data, output=output.data)
        if output != correct_grid:
            display(
                title=f"Correct Output {i+1}:",
                input=output.data,
                output=correct_grid.data,
            )
            assert False, f"Test {i+1} failed"
