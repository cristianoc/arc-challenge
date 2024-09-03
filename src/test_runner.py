import os
from typing import Callable
from objects import display, Object
import load_data
import numpy as np

# Get DISPLAY from environment variable or default to True if not set
DISPLAY = os.getenv("DISPLAY", "True").lower() in ["true", "1", "yes"]


def puzzle(name: str, transform: Callable[[Object], Object]) -> None:
    task = load_data.training_data[name]
    train_set = task["train"]
    test_set = task["test"]
    for i, example in enumerate(train_set):
        input = example[0]
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Example {i+1}:", input=input, output=output)
    for i, example in enumerate(test_set):
        input = example[0]
        correct_grid = example[1]
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Test {i+1}:", input=input, output=output)
        if output != correct_grid:
            display(
                title=f"Correct Output {i+1}:",
                input=output,
                output=correct_grid,
            )
            assert False, f"Test {i+1} failed"
