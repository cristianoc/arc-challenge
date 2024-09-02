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
        input = Object(np.array(example[0]))
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Example {i+1}:", input=input.datax, output=output.datax)
    for i, example in enumerate(test_set):
        input = Object(np.array(example[0]))
        correct_grid = Object(np.array(example[1]))
        output = transform(input)
        if DISPLAY:
            display(title=f"Train Test {i+1}:", input=input.datax, output=output.datax)
        if output != correct_grid:
            display(
                title=f"Correct Output {i+1}:",
                input=output.datax,
                output=correct_grid.datax,
            )
            assert False, f"Test {i+1} failed"
