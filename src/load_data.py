import json
import os
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Iterator

import numpy as np

from bi_types import Examples
from objects import Object


class Task:
    train: Examples[Object, Object]
    test: Examples[Object, Object]

    def __init__(self, train: Examples[Object, Object], test: Examples[Object, Object]):
        self.train = train
        self.test = test


TasksIterator = Iterator[Tuple[str, Task]]  # filename.json -> task


def iter_arc_data(directory: str) -> TasksIterator:
    """Iterates through ARC tasks in the given directory, yielding (filename, task) pairs."""
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                obj: Dict[str, Any] = json.load(file)
                task = Task(
                    train=[
                        (
                            Object(np.array(example["input"])),
                            Object(np.array(example["output"])),
                        )
                        for example in obj["train"]
                    ],
                    test=[
                        (
                            Object(np.array(example["input"])),
                            Object(np.array(example["output"])),
                        )
                        for example in obj["test"]
                    ],
                )
                yield filename, task


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the training and evaluation datasets
training_dataset_path = os.path.join(script_dir, "../data/training/")
evaluation_dataset_path = os.path.join(script_dir, "../data/evaluation/")

# Load training and evaluation datasets
training_data: TasksIterator = iter_arc_data(training_dataset_path)
evaluation_data: TasksIterator = iter_arc_data(evaluation_dataset_path)
