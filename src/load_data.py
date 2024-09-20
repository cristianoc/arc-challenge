import json
import os
from typing import Any, Dict, Generic, List, Tuple, TypeVar

import numpy as np

from objects import Object

T = TypeVar('T')

Example = Tuple[T, T]  # (input, output)

class Task(Generic[T]):
    train: List[Example[T]]
    test: List[Example[T]]

    def __init__(self, train: List[Example[T]], test: List[Example[T]]):
        self.train = train
        self.test = test

Tasks = Dict[str, Task[Any]]  # xxxx.json -> task


def load_arc_data(directory: str) -> Tasks:
    tasks: Tasks = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                obj: Dict[str, Any] = json.load(file)
                task: Task[Object] = Task(
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
                tasks[filename] = task
    return tasks


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the training and evaluation datasets
training_dataset_path = os.path.join(script_dir, "../data/training/")
evaluation_dataset_path = os.path.join(script_dir, "../data/evaluation/")

# Load training and evaluation datasets
training_data: Tasks = load_arc_data(training_dataset_path)
evaluation_data: Tasks = load_arc_data(evaluation_dataset_path)
