import os
import json
from typing import List, Dict, TypedDict, Tuple, Any

from objects import GridData

Example = Tuple[GridData, GridData]  # (input, output)


class Task(TypedDict):
    train: List[Example]
    test: List[Example]


Tasks = Dict[str, Task]  # xxxx.json -> task


def load_arc_data(directory: str) -> Tasks:
    tasks: Tasks = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                obj: Dict[str, Any] = json.load(file)
                task: Task = {
                    "train": [
                        (example["input"], example["output"])
                        for example in obj["train"]
                    ],
                    "test": [
                        (example["input"], example["output"]) for example in obj["test"]
                    ],
                }
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
