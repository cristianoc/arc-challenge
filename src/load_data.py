import os
import json
from typing import List, Dict

from grid_data import GridData

Example = Dict[str, GridData]  # {input, output} -> grid data
Task = Dict[str, List[Example]]  # {train, test} -> examples
Tasks = Dict[str, Task]  # xxxx.json -> task


def load_arc_data(directory: str) -> Tasks:
    tasks: Tasks = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                task: Task = json.load(file)
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


def iter_tasks(tasks: Tasks):
    for task_name, task in tasks.items():
        yield task_name, task


# Access train and test sets for the first task in the training data
training_1 = training_data.popitem()
tr_task_1: Task = training_1[1]
eval_1 = evaluation_data.popitem()
ev_task_1: Task = eval_1[1]

train_set = tr_task_1["train"]
test_set = tr_task_1["test"]


# logger.debug(f"Loaded {len(training_data)} training tasks")
# logger.debug(f"Loaded {len(evaluation_data)} evaluation tasks")
