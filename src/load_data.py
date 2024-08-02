import os
import json
from typing import List, Dict

from grid_data import GridData

Example = Dict[str, GridData]
Task = Dict[str, List[Example]]
Tasks = Dict[str, Task]


def load_arc_data(directory: str) -> Tasks:
    tasks: Tasks = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                task: Task = json.load(file)
                tasks[filename] = task
    return tasks


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the training and evaluation datasets
training_dataset_path = os.path.join(script_dir, '../data/training/')
evaluation_dataset_path = os.path.join(script_dir, '../data/evaluation/')

# Load training and evaluation datasets
training_data = load_arc_data(training_dataset_path)
evaluation_data = load_arc_data(evaluation_dataset_path)

# Access train and test sets for the first task in the training data
first_training = training_data.popitem()
first_training_task: Task = first_training[1]
first_evaluation = evaluation_data.popitem()
first_evaluation_task: Task = first_evaluation[1]

train_set = first_training_task['train']
test_set = first_training_task['test']


# Print the number of tasks loaded and an example from each dataset
print(f"Loaded {len(training_data)} training tasks")
print(f"Loaded {len(evaluation_data)} evaluation tasks")
