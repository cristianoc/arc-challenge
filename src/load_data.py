import os
import json
from typing import List, Dict

from grid import Grid, Raw

Example = Dict[str, Raw]
Task = Dict[str, List[Example]]


def load_arc_data(directory: str) -> List[Task]:
    tasks: List[Task] = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                task: Task = json.load(file)
                tasks.append(task)
    return tasks


def plot_grid(raw: Raw, title: str = "Grid") -> None:
    Grid(raw).display(title=title)


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the training and evaluation datasets
training_dataset_path = os.path.join(script_dir, '../data/training/')
evaluation_dataset_path = os.path.join(script_dir, '../data/evaluation/')

# Load training and evaluation datasets
training_data: List[Task] = load_arc_data(training_dataset_path)
evaluation_data: List[Task] = load_arc_data(evaluation_dataset_path)

# Print the number of tasks loaded and an example from each dataset
print(f"Loaded {len(training_data)} training tasks")
print("First training task:", json.dumps(training_data[0], indent=2))

print(f"\nLoaded {len(evaluation_data)} evaluation tasks")
print("First evaluation task:", json.dumps(evaluation_data[0], indent=2))

# Access train and test sets for the first task in the training data
first_training_task = training_data[0]
train_set = first_training_task['train']
test_set = first_training_task['test']

print("\nFirst training task train set:")
for example in train_set:
    print("Input:\n", example['input'])
    print("Output:\n", example['output'])
    plot_grid(example['input'], title="Train Input")
    plot_grid(example['output'], title="Train Output")

print("\nFirst training task test set:")
for example in test_set:
    print("Input:\n", example['input'])
    plot_grid(example['input'], title="Test Input")
    if 'output' in example:
        print("Output:\n", example['output'])
        plot_grid(example['output'], title="Test Output")
