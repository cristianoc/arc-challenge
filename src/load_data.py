import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_arc_data(directory):
    tasks = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                task = json.load(file)
                tasks.append(task)
    return tasks

def plot_grid(grid, title="Grid"):
    grid_array = np.array(grid)
    plt.imshow(grid_array, cmap='tab20')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the training and evaluation datasets
training_dataset_path = os.path.join(script_dir, '../data/training/')
evaluation_dataset_path = os.path.join(script_dir, '../data/evaluation/')

# Load training and evaluation datasets
training_data = load_arc_data(training_dataset_path)
evaluation_data = load_arc_data(evaluation_dataset_path)

# Print the number of tasks loaded and an example from each dataset
print("Loaded {} training tasks".format(len(training_data)))
print("First training task:", json.dumps(training_data[0], indent=2))

print("\nLoaded {} evaluation tasks".format(len(evaluation_data)))
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
