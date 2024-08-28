class GridTask:
    def __init__(self, examples, test_case):
        self.examples = examples
        self.test_case = test_case
        self.specs = [
            self.identity_spec,
            self.invert_colors_spec,
            self.vertical_flip_spec,
            self.horizontal_flip_spec
        ]

    def identity_spec(self, grid):
        return grid

    def invert_colors_spec(self, grid):
        return [[1 - cell for cell in row] for row in grid]

    def vertical_flip_spec(self, grid):
        return grid[::-1]

    def horizontal_flip_spec(self, grid):
        return [row[::-1] for row in grid]

    def is_spec_solution(self, spec):
        for (input_grid, expected_output) in self.examples:
            if spec(input_grid) != expected_output:
                return False
        return True

    @staticmethod
    def compare_specs(spec1, spec2):
        # Define an arbitrary complexity order
        spec_order = {
            'identity_spec': 0,
            'vertical_flip_spec': 1,
            'horizontal_flip_spec': 1,
            'invert_colors_spec': 2
        }

        spec1_name = spec1.__name__
        spec2_name = spec2.__name__

        if spec_order[spec1_name] < spec_order[spec2_name]:
            return spec1
        elif spec_order[spec1_name] > spec_order[spec2_name]:
            return spec2
        else:
            return None  # They are equally minimal


    def find_correct_spec(self):
        solutions = []
        for spec in self.specs:
            if self.is_spec_solution(spec):
                solutions.append(spec)

        # Print solutions
        print("Solutions:")
        for i, spec in enumerate(solutions):
            print(f"Solution {i+1}:")
            print(spec.__name__)
            print()

        # Only keep minimal solutions
        minimal_solutions = []
        for spec1 in solutions:
            is_minimal = True
            for spec2 in solutions:
                if spec1 != spec2 and self.compare_specs(spec1, spec2) == spec2:
                    is_minimal = False
                    break
            if is_minimal:
                minimal_solutions.append(spec1)
        
        print("Minimal Solutions:")
        for i, spec in enumerate(minimal_solutions):
            print(f"Minimal Solution {i+1}:")
            print(spec.__name__)
            print()

        # Check if there is a unique minimal spec
        if len(minimal_solutions) == 1:
            return minimal_solutions[0]
        else:
            return None

    def test_solution(self):
        correct_spec = self.find_correct_spec()
        if correct_spec is not None:
            return correct_spec(self.test_case)
        else:
            return "Task is not well-defined"

# Function to print grids in a readable format


def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))

# Example of a well-defined task


examples = [
    ([[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    ([[1, 1, 0], [0, 0, 1], [1, 1, 0]], [[0, 0, 1], [1, 1, 0], [0, 0, 1]])
]

test_case = [[0, 0, 1], [1, 1, 0], [0, 0, 1]]

task = GridTask(examples, test_case)

# Print examples
print("Examples for well-defined task:")
for i, (input_grid, output_grid) in enumerate(examples):
    print(f"Example {i+1} - Input:")
    print_grid(input_grid)
    print("Expected Output:")
    print_grid(output_grid)
    print()

# Print the test case input
print("Test Case Input for well-defined task:")
print_grid(test_case)
print()

output = task.test_solution()
print("Output for well-defined task:")
if isinstance(output, list):
    print_grid(output)
else:
    print(output)
print()

# Example of a task that is not well-defined

examples_not_well_defined = [
    ([[0, 1], [1, 0]], [[1, 0], [0, 1]])
]

test_case_not_well_defined = [[1, 0], [0, 0]]

task_not_well_defined = GridTask(
    examples_not_well_defined, test_case_not_well_defined)

# Print examples
print("Examples for not well-defined task:")
for i, (input_grid, output_grid) in enumerate(examples_not_well_defined):
    print(f"Example {i+1} - Input:")
    print_grid(input_grid)
    print("Expected Output:")
    print_grid(output_grid)
    print()

# Print the test case input
print("Test Case Input for not well-defined task:")
print_grid(test_case_not_well_defined)
print()

output_not_well_defined = task_not_well_defined.test_solution()
print("Output for not well-defined task:", output_not_well_defined)
