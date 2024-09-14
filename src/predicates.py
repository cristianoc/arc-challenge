import numpy as np
from objects import Object
from dataclasses import dataclass

@dataclass
class CardinalityInRowPredicate:
    value: int
    count: int

    def __str__(self):
        return f"CardinalityInRow({self.value}) == {self.count}"

@dataclass
class CardinalityInColumnPredicate:
    value: int
    count: int

    def __str__(self):
        return f"CardinalityInColumn({self.value}) == {self.count}"


def determine_global_predicates(grid: Object):
    """
    This function determines which row and column predicates hold globally in the grid.
    A predicate holds if each value appears exactly k times in every row or every column.

    The function returns two lists:
    - One for row predicates (CardinalityInRowPredicate objects)
    - One for column predicates (CardinalityInColumnPredicate objects)
    """
    (width, height) = grid.size
    row_predicates = []
    column_predicates = []

    # Get the unique values in the grid
    unique_values = np.unique(grid._data)

    # Check row predicates
    for value in unique_values:
        counts_in_rows = [np.sum(grid._data[row, :] == value) for row in range(height)]
        if np.all(counts_in_rows == counts_in_rows[0]):  # If same count in every row
            row_predicates.append(CardinalityInRowPredicate(value, counts_in_rows[0]))

    # Check column predicates
    for value in unique_values:
        counts_in_columns = [np.sum(grid._data[:, col] == value) for col in range(width)]
        if np.all(
            counts_in_columns == counts_in_columns[0]
        ):  # If same count in every column
            column_predicates.append(CardinalityInColumnPredicate(value, counts_in_columns[0]))

    return row_predicates, column_predicates


def print_predicates_better_notation(row_predicates, column_predicates):
    """
    This function prints the predicates in their improved syntactic form for both rows and columns.
    """
    if row_predicates:
        for predicate in row_predicates:
            print(predicate)
    else:
        print("No row predicates hold.")

    if column_predicates:
        for predicate in column_predicates:
            print(predicate)
    else:
        print("No column predicates hold.")


# Example filled grid
grid_filled_non_trivial = Object(
    np.array(
        [
            [1, 2, 2, 3],
            [1, 2, 2, 3],
            [1, 2, 2, 3],
            [1, 2, 2, 3],
        ]
    )
)

print(f"Original grid: {grid_filled_non_trivial}")

# Determine the global row and column predicates that hold
global_row_predicates_non_trivial, global_column_predicates_non_trivial = (
    determine_global_predicates(grid_filled_non_trivial)
)

# Print the predicates with improved notation
print_predicates_better_notation(
    global_row_predicates_non_trivial, global_column_predicates_non_trivial
)
