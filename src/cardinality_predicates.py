import numpy as np
from objects import Object
from dataclasses import dataclass
from typing import Union, List, Optional
from grid_types import BLUE, BLACK, GREEN, YELLOW, RED
from objects import display


@dataclass(frozen=True)
class CardinalityInRowPredicate:
    value: int
    count: int

    def __str__(self):
        return f"CardinalityInRow({self.value}) == {self.count}"


@dataclass(frozen=True)
class CardinalityInColumnPredicate:
    value: int
    count: int

    def __str__(self):
        return f"CardinalityInColumn({self.value}) == {self.count}"


CardinalityPredicate = Union[CardinalityInRowPredicate, CardinalityInColumnPredicate]


def fill_grid_based_on_predicate(
    grid: Object,
    predicate: CardinalityPredicate,
    unknown_value: int,
) -> bool:
    """
    This function fills in the blanks in the grid to satisfy a given cardinality predicate.
    The predicate specifies that the value must appear exactly `count` times
    in every row (for CardinalityInRowPredicate) or in every column (for CardinalityInColumnPredicate).

    Blank cells are denoted by the unknown_value, and the function modifies the grid in-place.
    Returns True if any changes were made to the grid, otherwise False.
    """
    (width, height) = grid.size
    is_row_predicate = isinstance(predicate, CardinalityInRowPredicate)
    color = predicate.value
    occurrences_required = predicate.count
    changes_made = False

    if is_row_predicate:
        # Iterate over each row
        for y in range(height):
            current_row = grid._data[y, :]
            value_count = np.sum(current_row == color)
            blanks = np.sum(current_row == unknown_value)

            remaining_occurrences = occurrences_required - value_count
            if remaining_occurrences > 0 and remaining_occurrences == blanks:
                for blank in range(width):
                    if grid[blank, y] == unknown_value:
                        grid[blank, y] = color
                        changes_made = True
    else:
        # Iterate over each column
        for x in range(width):
            current_col = grid._data[:, x]
            value_count = np.sum(current_col == color)
            blanks = np.sum(current_col == unknown_value)

            remaining_occurrences = occurrences_required - value_count
            if remaining_occurrences > 0 and remaining_occurrences == blanks:
                for blank in range(height):
                    if grid[x, blank] == unknown_value:
                        grid[x, blank] = color
                        changes_made = True

    return changes_made


def fill_grid_until_stable(
    grid: Object,
    predicates: list[Union[CardinalityInRowPredicate, CardinalityInColumnPredicate]],
    unknown_value: int,
):
    """
    This function repeatedly fills the grid based on the predicates until no further changes occur.
    """
    while True:
        changes_made = False
        for predicate in predicates:
            if fill_grid_based_on_predicate(grid, predicate, unknown_value):
                changes_made = True
        if not changes_made:
            break
    return grid


def find_cardinality_predicates(grid: Object) -> List[CardinalityPredicate]:
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
        counts_in_columns = [
            np.sum(grid._data[:, col] == value) for col in range(width)
        ]
        if np.all(
            counts_in_columns == counts_in_columns[0]
        ):  # If same count in every column
            column_predicates.append(
                CardinalityInColumnPredicate(value, counts_in_columns[0])
            )

    return row_predicates + column_predicates


def predicate_intersection(
    p1: CardinalityPredicate, p2: CardinalityPredicate
) -> Optional[CardinalityPredicate]:
    if isinstance(p1, type(p2)) and p1.value == p2.value and p1.count == p2.count:
        return p1
    else:
        return None


def predicates_intersection(
    p1: List[CardinalityPredicate], p2: List[CardinalityPredicate]
) -> List[CardinalityPredicate]:
    result = []

    # Create a dictionary with both the type and value as the key for predicates in p2
    p2_predicates_by_type_value = {(type(pred), pred.value): pred for pred in p2}

    # Iterate through p1 and find matching predicates in p2
    for pred1 in p1:
        key = (type(pred1), pred1.value)
        pred2 = p2_predicates_by_type_value.get(key)
        if pred2 is not None:
            intersection = predicate_intersection(pred1, pred2)
            if intersection is not None:
                result.append(intersection)

    return result


def test_fill_grid_based_on_predicate():
    # Example usage
    grid_with_blanks = Object(
        np.array(
            [
                [1, 2, 0, 3],
                [1, 0, 2, 0],
                [0, 2, 2, 0],
                [1, 2, 0, 3],
            ]
        )
    )

    unknown_value = 0
    row_predicate = CardinalityInRowPredicate(value=2, count=2)
    column_predicate = CardinalityInColumnPredicate(value=1, count=3)

    print("Original Grid:\n", grid_with_blanks)

    # Fill the grid based on the row predicate
    filled_grid_row = fill_grid_based_on_predicate(
        grid_with_blanks.copy(), row_predicate, unknown_value
    )
    print("Filled Grid (Row Predicate):\n", filled_grid_row)

    # Fill the grid based on the column predicate
    filled_grid_column = fill_grid_based_on_predicate(
        grid_with_blanks.copy(), column_predicate, unknown_value
    )
    print("Filled Grid (Column Predicate):\n", filled_grid_column)


def test_sudoku_example():
    # Define color variables
    b = BLUE
    g = GREEN
    y = YELLOW
    r = RED

    print("\nSudoku Example:")
    # double sudoku example: 2 colors in every row, but one in every column
    initial_grid = Object(
        np.array(
            [
                [b, 0, g, 0, 0, 0, r, b],
                [0, 0, r, b, r, b, y, g],
                [r, b, y, 0, b, r, 0, y],
                [0, g, 0, r, g, 0, 0, r],
            ]
        )
    )

    unknown_value = 0
    sudoku_grid = initial_grid.copy()

    all_predicates = []
    for value in [b, g, y, r]:  # Only non-black colors
        row_predicate = CardinalityInRowPredicate(value=value, count=2)
        col_predicate = CardinalityInColumnPredicate(value=value, count=1)
        all_predicates.append(row_predicate)
        all_predicates.append(col_predicate)  # Add column predicates as well

    # Apply all predicates until the grid is stable
    fill_grid_until_stable(sudoku_grid, all_predicates, unknown_value)

    print("Filled Sudoku Grid:\n", sudoku_grid)
    if False:
        display(initial_grid, sudoku_grid, title="Filled Sudoku Grid")
    # check that there are no blanks left
    assert np.sum(sudoku_grid._data == unknown_value) == 0

    # Determine the global row and column predicates that hold in the filled grid
    detected_predicates = find_cardinality_predicates(sudoku_grid)

    # Print the detected predicates
    print("\nDetected Predicates:")
    for p in detected_predicates:
        print(p)

    # Verify that the detected predicates match the initial predicates
    assert set(all_predicates) == set(
        detected_predicates
    ), "Detected predicates do not match the initial predicates."


def test_determine_global_predicates():
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
    predicates = find_cardinality_predicates(grid_filled_non_trivial)

    # Print the predicates with improved notation
    for p in predicates:
        print(p)


if __name__ == "__main__":
    test_fill_grid_based_on_predicate()
    test_sudoku_example()
    test_determine_global_predicates()
