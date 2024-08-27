from typing import List, Tuple

from grid import Grid
from grid_data import Object
from rule_based_selector import Features


num_difficulties = 5


def detect_numeric_features(grid: Grid, relative_difficulty: int) -> Features:
    height, width = grid.size
    colors = grid.get_colors(allow_black=False)
    num_colors = len(colors)
    grid_as_object = Object((0, 0), grid.data)
    num_cells = grid_as_object.num_cells(color=None)

    main_color = grid_as_object.main_color

    objects = grid.detect_objects()
    num_objects_of_main_color = 0
    num_cells_in_largest_object = 0
    num_objects_of_max_size = 0
    if objects:
        largest_object = max(
            objects, key=lambda obj: obj.num_cells(color=None), default=None)
        num_objects_of_main_color = sum(
            1 for obj in objects if obj.main_color == main_color)
        if largest_object:
            num_cells_in_largest_object = largest_object.num_cells(
                color=None)
            num_objects_of_max_size = sum(1 for obj in objects if obj.num_cells(
                color=None) == largest_object.num_cells(color=None))

    features: Features = {
    }
    # relative_difficulty is the difficulty minus the level before using linear programming
    if relative_difficulty >= 1:
        features["grid_height"] = height
        features["grid_width"] = width
    if relative_difficulty >= 2:
        features["num_colors"] = num_colors
        features["num_cells"] = num_cells
    if relative_difficulty >= 3:
        features["num_objects_of_main_color"] = num_objects_of_main_color
    if relative_difficulty >= 4:
        features["num_cells_in_largest_object"] = num_cells_in_largest_object
    if relative_difficulty >= 5:
        features["num_objects_of_max_size"] = num_objects_of_max_size

    assert num_difficulties == 5
    return features


def pretty_print_numeric_features(prediction: Tuple[Features, int]) -> str:
    """
    Pretty prints the numeric features.
    For each feature, write just the name if the value is 1, 
    otherwise 'n * feature' if the value is an integer n.
    Omit if the value is 0.
    Add the bias only if it is not zero.

    Args:
        prediction: A tuple containing a dictionary of features with their 
                    corresponding integer values and a bias value.

    Returns:
        A string representation of the numeric features and bias.
    """
    features, bias = prediction
    res: List[str] = []

    for feature, value in features.items():
        if value == 1:
            res.append(f"in.{feature}")
        elif value != 0:
            res.append(f"{value} * in.{feature}")

    result = " + ".join(res)

    if bias != 0:
        if result:
            result += f" + {bias}"
        else:
            result = str(bias)

    return result
