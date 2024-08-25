from typing import List, Tuple

from grid import Grid
from grid_data import Object
from rule_based_selector import Features


num_difficulties = 3

def detect_numeric_features(grid: Grid, initial_difficulty: int) -> Features:
    height, width = grid.size
    colors = grid.get_colors()
    num_colors = len(colors)
    grid_as_object = Object((0, 0), grid.data)
    num_cells = grid_as_object.num_cells(color=None)

    objects = grid.detect_objects()
    main_color = grid_as_object.main_color
    num_objects_of_main_color = sum(
        1 for obj in objects if obj.main_color == main_color)

    features : Features = {
        "grid_height": {"value": height, "difficulty": 1 + initial_difficulty},
        "grid_width": {"value": width, "difficulty": 1 + initial_difficulty},
        "num_colors": {"value": num_colors, "difficulty": 2 + initial_difficulty},
        "num_cells": {"value": num_cells, "difficulty": 2 + initial_difficulty},
        "num_objects_of_main_color": {"value": num_objects_of_main_color, "difficulty": 3 + initial_difficulty},
    }
    assert num_difficulties == 3
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
