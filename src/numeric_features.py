from typing import Dict, List, Tuple
from grid import Grid
from rule_based_selector import Features


def detect_numeric_features(grid: Grid) -> Features:
    height, width = grid.size
    num_colors = len(grid.get_colors())
    features = {"grid_height": height, "grid_width": width, "num_colors": num_colors}
    return features


def pretty_print_numeric_features(prediction: Tuple[Dict[str, int], int]) -> str:
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
        elif value > 1:
            res.append(f"{value} * in.{feature}")

    result = " + ".join(res)

    if bias != 0:
        if result:
            result += f" + {bias}"
        else:
            result = str(bias)

    return result
