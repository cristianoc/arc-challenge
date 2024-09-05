from typing import List, Tuple
from objects import Object
from rule_based_selector import Features
from visual_cortex import extract_subgrid, find_colored_objects


num_difficulties = 10


def detect_numeric_features(grid: Object, relative_difficulty: int) -> Features:
    width, height = grid.size
    colors = grid.get_colors(allow_black=False)
    num_colors = len(colors)
    num_cells = grid.num_cells(color=None)

    main_color = grid.main_color()

    objects = grid.detect_objects()
    num_objects_of_main_color = None
    num_cells_in_largest_object = None
    num_objects_of_max_size = None
    max_area_object_height = None
    max_area_object_width = None
    objects_are_vertical = None
    num_objects = None
    if objects:
        num_objects = len(objects)
        largest_object = max(
            objects, key=lambda obj: obj.num_cells(color=None), default=None
        )
        num_objects_of_main_color = sum(
            1 for obj in objects if obj.main_color() == main_color
        )
        if largest_object:
            num_cells_in_largest_object = largest_object.num_cells(color=None)
            num_objects_of_max_size = sum(
                1
                for obj in objects
                if obj.num_cells(color=None) == largest_object.num_cells(color=None)
            )
        max_area_object = max(objects, key=lambda obj: obj.area, default=None)
        if max_area_object:
            max_area_object_height = max_area_object.height
            max_area_object_width = max_area_object.width
        if len(objects) >= 2:

            def is_vertical(obj: Object) -> bool:
                return obj.height >= obj.width

            def is_horizontal(obj: Object) -> bool:
                return obj.width >= obj.height

            all_vertical = all(is_vertical(obj) for obj in objects)
            all_horizontal = all(is_horizontal(obj) for obj in objects)
            if all_vertical and all_horizontal:
                objects_are_vertical = None
            elif all_vertical or all_horizontal:
                objects_are_vertical = all_vertical
            else:
                objects_are_vertical = None
    subgrid = extract_subgrid(grid, color=None)
    subgrid_width = None
    subgrid_height = None
    if subgrid:
        subgrid_height = len(subgrid)
        subgrid_width = len(subgrid[0])
    colored_objects: List[Object] = find_colored_objects(grid)
    colored_object_max_height = None
    colored_object_max_width = None
    if len(colored_objects) >= 2:
        obj: Object | None = max(
            colored_objects, key=lambda o: o.height, default=None
        )
        if obj is not None:
            colored_object_max_height = obj.height
        obj = max(
            colored_objects, key=lambda o: o.width, default=None
        )
        if obj is not None:
            colored_object_max_width = obj.width
    grid_height_squared = height * height
    grid_width_squared = width * width

    features: Features = {}
    # relative_difficulty is the difficulty minus the level before using regularized regression
    if relative_difficulty >= 1:
        features["grid_height"] = height
        features["grid_width"] = width
    if relative_difficulty >= 2:
        features["num_colors"] = num_colors
        features["num_cells"] = num_cells
        if num_objects is not None:
            features["num_objects"] = num_objects
    if relative_difficulty >= 3:
        if num_objects_of_main_color is not None:
            features["num_objects_of_main_color"] = num_objects_of_main_color
    if relative_difficulty >= 4:
        if num_cells_in_largest_object is not None:
            features["num_cells_in_largest_object"] = num_cells_in_largest_object
    if relative_difficulty >= 5:
        if num_objects_of_max_size is not None:
            features["num_objects_of_max_size"] = num_objects_of_max_size
    if relative_difficulty >= 6:
        if max_area_object_height is not None:
            features["max_area_object_height"] = max_area_object_height
        if max_area_object_width is not None:
            features["max_area_object_width"] = max_area_object_width
    if relative_difficulty >= 7:
        if subgrid_height is not None:
            features["subgrid_height"] = subgrid_height
        if subgrid_width is not None:
            features["subgrid_width"] = subgrid_width
    if relative_difficulty >= 8:
        if colored_object_max_height is not None:
            features["colored_object_max_height"] = colored_object_max_height
        if colored_object_max_width is not None:
            features["colored_object_max_width"] = colored_object_max_width
    if relative_difficulty >= 9:
        features["grid_height_squared"] = grid_height_squared
        features["grid_width_squared"] = grid_width_squared
    if relative_difficulty >= 10:
        if objects_are_vertical is not None:
            features["objects_are_vertical"] = objects_are_vertical

    assert num_difficulties == 10
    return features


Solution = Tuple[Features, int]
BooleanSolution = Tuple[str, Solution, Solution]


def pretty_print_solution(prediction: Solution) -> str:
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


def pretty_print_boolean_solution(prediction: BooleanSolution) -> str:
    name, solution_true, solution_false = prediction
    res_true = pretty_print_solution(solution_true)
    res_false = pretty_print_solution(solution_false)
    return f"({res_true} if in.{name} else {res_false})"


def pretty_print_numeric_features(prediction: Solution | BooleanSolution) -> str:
    is_boolean_solution = isinstance(prediction[0], str)
    if is_boolean_solution:
        return pretty_print_boolean_solution(prediction)  # type: ignore
    else:
        return pretty_print_solution(prediction)  # type: ignore
