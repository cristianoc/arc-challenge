from typing import Callable, List, Optional, Set, Tuple, TypedDict

from color_features import detect_color_features
import numeric_features
from visual_cortex import Frame, find_largest_frame, find_smallest_frame, is_frame_part_of_lattice
from grid import Grid
from grid_data import BLACK, GREY, GridData, Object, display, display_multiple, logger
from load_data import Example, Task, Tasks, iter_tasks, training_data, evaluation_data
from numeric_features import detect_numeric_features, pretty_print_numeric_features
from rule_based_selector import DecisionRule, Features, select_object_minimal
from shape_features import detect_shape_features
from solve_integer_program import find_weights_and_bias
from symmetry_features import detect_symmetry_features

Size = Tuple[int, int]
ExampleGrids = List[Tuple[Grid, Grid]]

# returns the index of the object to pick
ObjectPicker = Callable[[List[Object]], int]


class Config:
    find_xform = True
    find_matched_objects = True
    predict_size_using_linear_programming = True
    try_remove_main_color = True
    difficulty = 1000
    task_name = None
    # task_name = "50cb2852.json"
    find_xform_color = True
    display_not_found = False


def output_size_is_input_size(grids: ExampleGrids, grid: Grid, task_name: str):
    return grid.size


def output_size_is_constant(grids: ExampleGrids, grid: Grid, task_name: str):
    return grids[0][1].size


def output_size_is_size_of_largest_nonblack_object(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    objects = grid.detect_objects()
    if not objects:
        return (0, 0)
    largest_object = max(objects, key=lambda obj: obj.num_cells(color=None))
    return largest_object.size


def output_size_is_size_of_object_inside_largest_frame(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    largest_frame = find_largest_frame(grid.data, None)
    if largest_frame:
        (top, left, bottom, right) = largest_frame
        width = right - left + 1
        height = bottom - top + 1
        if width >= 2 and height >= 2:
            logger.debug(f"Largest frame found: {largest_frame} height:{height} width:{width}")
        height_without_frame = height - 2
        width_without_frame = width - 2
        return (height_without_frame, width_without_frame)
    return (0, 0)


def output_size_is_size_of_largest_block_object(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    objects = grid.detect_objects(allow_black=True)
    # exclude full grid size
    objects = [obj for obj in objects if obj.size != grid.size and obj.is_block()]
    if not objects:
        return None
    largest_object = max(objects, key=lambda obj: obj.size[0] * obj.size[1])
    return largest_object.size


def output_size_is_size_of_largest_nonblack_block_object(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    objects = grid.detect_objects(allow_black=False)
    # exclude full grid size and objects smaller than 2x2
    objects = [obj for obj in objects if obj.size != grid.size and obj.size[0] >= 2 and obj.size[1] >= 2 and obj.is_block()]
    if not objects:
        return None
    largest_object = max(objects, key=lambda obj: obj.size[0] * obj.size[1])
    return largest_object.size


def output_size_is_size_of_largest_object_with_flexible_contours(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    objects = grid.detect_objects(allow_black=True)
    # exclude full grid size
    objects = [obj for obj in objects if obj.size != grid.size]
    if not objects:
        return None
    largest_object = max(objects, key=lambda obj: obj.size[0] * obj.size[1])
    largest_frame = find_largest_frame(largest_object.data, None)
    if largest_frame:
        # handle case where the largest object has a few extra cells around it
        # so we need to consider the frame inside
        (top, left, bottom, right) = largest_frame
        width = right - left + 1
        height = bottom - top + 1
        return (height, width)
    return largest_object.size


def output_size_is_size_of_repeating_subgrid_forming_a_lattice(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    def find_lattice(grid: Grid) -> Optional[Frame]:
        largest_frame = find_largest_frame(grid.data, None)
        logger.debug(f"largest_frame:{largest_frame}")
        if largest_frame is None:
            return None
        (top, left, bottom, right) = largest_frame
        width = right - left + 1
        height = bottom - top + 1
        logger.debug(f"Largest frame found: {largest_frame} height:{height} width:{width}")
        foreground = grid.data[top][left]
        # find minimal frame inside and see if it forms a lattice
        is_lattice = is_frame_part_of_lattice(grid.data, largest_frame, foreground)
        logger.debug(f"is_lattice:{is_lattice} foreground:{foreground}")
        smallest_frame = find_smallest_frame(grid.data, foreground, min_size=(3, 3))
        if smallest_frame is None:
            return None
        is_lattice = is_frame_part_of_lattice(grid.data, smallest_frame, foreground)
        logger.debug(f"smallest_frame:{smallest_frame} is_lattice:{is_lattice} foreground:{foreground}")
        if is_lattice:
            return smallest_frame
        else:
            return None

    lattice = find_lattice(grid)
    if lattice is None:
        return None
    (top, left, bottom, right) = lattice
    repeating_obj_height = bottom - top
    repeating_obj_width = right - left
    repeating_obj_height -= 1  # remove the frame
    repeating_obj_width -= 1  # remove the frame
    num_repeating_obj_rows = grid.height // repeating_obj_height
    num_repeating_obj_cols = grid.width // repeating_obj_width

    # check that the size is correct when accounting for the frame
    frame_part_in_rows = num_repeating_obj_cols
    frame_part_in_cols = num_repeating_obj_rows
    logger.debug(f"num_repeating_obj_rows:{num_repeating_obj_rows} num_repeating_obj_cols:{num_repeating_obj_cols}")

    # check that the size is correct when accounting for the frame
    # where -1 is to account for the overlap of the frames
    expected_height = num_repeating_obj_rows * repeating_obj_height + frame_part_in_rows - 1
    logger.debug(f"grid.height:{grid.height} expected height:{expected_height}")
    if grid.height != expected_height:
        return None
    expected_width = num_repeating_obj_cols * repeating_obj_width + frame_part_in_cols - 1
    logger.debug(f"grid.width:{grid.width} expected width:{expected_width}")
    if grid.width != expected_width:
        return None

    return (num_repeating_obj_rows, num_repeating_obj_cols)


SizeXform = Callable[[ExampleGrids, Grid, str], Optional[Size]]
class XformEntry(TypedDict):
    function: SizeXform
    difficulty: int

xforms: List[XformEntry] = [
    {"function": output_size_is_input_size, "difficulty": 1},  # Level 1: Very Simple
    {"function": output_size_is_constant, "difficulty": 2},  # Level 2: Simple with External Dependency
    {"function": output_size_is_size_of_object_inside_largest_frame, "difficulty": 4},  # Level 4: Complex
    {"function": output_size_is_size_of_largest_block_object, "difficulty": 3},  # Level 3: Moderate
    {"function": output_size_is_size_of_largest_nonblack_block_object, "difficulty": 3},  # Level 3: Moderate
    {"function": output_size_is_size_of_largest_nonblack_object, "difficulty": 3},  # Level 3: Moderate
    {"function": output_size_is_size_of_largest_object_with_flexible_contours, "difficulty": 4},  # Level 4: Complex
    {"function": output_size_is_size_of_repeating_subgrid_forming_a_lattice, "difficulty": 4}  # Level 4: Complex
]

ColorXform = Callable[[ExampleGrids, Grid, str], Optional[Set[int]]]  # List[int] for color indexes
class ColorXformEntry(TypedDict):
    function: ColorXform
    difficulty: int


def output_colors_are_input_colors(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return set(grid.get_colors())

def output_colors_are_constant(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return set(grids[0][1].get_colors())

def output_colors_are_input_colors_minus_num_colors(grids: ExampleGrids, grid: Grid, task_name: str, num: int) -> Optional[Set[int]]:
    # Check in grids if there are num colors that are always removed from the input to the output
    # If found, remove them from the grid colors
    candidate_colors: Optional[Set[int]] = None
    for (input, output) in grids:
        input_colors = set(input.get_colors())
        output_colors = set(output.get_colors())
        removed_colors = input_colors - output_colors
        if len(removed_colors) != num:
            return None
        if candidate_colors is None:
            candidate_colors = removed_colors
            continue
        if candidate_colors != removed_colors:
            return None
    if candidate_colors is None:
        return None
    return set(grid.get_colors()) - candidate_colors

def output_colors_are_input_colors_minus_one_color(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return output_colors_are_input_colors_minus_num_colors(grids, grid, task_name, 1)

def output_colors_are_input_colors_minus_two_colors(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return output_colors_are_input_colors_minus_num_colors(grids, grid, task_name, 2)

def output_colors_are_inout_colors_minus_one_color_plus_another_color(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    # Check in grids if there is one color that is always removed from the input to the output
    # and another color that is always added
    # If found, remove the removed color and add the added color to the grid colors
    candidate_removed_color: Optional[int] = None
    candidate_added_color: Optional[int] = None
    for (input, output) in grids:
        input_colors = set(input.get_colors())
        output_colors = set(output.get_colors())
        removed_colors = input_colors - output_colors
        added_colors = output_colors - input_colors
        if len(removed_colors) != 1 or len(added_colors) != 1:
            return None
        removed_color = next(iter(removed_colors))
        added_color = next(iter(added_colors))
        if candidate_removed_color is None:
            candidate_removed_color = removed_color
        if candidate_added_color is None:
            candidate_added_color = added_color
        if candidate_removed_color != removed_color or candidate_added_color != added_color:
            return None
    if candidate_removed_color is None or candidate_added_color is None:
        return None
    return (set(grid.get_colors()) - {candidate_removed_color}) | {candidate_added_color}

def output_colors_are_input_colors_plus_num_colors(grids: ExampleGrids, grid: Grid, task_name: str, num: int) -> Optional[Set[int]]:
    # Check in grids if there are num colors that are always added from the input to the output
    # If found, add them to the grid colors
    candidate_colors: Optional[Set[int]] = None
    for (input, output) in grids:
        input_colors = set(input.get_colors())
        output_colors = set(output.get_colors())
        added_colors = output_colors - input_colors
        if len(added_colors) != num:
            return None
        if candidate_colors is None:
            candidate_colors = added_colors
            continue
        if candidate_colors != added_colors:
            return None
    if candidate_colors is None:
        return None
    return set(grid.get_colors()) | candidate_colors


def output_colors_are_input_colors_plus_one_color(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return output_colors_are_input_colors_plus_num_colors(grids, grid, task_name, 1)

def output_colors_are_input_colors_plus_two_colors(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return output_colors_are_input_colors_plus_num_colors(grids, grid, task_name, 2)

def output_colors_are_input_colors_plus_black(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return set(grid.get_colors(allow_black=True)) | {BLACK}

def output_colors_are_input_colors_minus_black_grey(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Set[int]]:
    return set(grid.get_colors(allow_black=True)) - {BLACK, GREY}

xforms_color: List[ColorXformEntry] = [
        {"function": output_colors_are_input_colors, "difficulty": 1},
        {"function": output_colors_are_input_colors_plus_black, "difficulty": 1},
        {"function": output_colors_are_input_colors_minus_black_grey, "difficulty": 1},
        {"function": output_colors_are_constant, "difficulty": 2},
        {"function": output_colors_are_input_colors_minus_one_color, "difficulty": 3},
        {"function": output_colors_are_input_colors_minus_two_colors, "difficulty": 3},
        {"function": output_colors_are_input_colors_plus_one_color, "difficulty": 3},
        {"function": output_colors_are_input_colors_plus_two_colors, "difficulty": 3},
        {"function": output_colors_are_inout_colors_minus_one_color_plus_another_color, "difficulty": 4},
    ]

def check_xform_on_examples(xform: SizeXform, examples: List[Example], task_name: str, task_type: str) -> bool:
    grids = [(Grid(example['input']), Grid(example['output'])) for example in examples]
    logger.debug(f"Checking xform {xform.__name__} {task_type}")
    for i, example in enumerate(examples):
        logger.debug(f"  Example {i+1}/{len(examples)}")
        input = Grid(example['input'])
        output = Grid(example['output'])
        new_output_size = xform(grids, input, task_name)
        if new_output_size != output.size:
            logger.debug(f"  Example {i+1} failed")
            return False
    return True

def check_xform_on_examples_color(xform: ColorXform, examples: List[Example], task_name: str, task_type: str) -> bool:
    grids = [(Grid(example['input']), Grid(example['output'])) for example in examples]
    logger.debug(f"Checking xform {xform.__name__} {task_type}")
    for i, example in enumerate(examples):
        logger.debug(f"  Example {i+1}/{len(examples)}")
        input = Grid(example['input'])
        output = Grid(example['output'])
        output_colors = set(output.get_colors())
        logger.debug(f"output_colors:{output_colors}")
        new_output_colors = xform(grids, input, task_name)
        logger.debug(f"new_output_colors:{new_output_colors}")
        if new_output_colors is None:
            logger.debug(f"  Example {i+1} failed")
            return False
        if new_output_colors != output_colors:
            logger.debug(f"  Example {i+1} failed")
            return False
    return True

def find_xform(examples: List[Example], task: Task, task_name: str, task_type: str) -> Optional[XformEntry]:
    # check if at least one xform is correct
    correct_xform = None
    for xform in xforms:
        if Config.difficulty < xform["difficulty"]:
            continue
        func = xform["function"]
        logger.debug(f"Checking xform {func.__name__} {task_type}")
        if check_xform_on_examples(func, examples, task_name, task_type):
            if False and xform == output_size_is_constant_times_input_size:
                title = f"{xform.__name__} ({task_name})"
                logger.info(title)
                for i, e in enumerate(examples):
                    display(e['input'], output=e['output'],
                            title=f"Ex{i+1} " + title)
            correct_xform = xform
            logger.info(f"Xform {correct_xform['function'].__name__} is correct for all examples in {task_type}")
            test_examples = [examples for task_type, examples in task.items() if task_type == 'test']
            for i, test_example in enumerate(test_examples):
                if not check_xform_on_examples(correct_xform["function"], test_example, task_name, 'test'):
                    logger.warning(f"Xform {correct_xform['function'].__name__} failed for test example {i}")
                    correct_xform = None
                    break
            if correct_xform:
                break
    return correct_xform


# ObjectMatch is a type alias representing a match between a list of detected input objects
# and the index of the object within that list that is identical to the output object.
#
# The first element of the tuple (List[Object]) contains all the detected input objects,
# while the second element (int) specifies the index of the object in this list that is
# identical to the output object in terms of size and data.
ObjectMatch = Tuple[List[Object], int]


def detect_common_features(matched_objects: List[ObjectMatch], initial_difficulty: int):
    def detect_common_symmetry_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [detect_symmetry_features(obj.data) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Symmetry): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(decision_rule)
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Symmetry)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_color_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [detect_color_features(obj, input_objects) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Color): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(decision_rule)
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Color)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_shape_features(level: int) -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [detect_shape_features(obj, input_objects, level) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Shape): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(decision_rule)
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Shape)")
                common_decision_rule = None
                break
        return common_decision_rule

    common_decision_rule = None
    features_used = None

    # Try detecting common features in the order of shape, color, and symmetry

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 1:
        common_decision_rule = detect_common_shape_features(initial_difficulty+1)
        features_used = "Shape"

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 2:
        common_decision_rule = detect_common_color_features()
        features_used = "Color"

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 3:
        common_decision_rule = detect_common_symmetry_features()
        features_used = "Symmetry"
    assert num_difficulties_matching == 3

    return common_decision_rule, features_used


def find_matched_objects(examples: List[Example], task_type: str) -> Optional[List[ObjectMatch]]:
    """
    Identifies and returns a list of matched input objects that correspond to the output objects
    in the given examples. For each example, it detects candidate objects in the input grid 
    and matches them with the output grid based on size and data. If all examples have a match, 
    the function returns the list of matched objects; otherwise, it returns None.

    Args:
        examples: A list of examples, each containing an input and output grid.
        task_type: A string indicating the type of task (e.g., 'train' or 'test').

    Returns:
        A list of ObjectMatch tuples if matches are found for all examples, otherwise None.
    """

    def candidate_objects_for_matching(input: Grid, output: Grid) -> List[Object]:
        """
        Detects objects in the input grid that are candidates for matching the output grid.
        """
        output_as_object = Object((0, 0), output.data)
        if output_as_object.has_frame():
            # If the output is a frame, detect objects in the input as frames
            logger.debug("  Output is a frame")
        num_colors_output = len(output.get_colors(allow_black=True))
        return input.detect_rectangular_objects(allow_multicolor=num_colors_output > 1)

    def find_matching_input_object(input_objects: List[Object], output: Grid) -> Optional[int]:
        for i, io in enumerate(input_objects):
            if io.size == output.size and io.data == output.data:
                logger.debug(f"  Input object matching output: {io}")
                return i
        return None

    def get_matched_objects(examples: List[Example]) -> Optional[List[ObjectMatch]]:
        matched_objects: List[ObjectMatch] = []

        for example in examples:
            input = Grid(example['input'])
            output = Grid(example['output'])
            logger.info(f"  {task_type} {input.size} -> {output.size}")

            input_objects = candidate_objects_for_matching(input, output)
            matched_object_index = find_matching_input_object(input_objects, output)

            if matched_object_index is not None:
                matched_objects.append((input_objects, matched_object_index))

        return matched_objects if len(matched_objects) == len(examples) else None

    matched_objects = get_matched_objects(examples)
    return matched_objects


def predict_size_using_linear_programming(examples: List[Example], relative_difficulty: int):
    """
    Predicts the output size using linear programming. The function takes a list of input-output
    grid pairs and attempts to determine the output size by solving a linear program that minimizes
    the sum of weights and bias for each feature.
    """
    feature_vectors: List[Features] = []
    target_heights: List[int] = []
    target_widths: List[int] = []

    for example in examples:
        input_grid = Grid(example['input'])
        output_grid = Grid(example['output'])

        input_features = detect_numeric_features(input_grid, relative_difficulty)
        target_height, target_width = output_grid.size

        feature_vectors.append(input_features)
        target_heights.append(target_height)
        target_widths.append(target_width)

    predicted_height = find_weights_and_bias(feature_vectors, target_heights, "height")
    predicted_width = find_weights_and_bias(feature_vectors, target_widths, "width")
    return predicted_height, predicted_width

num_difficulties_xform = max(xform["difficulty"] for xform in xforms)
num_difficulties_matching = 3
num_difficulties_linear_programming = numeric_features.num_difficulties + 1
num_difficulties_total = num_difficulties_xform + num_difficulties_matching + num_difficulties_linear_programming

def process_tasks(tasks: Tasks, set: str):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in iter_tasks(tasks):
        if Config.task_name and task_name != Config.task_name:
            continue
        logger.info(f"\n***Task: {task_name} {set}***")

        for task_type, examples in task.items():
            if task_type not in ['train', 'test']:
                continue
            if task_type == 'test':
                continue

            current_difficulty = 0

            if Config.find_xform:
                correct_xform = find_xform(examples, task, task_name, task_type)
                if correct_xform:
                    logger.info(f"Xform {correct_xform['function'].__name__} is correct for all examples in {task_type} and test")
                    num_correct += 1
                    continue

            current_difficulty += num_difficulties_xform

            if Config.find_matched_objects:
                # Check if the input objects can be matched to the output objects
                logger.debug(f"Checking common features for {task_name} {set}")
                matched_objects = find_matched_objects(examples, task_type)
                if matched_objects:
                    # If the input objects can be matched to the output objects, try to detect common features
                    # to determine the correct object to pick
                    logger.debug(f"XXX Matched {len(matched_objects)}/{len(examples)} {task_name} {set}")
                    common_decision_rule, features_used = detect_common_features(matched_objects, current_difficulty)
                    if common_decision_rule:
                        logger.info(f"Common decision rule ({features_used}): {common_decision_rule}")
                        num_correct += 1
                        continue
                    else:
                        logger.warning(f"Could not find common decision rule for {task_name} {set}")
            current_difficulty += num_difficulties_matching

            def try_linear_programming(exs: List[Example]):
                logger.debug(f"Trying to determine dimensions via LP for {task_name} {set}")
                predicted_height, predicted_width = predict_size_using_linear_programming(exs, relative_difficulty=Config.difficulty - current_difficulty)
                if predicted_height and predicted_width:
                    logger.info(f"Predictions via LP: out.height=={pretty_print_numeric_features(predicted_height)}, out.width=={pretty_print_numeric_features(predicted_width)}")
                return predicted_height, predicted_width

            difficulty_after_linear_programming = current_difficulty + numeric_features.num_difficulties + 1
            if Config.predict_size_using_linear_programming and Config.difficulty >= current_difficulty:
                predicted_height, predicted_width = try_linear_programming(examples)
                if predicted_height and predicted_width:
                    num_correct += 1
                    continue
                if Config.try_remove_main_color and Config.difficulty >= difficulty_after_linear_programming:
                    # try to remove main color and try again
                    examples2: List[Example] = []
                    for example in examples:
                        input_obj = Object((0, 0), example['input'])
                        # change the main color to black
                        example['input'] = input_obj.change_color(input_obj.main_color, BLACK).data
                        # make a copt of example where you change the input
                        example_copy = example.copy()
                        example_copy['input'] = example['input']
                        examples2.append(example_copy)
                    predicted_height, predicted_width = try_linear_programming(examples2)
                    if predicted_height and predicted_width:
                        num_correct += 1
                        continue
            current_difficulty += num_difficulties_linear_programming

            if Config.display_not_found:
                grids: List[Tuple[GridData, Optional[GridData]]] = [
                    (Grid(example['input']).data, Grid(example['output']).data) for example in examples
                ]
                
                display_multiple(grids, title=f"{task_name} {set}")

            # If no valid dimensions could be determined, give up
            logger.warning(f"Could not find correct transformation or determine dimensions via Linear Programming for {task_name} {set} examples")
            num_incorrect += 1

    return num_correct, num_incorrect


def compute_perc_correct(num_correct: int, num_incorrect: int) -> Optional[float]:
    if num_correct + num_incorrect > 0:
        return int(1000 * num_correct / (num_correct + num_incorrect)) / 10
    return None

def predict_sizes():
    num_correct_tr, num_incorrect_tr = process_tasks(training_data, "training_data")
    num_correct_ev, num_incorrect_ev = process_tasks(evaluation_data, "evaluation_data")
    perc_correct_tr = compute_perc_correct(num_correct_tr, num_incorrect_tr)
    perc_correct_ev = compute_perc_correct(num_correct_ev, num_incorrect_ev)

    def log_evaluation_results(set: str, num_correct: int, num_incorrect: int):
        perc_correct = compute_perc_correct(num_correct, num_incorrect)
        if perc_correct is not None:
            logger.error(
                f"{set.capitalize()} data: "
                f"Correct: {num_correct}, Incorrect: {num_incorrect}, Score: {perc_correct}%")

    logger.error("\n***Summary***")
    log_evaluation_results("training", num_correct_tr, num_incorrect_tr)
    log_evaluation_results("evaluation", num_correct_ev, num_incorrect_ev)
    
    # Write summary of results to JSON file
    with open("predict_sizes.json", "w") as f:
        f.write(
            f'{{"training_data":{perc_correct_tr},"evaluation_data":{perc_correct_ev}}}'
        )

def find_xform_color(examples: List[Example], task: Task, task_name: str, task_type: str) -> Optional[ColorXformEntry]:
    # check if at least one xform is correct
    correct_xform = None
    for xform in xforms_color:
        if Config.difficulty < xform["difficulty"]:
            continue
        func = xform["function"]
        logger.debug(f"Checking xform {func.__name__} {task_type}")
        if check_xform_on_examples_color(func, examples, task_name, task_type):
            if False and xform == output_size_is_constant_times_input_size:
                title = f"{xform.__name__} ({task_name})"
                logger.info(title)
                for i, e in enumerate(examples):
                    display(e['input'], output=e['output'],
                            title=f"Ex{i+1} " + title)
            correct_xform = xform
            logger.info(f"Xform {correct_xform['function'].__name__} is correct for all examples in {task_type}")
            test_examples = [examples for task_type, examples in task.items() if task_type == 'test']
            for i, test_example in enumerate(test_examples):
                if not check_xform_on_examples_color(correct_xform["function"], test_example, task_name, 'test'):
                    logger.warning(f"Xform {correct_xform['function'].__name__} failed for test example {i}")
                    correct_xform = None
                    break
            if correct_xform:
                break
    return correct_xform

num_difficulties_xform_color = max(xform["difficulty"] for xform in xforms_color)
num_difficulties_total_color = num_difficulties_xform_color + num_difficulties_matching

def process_tasks_color(tasks: Tasks, set: str):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in iter_tasks(tasks):
        if Config.task_name and task_name != Config.task_name:
            continue
        logger.info(f"\n***Task: {task_name} {set}***")

        for task_type, examples in task.items():
            if task_type not in ['train', 'test']:
                continue
            if task_type == 'test':
                continue

            current_difficulty = 0

            if Config.find_xform_color:
                correct_xform = find_xform_color(examples, task, task_name, task_type)
                if correct_xform:
                    logger.info(f"Xform {correct_xform['function'].__name__} is correct for all examples in {task_type} and test")
                    num_correct += 1
                    continue

            current_difficulty += num_difficulties_xform_color

            if Config.find_matched_objects:
                # Check if the input objects can be matched to the output objects
                logger.debug(f"Checking common features for {task_name} {set}")
                matched_objects = find_matched_objects(examples, task_type)
                if matched_objects:
                    # If the input objects can be matched to the output objects, try to detect common features
                    # to determine the correct object to pick
                    logger.debug(f"XXX Matched {len(matched_objects)}/{len(examples)} {task_name} {set}")
                    common_decision_rule, features_used = detect_common_features(matched_objects, current_difficulty)
                    if common_decision_rule:
                        logger.info(f"Common decision rule ({features_used}): {common_decision_rule}")
                        num_correct += 1
                        continue
                    else:
                        logger.warning(f"Could not find common decision rule for {task_name} {set}")

            current_difficulty += num_difficulties_matching

            num_incorrect += 1
            logger.warning(f"Could not find correct color transformation for {task_name} {set}")
            if Config.display_not_found:
                grids: List[Tuple[GridData, Optional[GridData]]] = [
                    (Grid(example['input']).data, Grid(example['output']).data) for example in examples
                ]
                display_multiple(grids, title=f"{task_name} {set}")

    return num_correct, num_incorrect

def predict_colors():
    num_correct_tr, num_incorrect_tr = process_tasks_color(training_data, "training_data")
    num_correct_ev, num_incorrect_ev = process_tasks_color(evaluation_data, "evaluation_data")
    perc_correct_tr = compute_perc_correct(num_correct_tr, num_incorrect_tr)
    perc_correct_ev = compute_perc_correct(num_correct_ev, num_incorrect_ev)

    def log_evaluation_results(set: str, num_correct: int, num_incorrect: int):
        perc_correct = compute_perc_correct(num_correct, num_incorrect)
        if perc_correct is not None:
            logger.error(
                f"{set.capitalize()} data: "
                f"Correct: {num_correct}, Incorrect: {num_incorrect}, Score: {perc_correct}%")

    logger.error("\n***Summary***")
    log_evaluation_results("training", num_correct_tr, num_incorrect_tr)
    log_evaluation_results("evaluation", num_correct_ev, num_incorrect_ev)
    
    # Write summary of results to JSON file
    with open("predict_colors.json", "w") as f:
        f.write(
            f'{{"training_data":{perc_correct_tr},"evaluation_data":{perc_correct_ev}}}'
        )

if __name__ == "__main__":
    predict_colors()
