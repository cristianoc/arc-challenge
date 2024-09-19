from typing import (
    List,
    Optional,
    Tuple,
)

from color_features import detect_color_features
from logger import logger
from objects import Object, display, display_multiple
from load_data import Example, Task, Tasks, training_data, evaluation_data
from rule_based_selector import DecisionRule, select_object_minimal
from shape_features import detect_shape_features
from symmetry_features import detect_symmetry_features
from symmetry import (
    find_periodic_symmetry_predicates,
    find_non_periodic_symmetry_predicates,
    fill_grid,
    PeriodicGridSymmetry,
    NonPeriodicGridSymmetry,
)
from visual_cortex import find_rectangular_objects, regularity_score
import numpy as np
from dataclasses import dataclass
from grid_normalization import ClockwiseRotation, XReflection, RigidTransformation
from cardinality_predicates import (
    find_cardinality_predicates,
    CardinalityPredicate,
    predicates_intersection,
)
from bi_types import Match, GridAndObjects, XformEntry, Config
from primitives import xform_identity, primitive_to_xform, translate_down_1
from canvas_grid_match import canvas_grid_xform
from inpainting_match import (
    is_inpainting_puzzle,
    inpainting_xform_no_mask,
    inpainting_xform_with_mask,
    mask_from_all_outputs,
    inpainting_xform,
)
from canvas_grid_match import equal_modulo_rigid_transformation
from split_mirrot_match import frame_split_and_mirror_xform
from expansion_match import (
    fractal_expansion,
    stretch_height,
    check_fractal_expansion_sizes,
)
from map_function_match import expansion_xforms, out_objects_are_a_subset



def filter_simple_xforms(task: Task, task_name: str):
    examples = task.train
    tests = task.test
    for example in examples:
        input = example[0]
        output = example[1]
        if (
            input.width > Config.max_size
            or input.height > Config.max_size
            or input.size != output.size
            or input.get_colors(allow_black=True) != output.get_colors(allow_black=True)
            or len(input.get_colors(allow_black=True)) > Config.max_colors
        ):
            return False
    return True


def check_matching_colored_objects_count_and_color(examples: List[Example]) -> bool:
    for input, output in examples:
        input_objects = input.detect_colored_objects(background_color=0)
        output_objects = output.detect_colored_objects(background_color=0)
        if len(input_objects) != len(output_objects):
            return False

        different_color = any(
            input_object.first_color != output_object.first_color
            for input_object, output_object in zip(input_objects, output_objects)
        )

        if different_color:
            return False
    return True


def match_colored_objects(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object]]:

    logger.info(
        f"{'  ' * nesting_level}match_colored_objects examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    color_match = check_matching_colored_objects_count_and_color(examples)
    if color_match is None:
        return None
    # now the colored input

    # each example has the same number of input and output objects
    # so we can turn those lists into and ObjectListExample
    object_list_examples: List[Example[GridAndObjects]] = []

    def get_background_color(input: Object) -> int:
        background_color = 0  # TODO: determine background color
        return background_color

    def get_grid_and_objects(input: Object) -> GridAndObjects:
        background_color = get_background_color(input)
        input_objects: List[Object] = input.detect_colored_objects(background_color)
        return (input, input_objects)

    input_grid_and_objects: GridAndObjects
    output_grid_and_objects: GridAndObjects
    for input, output in examples:
        input_grid_and_objects = get_grid_and_objects(input)
        output_grid_and_objects = get_grid_and_objects(output)
        input_objects = input_grid_and_objects[1]
        output_objects = output_grid_and_objects[1]

        if len(input_objects) == 0 or len(output_objects) == 0:
            return None

        if False:
            display_multiple(
                [
                    (input_object, output_object)
                    for input_object, output_object in zip(
                        input_objects, output_objects
                    )
                ],
                title=f"Colored Objects [Exam]",
            )

        object_list_example: Example[GridAndObjects] = (
            input_grid_and_objects,
            output_grid_and_objects,
        )
        object_list_examples.append(object_list_example)

    for list_xform in list_xforms:
        match: Optional[Match[GridAndObjects]] = list_xform.xform(
            object_list_examples, task_name, nesting_level + 1
        )
        if match is not None:
            state_list_xform, apply_state_list_xform = match

            def solve(input: Object) -> Optional[Object]:
                background_color = get_background_color(input)
                input_objects = input.detect_colored_objects(background_color)
                grid_and_objects: GridAndObjects = (input, input_objects)
                result = apply_state_list_xform(grid_and_objects)
                if result is None:
                    return None
                s, output_objects = result
                output_grid = None
                if False:
                    display_multiple(
                        list(zip(input_objects, output_objects)),
                        title=f"Output Objects",
                    )
                for output in output_objects:
                    if output_grid is None:
                        output_grid = output.copy()
                    else:
                        output_grid.add_object_in_place(output)
                assert output_grid is not None
                if False:
                    display(output_grid, title=f"Output Grid")
                return output_grid

            return (
                f"{list_xform.xform.__name__}({state_list_xform})",
                solve,
            )
        else:
            logger.info(
                f"{'  ' * nesting_level}Xform {list_xform.xform.__name__} is not applicable"
            )

    return None


gridxforms: List[XformEntry[Object]] = [
    XformEntry(match_colored_objects, 3),
    XformEntry(xform_identity, 1),
    XformEntry(equal_modulo_rigid_transformation, 2),
    XformEntry(primitive_to_xform(translate_down_1), 2),
    XformEntry(canvas_grid_xform, 2),
    XformEntry(inpainting_xform_no_mask, 2),
] + (
    [
        XformEntry(inpainting_xform_with_mask, 2),
    ]
    if Config.find_frame_rule
    else []
)


# brute force search xforms to be used when all else fails
desperatexforms: List[XformEntry[Object]] = [] + (
    [XformEntry(frame_split_and_mirror_xform, 100)] if Config.find_frame_rule else []
)




map_xforms: List[XformEntry[Object]] = [XformEntry(stretch_height, 1)]


from typing import List, Tuple


class ObjectListMatch:
    @staticmethod
    def check_list_of_objects_subset(
        examples: List[Example[GridAndObjects]],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Check if the output objects are a subset of the input objects based on their colors.
        Returns a list of indices of the input objects that correspond to the output objects.
        The same list must apply to all examples.
        """
        input_to_output_indices_list = []
        for (_, input_objects), (_, output_objects) in examples:
            if len(input_objects) < 2:
                return None
            input_to_output_indices = out_objects_are_a_subset(
                input_objects, output_objects
            )
            if input_to_output_indices is None:
                return None
            # store the indices
            input_to_output_indices_list.append(input_to_output_indices)
        # check if they are all the same
        if len(set(tuple(indices) for indices in input_to_output_indices_list)) != 1:
            return None
        logger.info(f"input_to_output_indices_list: {input_to_output_indices_list}")
        input_to_output_indices = input_to_output_indices_list[0]
        if len(input_to_output_indices) == 0:
            return None
        return input_to_output_indices

    @staticmethod
    def map_first_input_to_output_grid(
        examples: List[Example[GridAndObjects]],
    ) -> List[Example[Object]]:
        input_output_objects_examples: List[Example[Object]] = []
        for (input_grid, input_objects), (output_grid, output_objects) in examples:
            input_output_objects_examples.append((input_objects[0], output_grid))

        return input_output_objects_examples

    @staticmethod
    def match_list_of_objects(
        examples: List[Example[GridAndObjects]],
        task_name: str,
        nesting_level: int,
    ) -> Optional[Match[GridAndObjects]]:
        logger.info(
            f"{'  ' * nesting_level}match_list_of_objects examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
        )

        if check_fractal_expansion_sizes(examples):
            input_output_objects_examples = (
                ObjectListMatch.map_first_input_to_output_grid(examples)
            )

            # now pattern match recursively
            match: Optional[Match[Object]] = find_xform_for_examples(
                expansion_xforms,
                input_output_objects_examples,
                task_name,
                nesting_level + 1,
            )
            if match is not None:
                state, solve = match

                def solve_grid_and_objects(
                    grid_and_objects: GridAndObjects,
                ) -> Optional[GridAndObjects]:
                    grid, objects = grid_and_objects
                    solved_objects_ = [solve(obj) for obj in objects]
                    solved_objects = [obj for obj in solved_objects_ if obj is not None]
                    if len(solved_objects) != len(objects):
                        return None
                    return (grid, solved_objects)

                return state, solve_grid_and_objects

        # check if the input objects can be matched to the output objects
        input_to_output_indices = ObjectListMatch.check_list_of_objects_subset(examples)
        if input_to_output_indices is not None:
            logger.info(
                f"{'  ' * nesting_level}Found input_to_output_indices: {input_to_output_indices}"
            )

            new_examples_train: List[List[Example[Object]]] = [
                [] for _ in input_to_output_indices
            ]
            for (_, e_inputs), (_, e_outputs) in examples:
                for i, (input_index, output_index) in enumerate(
                    input_to_output_indices
                ):
                    new_examples_train[i].append(
                        (e_inputs[input_index], e_outputs[output_index])
                    )

            for xform in map_xforms:
                matches = []  # for each input/output index pair, the match
                for i, (input_index, output_index) in enumerate(
                    input_to_output_indices
                ):
                    match = xform.xform(
                        new_examples_train[i],
                        task_name,
                        nesting_level,
                    )
                    if match is None:
                        logger.info(
                            f"Xform {xform.xform.__name__} index:{output_index} failed: no match"
                        )
                        return None
                    else:
                        matches.append(match)

            logger.info(f"Xform {xform.xform.__name__} succeeded")

            new_state = "{"
            for i, (s, _) in enumerate(matches):
                new_state += f"{i}:{s}, "
            new_state += "}"

            def solve_grid_and_objects(
                grid_and_objects: GridAndObjects,
            ) -> Optional[GridAndObjects]:
                input_grid, input_objects = grid_and_objects
                outputs = []
                assert input_to_output_indices is not None
                for i, (input_index, output_index) in enumerate(
                    input_to_output_indices
                ):
                    state, solve = matches[i]
                    output = solve(input_objects[input_index])
                    if output is None:
                        return None
                    outputs.append(output)
                return (input_grid, outputs)

            return new_state, solve_grid_and_objects

        logger.info(f"{'  ' * nesting_level}TODO: more cases of match_list_of_objects")

        return None


list_xforms: List[XformEntry[GridAndObjects]] = [
    XformEntry(ObjectListMatch.match_list_of_objects, 4),
]


def find_xform_for_examples(
    xforms: List[XformEntry[Object]],
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
    xform_name: List[str] = [],
) -> Optional[Match[Object]]:
    logger.info(
        f"\n{'  ' * nesting_level}find_xform_for_examples examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    for xform in xforms:
        if Config.difficulty < xform.difficulty:
            continue
        func = xform.xform
        logger.debug(f"{'  ' * nesting_level}Checking xform {func.__name__}")
        match = func(examples, task_name, nesting_level + 1)
        if match is not None:
            state, solve = match
            # sanity check: if it self detects an issue on the test input, fail
            first_test_input = examples[0][0]
            result_on_test = solve(first_test_input)
            if result_on_test is None:
                logger.info(
                    f"{'  ' * nesting_level}Xform {xform.xform.__name__} state:{match[0]} self detects an issue on the test input"
                )
                continue
            else:
                logger.info(
                    f"{'  ' * nesting_level}Xform {xform.xform.__name__} state:{match[0]} is correct for examples"
                )
            xform_name.append(xform.xform.__name__)
            return match
        else:
            logger.info(
                f"{'  ' * nesting_level}Xform {func.__name__} is not applicable"
            )

    return None


def find_xform(
    xforms: List[XformEntry[Object]],
    examples: List[Example[Object]],
    tests: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object]]:
    logger.info(
        f"\n{'  ' * nesting_level}find_xform examples:{len(examples)} tests:{len(tests)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    xform_name_list = ["no_xform"]
    match = find_xform_for_examples(
        xforms, examples, task_name, nesting_level, xform_name_list
    )
    if match is None:
        return None
    xform_name = xform_name_list[-1]

    state, solve = match

    for i, test_example in enumerate(tests):
        test_input = test_example[0]
        test_output = test_example[1]
        result_on_test = solve(test_input)
        if result_on_test is None:
            logger.info(
                f"Xform {xform_name} state:{state} failed returning None for test input {i}"
            )
            return None
        if result_on_test != test_output:
            logger.info(f"Xform {xform_name} state:{state} failed for test input {i}")
            if Config.display_verbose:
                width, height = test_output.size
                for x in range(width):
                    for y in range(height):
                        if test_output[x, y] != result_on_test[x, y]:
                            logger.info(
                                f"Xform {xform_name} state:{state} failed for test input {i} at {x},{y}: {test_output[x, y]} != {result_on_test[x, y]}"
                            )
                display(
                    result_on_test,
                    test_output,
                    title=f"Test {i} Fail",
                    left_title=f"Result",
                    right_title=f"Expected",
                )
            return None

    logger.info(f"Xform {xform_name} state:{state} succeeded for all tests")
    return match


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
            embeddings = [detect_symmetry_features(obj) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Symmetry): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
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
            embeddings = [
                detect_color_features(obj, input_objects) for obj in input_objects
            ]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Color): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
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
            embeddings = [
                detect_shape_features(obj, input_objects, level)
                for obj in input_objects
            ]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Shape): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
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
        common_decision_rule = detect_common_shape_features(initial_difficulty + 1)
        features_used = "Shape"

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 2:
        common_decision_rule = detect_common_color_features()
        features_used = "Color"

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 3:
        common_decision_rule = detect_common_symmetry_features()
        features_used = "Symmetry"
    assert num_difficulties_matching == 3

    return common_decision_rule, features_used


def find_matched_objects(
    examples: List[Example], task_type: str
) -> Optional[List[ObjectMatch]]:
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

    def candidate_objects_for_matching(input: Object, output: Object) -> List[Object]:
        """
        Detects objects in the input grid that are candidates for matching the output grid.
        """
        if output.has_frame():
            # If the output is a frame, detect objects in the input as frames
            logger.debug("  Output is a frame")
        num_colors_output = len(output.get_colors(allow_black=True))
        return find_rectangular_objects(input, allow_multicolor=num_colors_output > 1)

    def find_matching_input_object(
        input_objects: List[Object], output: Object
    ) -> Optional[int]:
        for i, io in enumerate(input_objects):
            if io.size == output.size and np.array_equal(io._data, output._data):
                logger.debug(f"  Input object matching output: {io}")
                return i
        return None

    def get_matched_objects(examples: List[Example]) -> Optional[List[ObjectMatch]]:
        matched_objects: List[ObjectMatch] = []

        for example in examples:
            input = example[0]
            output = example[1]
            logger.info(f"  {task_type} {input.size} -> {output.size}")

            input_objects = candidate_objects_for_matching(input, output)
            matched_object_index = find_matching_input_object(input_objects, output)

            if matched_object_index is not None:
                matched_objects.append((input_objects, matched_object_index))

        return matched_objects if len(matched_objects) == len(examples) else None

    matched_objects = get_matched_objects(examples)
    return matched_objects


num_difficulties_xform = max(xform.difficulty for xform in gridxforms + desperatexforms)
num_difficulties_matching = 3
num_difficulties_total = num_difficulties_xform + num_difficulties_matching


def process_tasks(tasks: Tasks, set: str):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in tasks.items():
        Config.display_this_task = False
        if Config.task_name and task_name != Config.task_name:
            continue
        if task_name in Config.blacklisted_tasks:
            continue
        if (
            filter_simple_xforms(task, task_name) == False
            and Config.only_simple_examples
            and task_name not in Config.whitelisted_tasks
        ):
            continue
        if Config.only_inpainting_puzzles and not is_inpainting_puzzle(task.train):
            continue
        logger.info(f"\n***Task: {task_name} {set}***")

        examples = task.train

        tests = task.test
        task_type = "train"

        if True:
            current_difficulty = 0

            if Config.find_xform:
                correct_xform = find_xform(
                    gridxforms + desperatexforms, examples, tests, task_name, 0
                )
                if correct_xform is not None:
                    num_correct += 1
                    if False:
                        grids = [(example[0], example[1]) for example in examples]
                        display_multiple(grids, title=f"{task_name} {set}")
                    continue

            current_difficulty += num_difficulties_xform

            if Config.find_matched_objects:
                # Check if the input objects can be matched to the output objects
                logger.debug(f"Checking common features for {task_name} {set}")
                matched_objects = find_matched_objects(examples, task_type)
                if matched_objects:
                    # If the input objects can be matched to the output objects, try to detect common features
                    # to determine the correct object to pick
                    logger.debug(
                        f"XXX Matched {len(matched_objects)}/{len(examples)} {task_name} {set}"
                    )
                    common_decision_rule, features_used = detect_common_features(
                        matched_objects, current_difficulty
                    )
                    if common_decision_rule:
                        logger.info(
                            f"Common decision rule ({features_used}): {common_decision_rule}"
                        )
                        num_correct += 1
                        continue
                    else:
                        logger.warning(
                            f"Could not find common decision rule for {task_name} {set}"
                        )
            current_difficulty += num_difficulties_matching

            if Config.display_not_found:
                Config.display_this_task = True
            if Config.display_this_task:
                grids = [(example[0], example[1]) for example in examples]
                display_multiple(grids, title=f"{task_name} {set}")

            # If no valid dimensions could be determined, give up
            logger.warning(
                f"Could not find correct transformation for {task_name} {set} examples"
            )
            num_incorrect += 1

    return num_correct, num_incorrect


def compute_perc_correct(num_correct: int, num_incorrect: int) -> Optional[float]:
    if num_correct + num_incorrect > 0:
        return int(1000 * num_correct / (num_correct + num_incorrect)) / 10
    return None


def simple():
    num_correct_tr, num_incorrect_tr = process_tasks(training_data, "training_data")
    num_correct_ev, num_incorrect_ev = process_tasks(evaluation_data, "evaluation_data")
    perc_correct_tr = compute_perc_correct(num_correct_tr, num_incorrect_tr)
    perc_correct_ev = compute_perc_correct(num_correct_ev, num_incorrect_ev)

    def log_evaluation_results(set: str, num_correct: int, num_incorrect: int):
        perc_correct = compute_perc_correct(num_correct, num_incorrect)
        if perc_correct is not None:
            logger.error(
                f"{set.capitalize()} data: "
                f"Correct: {num_correct}, Incorrect: {num_incorrect}, Score: {perc_correct}%"
            )

    logger.error("\n***Summary***")
    log_evaluation_results("training", num_correct_tr, num_incorrect_tr)
    log_evaluation_results("evaluation", num_correct_ev, num_incorrect_ev)

    # Write summary of results to JSON file
    with open("simple.json", "w") as f:
        f.write(
            f'{{"training_data":{perc_correct_tr},"evaluation_data":{perc_correct_ev}}}'
        )


if __name__ == "__main__":
    simple()
