from typing import Callable, List, Optional, Set, Tuple, TypedDict

from predict_size import (
    Config,
    ExampleGrids,
    compute_perc_correct,
    detect_common_features,
    find_matched_objects,
    num_difficulties_matching,
)
from grid import Grid
from grid_data import BLACK, GREY, GridData, display, display_multiple, logger
from load_data import Example, Task, Tasks, iter_tasks, training_data, evaluation_data


ColorXform = Callable[
    [ExampleGrids, Grid, str], Optional[Set[int]]
]  # List[int] for color indexes


class ColorXformEntry(TypedDict):
    function: ColorXform
    difficulty: int


def output_colors_are_input_colors(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return set(grid.get_colors())


def output_colors_are_input_colors_minus_black_grey(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return set(grid.get_colors(allow_black=True)) - {BLACK, GREY}


def output_colors_are_input_colors_plus_black(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return set(grid.get_colors(allow_black=True)) | {BLACK}


def output_colors_are_constant(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return set(grids[0][1].get_colors())


def output_colors_are_input_colors_minus_num_colors(
    grids: ExampleGrids, grid: Grid, task_name: str, num: int
) -> Optional[Set[int]]:
    # Check in grids if there are num colors that are always removed from the input to the output
    # If found, remove them from the grid colors
    candidate_colors: Optional[Set[int]] = None
    for input, output in grids:
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


def output_colors_are_input_colors_minus_one_color(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return output_colors_are_input_colors_minus_num_colors(grids, grid, task_name, 1)


def output_colors_are_input_colors_minus_two_colors(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return output_colors_are_input_colors_minus_num_colors(grids, grid, task_name, 2)


def output_colors_are_input_colors_plus_num_colors(
    grids: ExampleGrids, grid: Grid, task_name: str, num: int
) -> Optional[Set[int]]:
    # Check in grids if there are num colors that are always added from the input to the output
    # If found, add them to the grid colors
    candidate_colors: Optional[Set[int]] = None
    for input, output in grids:
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


def output_colors_are_input_colors_plus_one_color(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return output_colors_are_input_colors_plus_num_colors(grids, grid, task_name, 1)


def output_colors_are_input_colors_plus_two_colors(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    return output_colors_are_input_colors_plus_num_colors(grids, grid, task_name, 2)


def output_colors_are_inout_colors_minus_one_color_plus_another_color(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    # Check in grids if there is one color that is always removed from the input to the output
    # and another color that is always added
    # If found, remove the removed color and add the added color to the grid colors
    candidate_removed_color: Optional[int] = None
    candidate_added_color: Optional[int] = None
    for input, output in grids:
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
        if (
            candidate_removed_color != removed_color
            or candidate_added_color != added_color
        ):
            return None
    if candidate_removed_color is None or candidate_added_color is None:
        return None
    return (set(grid.get_colors()) - {candidate_removed_color}) | {
        candidate_added_color
    }


def output_colors_are_input_colors_minus_color_of_max_black_cells_object(
    grids: ExampleGrids, grid: Grid, task_name: str
) -> Optional[Set[int]]:
    objects = grid.detect_objects()
    if not objects:
        return None
    max_black_cells_object = max(objects, key=lambda obj: obj.num_cells(color=0))
    return set(grid.get_colors()) - {max_black_cells_object.main_color()}


xforms: List[ColorXformEntry] = [
    {"function": output_colors_are_input_colors, "difficulty": 1},
    {"function": output_colors_are_input_colors_plus_black, "difficulty": 1},
    {"function": output_colors_are_input_colors_minus_black_grey, "difficulty": 1},
    {"function": output_colors_are_constant, "difficulty": 2},
    {"function": output_colors_are_input_colors_minus_one_color, "difficulty": 3},
    {"function": output_colors_are_input_colors_minus_two_colors, "difficulty": 3},
    {"function": output_colors_are_input_colors_plus_one_color, "difficulty": 3},
    {"function": output_colors_are_input_colors_plus_two_colors, "difficulty": 3},
    {
        "function": output_colors_are_inout_colors_minus_one_color_plus_another_color,
        "difficulty": 4,
    },
    {
        "function": output_colors_are_input_colors_minus_color_of_max_black_cells_object,
        "difficulty": 5,
    },
]


def check_xform_on_examples(
    xform: ColorXform, examples: List[Example], task_name: str, task_type: str
) -> bool:
    grids = [(Grid(example["input"]), Grid(example["output"])) for example in examples]
    logger.debug(f"Checking xform {xform.__name__} {task_type}")
    for i, example in enumerate(examples):
        logger.debug(f"  Example {i+1}/{len(examples)}")
        input = Grid(example["input"])
        output = Grid(example["output"])
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


def find_xform(
    examples: List[Example], task: Task, task_name: str, task_type: str
) -> Optional[ColorXformEntry]:
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
                    display(e["input"], output=e["output"], title=f"Ex{i+1} " + title)
            correct_xform = xform
            logger.info(
                f"Xform {correct_xform['function'].__name__} is correct for all examples in {task_type}"
            )
            test_examples = [
                examples for task_type, examples in task.items() if task_type == "test"
            ]
            for i, test_example in enumerate(test_examples):
                if not check_xform_on_examples(
                    correct_xform["function"], test_example, task_name, "test"
                ):
                    logger.warning(
                        f"Xform {correct_xform['function'].__name__} failed for test example {i}"
                    )
                    correct_xform = None
                    break
            if correct_xform:
                break
    return correct_xform


num_difficulties_xform = max(xform["difficulty"] for xform in xforms)
num_difficulties_total = num_difficulties_xform + num_difficulties_matching


def process_tasks(tasks: Tasks, set: str):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in iter_tasks(tasks):
        if Config.task_name and task_name != Config.task_name:
            continue
        logger.info(f"\n***Task: {task_name} {set}***")

        for task_type, examples in task.items():
            if task_type not in ["train", "test"]:
                continue
            if task_type == "test":
                continue

            current_difficulty = 0

            if Config.find_xform_color:
                correct_xform = find_xform(examples, task, task_name, task_type)
                if correct_xform:
                    logger.info(
                        f"Xform {correct_xform['function'].__name__} is correct for all examples in {task_type} and test"
                    )
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

            num_incorrect += 1
            logger.warning(
                f"Could not find correct color transformation for {task_name} {set}"
            )
            if Config.display_not_found:
                grids: List[Tuple[GridData, Optional[GridData]]] = [
                    (Grid(example["input"]).data, Grid(example["output"]).data)
                    for example in examples
                ]
                display_multiple(grids, title=f"{task_name} {set}")

    return num_correct, num_incorrect


def predict_colors():
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
    with open("predict_colors.json", "w") as f:
        f.write(
            f'{{"training_data":{perc_correct_tr},"evaluation_data":{perc_correct_ev}}}'
        )


if __name__ == "__main__":
    predict_colors()
