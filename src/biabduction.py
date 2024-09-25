from typing import List, Optional

import config
from bi_types import Examples, XformEntry
from canvas_grid_match import canvas_grid_xform, equal_modulo_rigid_transformation
from find_xform import find_xform
from inpainting_match import (
    inpainting_xform_no_mask,
    inpainting_xform_output_is_block,
    inpainting_xform_with_mask,
    is_inpainting_puzzle,
)
from load_data import Task, Tasks, evaluation_data, training_data
from logger import logger
from match_colored_objects import match_colored_objects
from match_n_objects_with_output import match_n_objects_with_output
from match_objects_in_grid import match_rectangular_objects_in_grid
from match_subgrids_in_lattice import match_subgrids_in_lattice
from objects import Object, display_multiple
from primitives import primitive_to_xform, translate_down_1, xform_identity
from split_mirrot_match import frame_split_and_mirror_xform


def filter_simple_xforms(task: Task, task_name: str):
    examples = task.train
    for example in examples:
        input = example[0]
        output = example[1]
        if (
            input.width > config.max_size
            or input.height > config.max_size
            or input.size != output.size
            or input.get_colors(allow_black=True) != output.get_colors(allow_black=True)
            or len(input.get_colors(allow_black=True)) > config.max_colors
        ):
            return False
    return True


def filter_complex_xforms(task: Task, task_name: str):
    examples = task.train
    for example in examples:
        input = example[0]
        output = example[1]
        if (
            input.width < config.min_size
            or input.height < config.min_size
            or len(input.get_colors(allow_black=True)) < config.min_colors
        ):
            return False
    return True


gridxforms: List[XformEntry[Object, Object]] = (
    [
        XformEntry(match_subgrids_in_lattice, 3),
        XformEntry(match_colored_objects, 3),
        XformEntry(xform_identity, 1),
        XformEntry(equal_modulo_rigid_transformation, 2),
        XformEntry(primitive_to_xform(translate_down_1), 2),
        XformEntry(canvas_grid_xform, 2),
        XformEntry(match_rectangular_objects_in_grid, 3),
        XformEntry(inpainting_xform_no_mask, 2),
        XformEntry(inpainting_xform_output_is_block, 2),
        # XformEntry(match_n_objects_with_output, 3),
    ]
    + []
    + (
        [
            XformEntry(inpainting_xform_with_mask, 2),
        ]
        if config.find_frame_rule
        else []
    )
)


# brute force search xforms to be used when all else fails
desperatexforms: List[XformEntry[Object, Object]] = [] + (
    [XformEntry(frame_split_and_mirror_xform, 100)] if config.find_frame_rule else []
)


num_difficulties_xform = max(xform.difficulty for xform in gridxforms + desperatexforms)


def process_tasks(tasks: Tasks, set: str):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in tasks.items():
        config.display_this_task = False
        if config.task_name and task_name != config.task_name:
            continue
        if task_name in config.blacklisted_tasks:
            continue
        if (
            config.only_simple_examples
            and filter_simple_xforms(task, task_name) == False
            and task_name not in config.whitelisted_tasks
        ):
            continue
        if (
            config.only_complex_examples
            and filter_complex_xforms(task, task_name) == False
        ):
            continue
        if config.only_inpainting_puzzles and not is_inpainting_puzzle(
            task.train, output_is_block=False
        ):
            continue
        logger.info(f"\n***Task: {task_name} {set}***")

        examples: Examples[Object, Object] = task.train
        tests: Examples[Object, Object] = task.test
        task_type = "train"

        current_difficulty = 0

        if config.find_xform:
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

        if config.display_not_found:
            config.display_this_task = True
        if config.display_this_task:
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


def bi_abduction():
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
    bi_abduction()
