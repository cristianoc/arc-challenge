from typing import List, Optional

from bi_types import Config, XformEntry
from canvas_grid_match import canvas_grid_xform, equal_modulo_rigid_transformation
from find_xform import find_xform
from inpainting_match import (
    inpainting_xform_no_mask,
    inpainting_xform_with_mask,
    is_inpainting_puzzle,
)
from load_data import Task, Tasks, evaluation_data, training_data
from logger import logger
from match_colored_objects import match_colored_objects
from matched_objects import handle_matched_objects
from objects import Object, display_multiple
from primitives import primitive_to_xform, translate_down_1, xform_identity
from split_mirrot_match import frame_split_and_mirror_xform


def filter_simple_xforms(task: Task, task_name: str):
    examples = task.train
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


def filter_complex_xforms(task: Task, task_name: str):
    examples = task.train
    for example in examples:
        input = example[0]
        output = example[1]
        if (
            input.width < Config.min_size
            or input.height < Config.min_size
            or len(input.get_colors(allow_black=True)) < Config.min_colors
        ):
            return False
    return True


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
            Config.only_simple_examples
            and filter_simple_xforms(task, task_name) == False
            and task_name not in Config.whitelisted_tasks
        ):
            continue
        if (
            Config.only_complex_examples
            and filter_complex_xforms(task, task_name) == False
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
                should_continue = handle_matched_objects(
                    examples, task_name, task_type, set, current_difficulty
                )
                if should_continue:
                    continue
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
