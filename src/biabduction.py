from typing import List, Optional

import config
import xforms
from bi_types import Examples
from find_xform import find_xform
from inpainting_match import is_inpainting_puzzle
from load_data import Task, Tasks, evaluation_data, training_data
from logger import logger
from objects import Object, display_multiple


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


def process_tasks(tasks: Tasks, set: str):
    correct = []
    incorrect = []
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
                xforms.gridxforms + xforms.desperatexforms, examples, tests, task_name, 0
            )
            if correct_xform is not None:
                correct.append(task_name)
                if False:
                    grids = [(example[0], example[1]) for example in examples]
                    display_multiple(grids, title=f"{task_name} {set}")
                continue

        if config.display_not_found:
            config.display_this_task = True
        if config.display_this_task:
            grids = [(example[0], example[1]) for example in examples]
            display_multiple(grids, title=f"{task_name} {set}")

        # If no valid dimensions could be determined, give up
        logger.warning(
            f"Could not find correct transformation for {task_name} {set} examples"
        )
        incorrect.append(task_name)

    return correct, incorrect


def compute_perc_correct(correct: List[str], incorrect: List[str]) -> Optional[float]:
    if len(correct) + len(incorrect) > 0:
        return int(1000 * len(correct) / (len(correct) + len(incorrect))) / 10
    return None


def bi_abduction():
    xforms.init()
    correct_tr, incorrect_tr = process_tasks(training_data, "training_data")
    correct_ev, incorrect_ev = process_tasks(evaluation_data, "evaluation_data")
    perc_correct_tr = compute_perc_correct(correct_tr, incorrect_tr)
    perc_correct_ev = compute_perc_correct(correct_ev, incorrect_ev)

    def log_evaluation_results(set: str, correct: List[str], incorrect: List[str]):
        perc_correct = compute_perc_correct(correct, incorrect)
        if perc_correct is not None:
            logger.error(
                f"{set.capitalize()} data: "
                f"Correct: {len(correct)}, Incorrect: {len(incorrect)}, Score: {perc_correct}%"
            )

    logger.error("\n***Summary***")
    log_evaluation_results("training", correct_tr, incorrect_tr)
    log_evaluation_results("evaluation", correct_ev, incorrect_ev)

    # Write summary of results to JSON file
    import json

    results = {
        "training_score": perc_correct_tr,
        "evaluation_score": perc_correct_ev,
        "training_correct": correct_tr,
        "evaluation_correct": correct_ev
    }

    with open("simple.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    bi_abduction()
