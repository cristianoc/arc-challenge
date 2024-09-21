from typing import List, Optional

from bi_types import Config, Match, XformEntry
from load_data import Example
from logger import logger
from objects import Object, display


def find_xform_for_examples(
    xforms: List[XformEntry[Object, Object]],
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
    xform_name: List[str] = [],
) -> Optional[Match[Object, Object]]:
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
    xforms: List[XformEntry[Object, Object]],
    examples: List[Example[Object]],
    tests: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
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
