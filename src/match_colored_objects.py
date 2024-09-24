from typing import Callable, List, Optional

from bi_types import Example, Examples, GridAndObjects, Match, XformEntry
from find_xform import find_xform_for_examples
from logger import logger
from match_object_list_to_object import (
    get_colored_objects,
    match_object_list_to_object_by_painting,
    object_list_to_object_xforms,
)
from objects import Object, display_multiple


def check_matching_colored_objects_count_and_color(examples: Examples[Object, Object]) -> bool:
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
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:

    logger.info(
        f"{'  ' * nesting_level}match_colored_objects examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    color_match = check_matching_colored_objects_count_and_color(examples)
    if color_match is None:
        return None
    # now the colored input

    # each example has the same number of input and output objects
    # so we can turn those lists into and ObjectListExample
    object_list_examples: Examples[GridAndObjects, GridAndObjects] = []

    def get_grid_and_objects(input: Object) -> GridAndObjects:
        input_objects = get_colored_objects(input)
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

        object_list_example: Example[GridAndObjects, GridAndObjects] = (
            input_grid_and_objects,
            output_grid_and_objects,
        )
        object_list_examples.append(object_list_example)

    for xform in object_list_to_object_xforms:
        match = xform.xform(object_list_examples, task_name, nesting_level+1)
        if match is None:
            continue
        return match

    return None
