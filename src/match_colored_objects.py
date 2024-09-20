from typing import Callable, List, Optional

from bi_types import Example, GridAndObjects, Match
from logger import logger
from match_object_list_to_object import match_object_list_to_object_by_painting
from objects import Object, display_multiple


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

    def get_objects(input: Object) -> List[Object]:
        background_color = get_background_color(input)
        input_objects = input.detect_colored_objects(background_color)
        return input_objects

    def get_grid_and_objects(input: Object) -> GridAndObjects:
        input_objects = get_objects(input)
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

    return match_object_list_to_object_by_painting(object_list_examples, get_objects, task_name, nesting_level)
