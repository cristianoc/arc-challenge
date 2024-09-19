from bi_types import Example, GridAndObjects, Match
from logger import logger
from objects import Object, display_multiple, display
from typing import List, Optional
from bi_types import XformEntry
from object_list_match import match_list_of_objects



list_xforms: List[XformEntry[GridAndObjects]] = [
    XformEntry(match_list_of_objects, 4),
]

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
