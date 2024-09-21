from typing import Callable, List, Optional

from logger import logger
from objects import Object, display_multiple, display
from bi_types import GridAndObjects, Match, XformEntry
from load_data import Example
from match_object_list import match_object_list

object_list_xforms: List[XformEntry[GridAndObjects, GridAndObjects]] = [
    XformEntry(match_object_list, 4),
]

def match_object_list_to_object_by_painting(
    examples: List[Example[GridAndObjects]],
    get_objects: Callable[[Object], List[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    """
    Attempts to transform a list of examples by sequentially painting objects on top of each other.

    Args:
        examples (List[Example[GridAndObjects]]): A list of examples to be transformed.
        get_objects (Callable[[Object], List[Object]]): A callable that extracts a list of objects from an Object.
        task_name (str): The name of the task for logging and identification.
        nesting_level (int): The current level of nesting for logging purposes.

    Returns:
        Optional[Match[Object]]: A tuple containing the transformation name and a solver function if a match is found, otherwise None.
    """
    for list_xform in object_list_xforms:
        match: Optional[Match[GridAndObjects, GridAndObjects]] = list_xform.xform(
            examples, task_name, nesting_level + 1
        )
        if match is not None:
            state_list_xform, apply_state_list_xform = match

            def solve(input: Object) -> Optional[Object]:
                input_objects = get_objects(input)
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

def get_background_color(input: Object) -> int:
    background_color = 0  # TODO: determine background color
    return background_color


def get_colored_objects(input: Object) -> List[Object]:
    background_color = get_background_color(input)
    input_objects = input.detect_colored_objects(background_color)
    return input_objects


def match_colored_objects_to_object_by_painting(
    examples: List[Example[GridAndObjects]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    return match_object_list_to_object_by_painting(
        examples, get_colored_objects, task_name, nesting_level
    )


object_list_to_object_xforms: List[XformEntry[GridAndObjects, Object]] = [
    XformEntry(match_colored_objects_to_object_by_painting, 4),
]
