from typing import List, Optional

from bi_types import Example, Match, Object
from logger import logger
from objects import display_multiple, display
from visual_cortex import find_rectangular_objects
import config


def match_two_objects_with_output(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_two_objects_with_output examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )
    for input, output in examples:
        background_color = 0  # TODO
        objects = input.detect_objects(
            diagonals=True, background_color=background_color, multicolor=True
        )
        if len(objects) == 0:
            return None
        largest_object = max(objects, key=lambda o: o.area)
        other_objects = [o for o in objects if o != largest_object]
        if len(other_objects) != 1:
            return None
        other_object = other_objects[0]

        if largest_object.size == input.size:
            return None
        if output.size != largest_object.size and output.size != other_object.size:
            return None

        config.display_this_task = True
        
        # display_multiple(
        #     [largest_object, other_object, output], title=f"objects:{len(objects)}"
        # )

    return None
