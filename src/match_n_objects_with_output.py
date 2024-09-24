from typing import List, Optional

from bi_types import Example, Match, Object
from logger import logger
from objects import display_multiple, display
from visual_cortex import find_rectangular_objects
import config


def match_n_objects_with_output(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_n_objects_with_output examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )
    for input, output in examples:
        if input.size == output.size:
            return None

        background_color = input.main_color()
        objects = input.detect_objects(
            diagonals=True, background_color=background_color, multicolor=True
        )
        n = len(objects)
        if n <= 1:
            return None

        output_indices = [i for i, o in enumerate(objects) if o.size == output.size]
        if len(output_indices) != 1:
            return None
        output_index = output_indices[0]

        logger.info(f"{'  ' * nesting_level} num objects:{n} output index:{output_index}")
    config.display_this_task = True

    return None
