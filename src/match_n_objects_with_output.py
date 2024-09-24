from typing import List, Optional, Tuple

import config
from bi_types import Examples, Match, Object, Xform, XformEntry
from logger import logger
from objects import display, display_multiple


def match_smallest_object(
    examples: Examples[List[Object], int], task_name: str, nesting_level: int
) -> Optional[Match[List[Object], int]]:
    def get_smallest_index(inputs: List[Object]) -> int:
        smallest_index, _ = min(
            [(i, o) for i, o in enumerate(inputs)],
            key=lambda x: x[1].size,
        )
        return smallest_index

    for inputs, index in examples:
        smallest_index = get_smallest_index(inputs)
        if smallest_index != index:
            return None

    state = "match_smallest_object"

    def solve(inputs: List[Object]) -> Optional[int]:
        return get_smallest_index(inputs)

    match : Match[List[Object], int] = (state, solve)
    return match

q :Xform[List[Object], int] = match_smallest_object

# Xform = Callable[[Examples[T1, T2], str, int], Optional[Match[T2, T2]]]


# xform to select the index of the object that matches the output
# select_object_xforms: List[XformEntry[List[Object], int]] = [
#     XformEntry(match_smallest_object, 3)
# ]


def match_n_objects_with_output(
    examples: Examples[Object, Object],
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

        logger.info(
            f"{'  ' * nesting_level} num objects:{n} output index:{output_index}"
        )
    config.display_this_task = True

    return None
