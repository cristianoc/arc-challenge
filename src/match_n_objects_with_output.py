from typing import List, Optional, Tuple, Callable

import config
from bi_types import Examples, Match, Object, Xform, XformEntry
from logger import logger
from objects import display, display_multiple


def matcher_from_primitive(
    get_index: Callable[[List[Object]], int]
) -> Xform[List[Object], int]:
    def matcher(
        examples: Examples[List[Object], int], task_name: str, nesting_level: int
    ) -> Optional[Match[List[Object], int]]:
        for inputs, index in examples:
            if index != get_index(inputs):
                return None

        state = "match_smallest_object"

        state = f"matcher_from_primitive({get_index.__name__})"
        return (state, lambda inputs: get_index(inputs))

    return matcher

def feat_smallest_area(inputs: List[Object]) -> int:
    smallest_index, _ = min(
        [(i, o) for i, o in enumerate(inputs)],
        key=lambda x: x[1].area,
    )
    return smallest_index

# TODO: use detect_common_features

select_object_xforms: List[XformEntry[List[Object], int]] = [
    XformEntry(matcher_from_primitive(feat_smallest_area), 3)
]


def match_n_objects_with_output(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_n_objects_with_output examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )
    objext_and_index_list: List[Tuple[List[Object], int]] = []
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

        objext_and_index_list.append((objects, output_index))

    for xform in select_object_xforms:
        result = xform.xform(objext_and_index_list, task_name, nesting_level)
        if result is None:
            continue
        config.display_this_task = True

    return None
