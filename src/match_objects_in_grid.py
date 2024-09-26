from typing import List, Optional, Callable

from bi_types import Examples, Match
from logger import logger
from match_object_list_to_object import match_object_list_with_decision_rule
from objects import Object
from visual_cortex import find_rectangular_objects


def get_objects(input: Object, multicolor: bool) -> List[Object]:
    objects = find_rectangular_objects(input, allow_multicolor=multicolor)
    objects_reset_origin = [Object(o._data) for o in objects]
    return objects_reset_origin


def get_objects_monocolor(input: Object) -> List[Object]:
    return get_objects(input, multicolor=False)


def get_objects_multicolor(input: Object) -> List[Object]:
    return get_objects(input, multicolor=True)


def candidate_objects_for_matching(
    input: Object, output: Object, get_objects: Callable[[Object], List[Object]]
) -> List[Object]:
    """
    Detects objects in the input grid that are candidates for matching the output grid.
    """
    if output.has_frame():
        # If the output is a frame, detect objects in the input as frames
        logger.debug("  Output is a frame")
    objects = get_objects(input)
    return objects


def match_rectangular_objects_in_grid(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_rectangular_objects_in_grid examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    def gen_examples2(get_objects: Callable[[Object], List[Object]]):
        examples2 = []
        for input, output in examples:
            input_objects = candidate_objects_for_matching(input, output, get_objects)
            examples2.append((input_objects, output))
        return examples2

    # Try mono-color first, then multi-color if needed
    # Mono-color establishes stronger invariants and prevents false positives
    # Example: https://arcprize.org/play?task=1f85a75f
    match_non_multicolor = match_object_list_with_decision_rule(
        gen_examples2(get_objects_monocolor),
        task_name,
        nesting_level + 1,
        minimal=False,
        get_objects=get_objects_monocolor,
    )
    if match_non_multicolor is not None:
        return match_non_multicolor

    match_multicolor = match_object_list_with_decision_rule(
        gen_examples2(get_objects_multicolor),
        task_name,
        nesting_level + 1,
        minimal=False,
        get_objects=get_objects_multicolor,
    )
    return match_multicolor
