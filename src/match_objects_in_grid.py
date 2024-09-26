from typing import List, Optional

from bi_types import Examples, Match
from logger import logger
from match_object_list_to_object import match_object_list_with_decision_rule
from objects import Object, display_multiple
from visual_cortex import extract_lattice_subgrids, find_rectangular_objects


def get_objects_multicolor(input: Object) -> List[Object]:
    objects = find_rectangular_objects(input, allow_multicolor=True)
    objects_reset_origin = [Object(o._data) for o in objects]
    return objects_reset_origin


def get_objects_non_multicolor(input: Object) -> List[Object]:
    objects = find_rectangular_objects(input, allow_multicolor=False)
    objects_reset_origin = [Object(o._data) for o in objects]
    return objects_reset_origin


def candidate_objects_for_matching_multicolor(input: Object, output: Object) -> List[Object]:
    """
    Detects objects in the input grid that are candidates for matching the output grid.
    """
    if output.has_frame():
        # If the output is a frame, detect objects in the input as frames
        logger.debug("  Output is a frame")
    objects = get_objects_multicolor(input)
    return objects


def candidate_objects_for_matching_non_multicolor(
    input: Object, output: Object
) -> List[Object]:
    """
    Detects objects in the input grid that are candidates for matching the output grid.
    """
    if output.has_frame():
        # If the output is a frame, detect objects in the input as frames
        logger.debug("  Output is a frame")
    objects = get_objects_non_multicolor(input)
    return objects


def match_rectangular_objects_in_grid_multicolor(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_rectangular_objects_in_grid examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )
    examples2 = []
    for input, output in examples:
        input_objects = candidate_objects_for_matching_multicolor(input, output)
        examples2.append((input_objects, output))

    return match_object_list_with_decision_rule(
        examples2, task_name, nesting_level + 1, minimal=False, get_objects=get_objects_multicolor
    )


def match_rectangular_objects_in_grid_non_multicolor(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_rectangular_objects_in_grid examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )
    examples2 = []
    for input, output in examples:
        input_objects = candidate_objects_for_matching_non_multicolor(input, output)
        examples2.append((input_objects, output))

    return match_object_list_with_decision_rule(
        examples2, task_name, nesting_level + 1, minimal=False, get_objects=get_objects_non_multicolor
    )
