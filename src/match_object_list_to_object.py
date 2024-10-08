from typing import Callable, List, Optional, Tuple

import config
from bi_types import Examples, Match, XformEntry
from logger import logger
from match_object_list import match_object_list
from matched_objects import (
    ObjectMatch,
    check_grid_satisfies_rule,
    detect_common_features,
)
from objects import Object, display, display_multiple

object_list_xforms: List[XformEntry[List[Object], List[Object]]] = [
    XformEntry(match_object_list, 4),
]


def match_object_list_to_object_by_painting(
    examples: Examples[List[Object], List[Object]],
    get_objects: Callable[[Object], List[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    """
    Attempts to transform a list of examples by sequentially painting objects on top of each other.

    Args:
        examples (List[Example[List[Object]]]): A list of examples to be transformed.
        get_objects (Callable[[Object], List[Object]]): A callable that extracts a list of objects from an Object.
        task_name (str): The name of the task for logging and identification.
        nesting_level (int): The current level of nesting for logging purposes.

    Returns:
        Optional[Match[Object]]: A tuple containing the transformation name and a solver function if a match is found, otherwise None.
    """
    for list_xform in object_list_xforms:
        match: Optional[Match[List[Object], List[Object]]] = list_xform.xform(
            examples, task_name, nesting_level + 1
        )
        if match is not None:
            list_state, list_solve = match

            def solve(input: Object) -> Optional[Object]:
                input_objects = get_objects(input)
                output_objects = list_solve(input_objects)
                if output_objects is None:
                    return None
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
                f"{list_xform.xform.__name__}({list_state})",
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


def match_object_list_with_decision_rule(
    examples: List[Tuple[List[Object], Object]],
    task_name: str,
    nesting_level: int,
    minimal: bool,
    get_objects: Callable[[Object], List[Object]],
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_object_list_with_decision_rule examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    object_matches: List[ObjectMatch] = []
    for input_objs, output_obj in examples:
        try:
            index = input_objs.index(output_obj)
            logger.info(
                f"{'  ' * nesting_level}match_object_list_with_decision_rule found a match at index {index}"
            )
            object_matches.append(
                ObjectMatch(input_objects=input_objs, matched_index=index)
            )
        except ValueError:
            logger.info(
                f"{'  ' * nesting_level}match_object_list_with_decision_rule no match"
            )
    common_decision_rule, features_used = detect_common_features(
        object_matches, 3, minimal
    )
    if common_decision_rule is None:
        logger.info(
            f"{'  ' * nesting_level}match_object_list_with_decision_rule common_decision_rule is None"
        )
        return None
    logger.info(
        f"{'  ' * nesting_level}match_object_list_with_decision_rule common_decision_rule:{common_decision_rule}"
    )

    state = f"Select_sub({common_decision_rule})"

    def solve(input_g: Object) -> Optional[Object]:
        input_subgrids = get_objects(input_g)
        if config.display_verbose:
            display_multiple(input_subgrids, title=f"input_subgrids")
        # need to find the subgrid that satisfies the common_decision_rule
        for i, subgrid in enumerate(input_subgrids):
            if check_grid_satisfies_rule(subgrid, input_subgrids, common_decision_rule):
                if config.display_verbose:
                    display(input_g, subgrid, title=f"subgrid {i}")
                return subgrid
        logger.info(
            f"{'  ' * nesting_level}match_object_list_with_decision_rule no subgrid satisfies the rule"
        )
        return None

    return (state, solve)
