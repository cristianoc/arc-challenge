from typing import List, Optional

from bi_types import Example, Match
from logger import logger
from match_object_list_to_object import match_object_list_with_decision_rule
from objects import Object
from visual_cortex import extract_lattice_subgrids


def match_subgrids_in_lattice(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    """
    Matches subgrids within lattice structures with the output.
    Some rule neds to be determined to determine which subgrid matches the output.
    This rule should be bease on features of the subgrid (such as the number of cells, colors etc)
    """
    logger.info(
        f"{'  ' * nesting_level}match_subgrids_in_lattice examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    def get_objects(input: Object) -> List[Object]:
        input_subgrids = extract_lattice_subgrids(input)
        if input_subgrids is None:
            return []
        flattened_subgrids = [
            Object(obj._data)  # reset origin to (0,0)
            for sublist in input_subgrids
            for obj in sublist
        ]
        return flattened_subgrids

    examples2 = []
    for input_obj, output_obj in examples:
        flattened_subgrids = get_objects(input_obj)
        if flattened_subgrids is None:
            return None
        examples2.append((flattened_subgrids, output_obj))
    return match_object_list_with_decision_rule(
        examples2, task_name, nesting_level + 1, minimal=True, get_objects=get_objects
    )
