from typing import List, Optional, Tuple

from bi_types import Example, GridAndObjects, Match, XformEntry
from logger import logger
from object_list_match import list_xforms
from objects import Object, display, display_multiple
from visual_cortex import extract_lattice_subgrids


def match_subgrids_in_lattice(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object]]:
    """
    Matches subgrids within lattice structures with the output.
    Some rule neds to be determined to determine which subgrid matches the output.
    This rule should be bease on features of the subgrid (such as the number of cells, colors etc)
    """
    logger.info(
        f"{'  ' * nesting_level}match_subgrids_in_lattice examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    for input_obj, output_obj in examples:
        input_subgrids = extract_lattice_subgrids(input_obj)

        if input_subgrids is None:
            return None

        # Flatten the list of lists into a single list of Object
        flattened_subgrids = [obj for sublist in input_subgrids for obj in sublist]
        display_multiple(flattened_subgrids, title="input_subgrids")

        # TODO: need to find a mapping between input subgrids and the output
        # check if the output is equal to one of the input subgrids
        if output_obj in input_subgrids:
            logger.info(
                f"{'  ' * nesting_level}match_subgrids_in_lattice found a match"
            )
        else:
            logger.info(f"{'  ' * nesting_level}match_subgrids_in_lattice no match")

    return None
