from typing import List, Optional

from bi_types import Example, Match
from logger import logger
from matched_objects import ObjectMatch, detect_common_features
from objects import Object
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

    object_matches: List[ObjectMatch] = []
    for input_obj, output_obj in examples:
        input_subgrids = extract_lattice_subgrids(input_obj)

        if input_subgrids is None:
            return None

        # Flatten the list of lists into a single list of Object
        flattened_subgrids = [obj for sublist in input_subgrids for obj in sublist]

        try:
            index = flattened_subgrids.index(output_obj)
            logger.info(
                f"{'  ' * nesting_level}match_subgrids_in_lattice found a match at index {index}"
            )
            object_matches.append(
                ObjectMatch(input_objects=flattened_subgrids, matched_index=index)
            )
        except ValueError:
            logger.info(f"{'  ' * nesting_level}match_subgrids_in_lattice no match")
    common_decision_rule, features_used = detect_common_features(object_matches, 3)
    if common_decision_rule is not None:
        logger.info(
            f"{'  ' * nesting_level}match_subgrids_in_lattice common_decision_rule:{common_decision_rule} features_used:{features_used}"
        )

    return None
