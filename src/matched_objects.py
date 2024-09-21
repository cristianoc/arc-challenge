from typing import List, NamedTuple, Optional

import numpy as np

from color_features import detect_color_features
from load_data import Example
from logger import logger
from objects import Object
from rule_based_selector import DecisionRule, Features, select_object_minimal
from shape_features import detect_shape_features
from symmetry_features import detect_symmetry_features
from visual_cortex import find_rectangular_objects


class ObjectMatch(NamedTuple):
    """
    A match between an input object and an output object.
    """

    input_objects: List[Object]
    matched_index: int


def check_grid_satisfies_rule(
    obj: Object, all_objects: List[Object], decision_rule: DecisionRule
) -> bool:
    """
    Check if the given grid satisfies the specified decision rule.
    """
    features = {}

    shape_features = detect_shape_features(obj, all_objects)
    features.update(shape_features)  # Flattening the nested dictionary

    color_features = detect_color_features(obj, all_objects)
    features.update(color_features)  # Flattening the nested dictionary

    symmetry_features = detect_symmetry_features(obj)
    features.update(symmetry_features)  # Flattening the nested dictionary

    feature_rule = DecisionRule(features)

    return decision_rule.is_subset(feature_rule)


def detect_common_features(
    matched_objects: List[ObjectMatch], difficulty: int, minimal: bool
):
    def detect_common_symmetry_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for match in matched_objects:
            embeddings = [detect_symmetry_features(obj) for obj in match.input_objects]
            decision_rule = select_object_minimal(
                embeddings, match.matched_index, minimal
            )
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Symmetry): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Symmetry)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_color_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for match in matched_objects:
            embeddings = [
                detect_color_features(obj, match.input_objects)
                for obj in match.input_objects
            ]
            decision_rule = select_object_minimal(
                embeddings, match.matched_index, minimal
            )
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Color): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Color)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_shape_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for match in matched_objects:
            embeddings = [
                detect_shape_features(obj, match.input_objects)
                for obj in match.input_objects
            ]
            decision_rule = select_object_minimal(
                embeddings, match.matched_index, minimal
            )
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Shape): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Shape)")
                common_decision_rule = None
                break
        return common_decision_rule

    common_decision_rule = None
    features_used = None

    # Try detecting common features in the order of shape, color, and symmetry

    if common_decision_rule is None and difficulty >= 1:
        common_decision_rule = detect_common_shape_features()
        features_used = "Shape"

    if common_decision_rule is None and difficulty >= 2:
        common_decision_rule = detect_common_color_features()
        features_used = "Color"

    if common_decision_rule is None and difficulty >= 3:
        common_decision_rule = detect_common_symmetry_features()
        features_used = "Symmetry"

    return common_decision_rule, features_used


def find_matched_objects(
    examples: List[Example], task_type: str
) -> Optional[List[ObjectMatch]]:
    """
    Identifies and returns a list of matched input objects that correspond to the output objects
    in the given examples. For each example, it detects candidate objects in the input grid
    and matches them with the output grid based on size and data. If all examples have a match,
    the function returns the list of matched objects; otherwise, it returns None.

    Args:
        examples: A list of examples, each containing an input and output grid.
        task_type: A string indicating the type of task (e.g., 'train' or 'test').

    Returns:
        A list of ObjectMatch instances if matches are found for all examples, otherwise None.
    """

    def get_objects(input: Object) -> List[Object]:
        num_colors_input = len(input.get_colors(allow_black=True))
        return find_rectangular_objects(input, allow_multicolor=num_colors_input > 1)

    def candidate_objects_for_matching(input: Object, output: Object) -> List[Object]:
        """
        Detects objects in the input grid that are candidates for matching the output grid.
        """
        if output.has_frame():
            # If the output is a frame, detect objects in the input as frames
            logger.debug("  Output is a frame")
        return get_objects(input)

    def find_matching_input_object(
        input_objects: List[Object], output: Object
    ) -> Optional[int]:
        for i, io in enumerate(input_objects):
            if io.size == output.size and np.array_equal(io._data, output._data):
                logger.debug(f"  Input object matching output: {io}")
                return i
        return None

    def get_matched_objects(examples: List[Example]) -> Optional[List[ObjectMatch]]:
        matched_objects: List[ObjectMatch] = []

        for input, output in examples:
            logger.info(f"  {task_type} {input.size} -> {output.size}")
            logger.info(f"  {task_type} {input.size} -> {output.size}")

            input_objects = candidate_objects_for_matching(input, output)
            matched_object_index = find_matching_input_object(input_objects, output)

            if matched_object_index is not None:
                matched_objects.append(ObjectMatch(input_objects, matched_object_index))

        return matched_objects if len(matched_objects) == len(examples) else None

    matched_objects = get_matched_objects(examples)
    return matched_objects


def minimize_common_features(
    common_decision_rule: DecisionRule, matched_objects: List[ObjectMatch]
):
    """
    Minimizes the common features by trying to reduce the number of features one by one.

    Args:
        common_decision_rule: The initial set of common features discovered.
        matched_objects: The list of matched input-output object pairs.

    Returns:
        A minimized set of features that still correctly classifies the object.
    """
    # Start with the full set of common features
    minimized_rule = common_decision_rule

    # Get all the features in the rule
    features = list(minimized_rule.features)

    for feature in features:
        # Try removing one feature at a time
        temp_rule = minimized_rule.without_feature(feature)

        # Check if the reduced set still correctly classifies the objects
        all_match = True
        for match in matched_objects:
            embeddings = [detect_symmetry_features(obj) for obj in match.input_objects]
            decision_rule = select_object_minimal(
                embeddings, match.matched_index, minimal=False
            )
            if decision_rule is None or not temp_rule.is_subset(decision_rule):
                all_match = False
                break

        # If removing the feature doesn't affect classification, keep it removed
        if all_match:
            minimized_rule = temp_rule

    return minimized_rule


def handle_matched_objects(
    examples, task_name, task_type, set, current_difficulty
) -> bool:
    should_continue = False

    # Check if the input objects can be matched to the output objects
    logger.debug(f"Checking common features for {task_name} {set}")
    matched_objects = find_matched_objects(examples, task_type)
    if matched_objects:
        # If the input objects can be matched to the output objects, try to detect common features
        # to determine the correct object to pick
        logger.debug(
            f"XXX Matched {len(matched_objects)}/{len(examples)} {task_name} {set}"
        )
        common_decision_rule, features_used = detect_common_features(
            matched_objects, current_difficulty, minimal=False
        )
        if common_decision_rule:
            logger.info(
                f"Common decision rule ({features_used}): {common_decision_rule}"
            )
            # TODO: update the number if this code is resurrected
            #  num_correct += 1
            should_continue = True
        else:
            logger.warning(f"Could not find common decision rule for {task_name} {set}")
    return should_continue
