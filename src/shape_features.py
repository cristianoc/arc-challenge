from enum import Enum, auto
from typing import List

from grid_types import logger
from rule_based_selector import Features
from grid_data import Object

class ColorFeatures(Enum):
    LARGEST_SIZE = (
        auto()
    )  # Whether the object has the largest size of the objects (bool)
    SMALLEST_SIZE = (
        auto()
    )  # Whether the object has the smallest size of the objects (bool)
    # Whether the object has the maximum number of non-trivial subobjects (bool)
    MAX_NUMBER_NONTRIVIAL_SUBOBJECTS = auto()


LARGEST_SIZE, SMALLEST_SIZE, MAX_NUMBER_NONTRIVIAL_SUBOBJECTS = ColorFeatures

# Functions to detect the features


def detect_has_largest_size(object: Object, all_objects: List[Object]) -> bool:
    if len(all_objects) == 1:
        return True
    max_size = max(obj.width * obj.height for obj in all_objects if obj != object)
    return object.width * object.height > max_size


def detect_has_smallest_size(object: Object, all_objects: List[Object]) -> bool:
    if len(all_objects) == 1:
        return True
    min_size = min(obj.width * obj.height for obj in all_objects if obj != object)
    return object.width * object.height < min_size


def detect_has_max_number_nontrivial_subobjects(
    object: Object, all_objects: List[Object]
) -> bool:
    """
    Detects if the given object has the maximum number of nontrivial subobjects among all provided objects.

    Parameters:
        object (Object): The object to check for nontrivial subobjects.
        all_objects (List[Object]): List of all objects to compare against.

    Returns:
        bool: True if the given object has the maximum number of nontrivial subobjects, False otherwise.
    """

    def is_trivial(subobj: Object) -> bool:
        return subobj.size <= (2, 2)

    def obj_count_nontrivial_subobjects(obj: Object) -> int:
        subobjects = Object(obj.data).detect_objects(diagonals=False)
        num_nontrivial_subobjects = sum(
            1 for o in subobjects if o.size != obj.size and not is_trivial(o)
        )
        return num_nontrivial_subobjects

    num_nontrivial_subobjects = obj_count_nontrivial_subobjects(object)
    res = all(
        obj_count_nontrivial_subobjects(obj) < num_nontrivial_subobjects
        for obj in all_objects
        if obj != object
    )
    logger.debug(
        f"num_nontrivial_subobjects: {num_nontrivial_subobjects} is_max: {res}"
    )
    return res


def detect_shape_features(
    object: Object, all_objects: List[Object], level: int
) -> Features:
    features: Features = {
        LARGEST_SIZE.name: detect_has_largest_size(object, all_objects),
        SMALLEST_SIZE.name: detect_has_smallest_size(object, all_objects),
        MAX_NUMBER_NONTRIVIAL_SUBOBJECTS.name: detect_has_max_number_nontrivial_subobjects(
            object, all_objects
        ),
    }
    return features
