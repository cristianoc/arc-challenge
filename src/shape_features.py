from enum import Enum, auto
from typing import List

from grid_data import Object
from rule_based_selector import Embedding


class ColorFeatures(Enum):
    LARGEST_SIZE = auto()  # Whether the object has the largest size of the objects (bool)
    SMALLEST_SIZE = auto()  # Whether the object has the smallest size of the objects (bool)


LARGEST_SIZE, SMALLEST_SIZE = ColorFeatures


# Functions to detect the features

def detect_has_largest_size(object: Object, all_objects: List[Object]) -> bool:
    max_size = max(
        obj.width * obj.height for obj in all_objects if obj != object)
    return object.width * object.height > max_size


def detect_has_smallest_size(object: Object, all_objects: List[Object]) -> bool:
    min_size = min(
        obj.width * obj.height for obj in all_objects if obj != object)
    return object.width * object.height < min_size


def detect_shape_features(object: Object, all_objects: List[Object]) -> Embedding:
    embedding: Embedding = {}
    embedding[LARGEST_SIZE.name] = detect_has_largest_size(object, all_objects)
    embedding[SMALLEST_SIZE.name] = detect_has_smallest_size(
        object, all_objects)
    return embedding
