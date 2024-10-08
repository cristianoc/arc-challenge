from enum import Enum, auto
from typing import List

from grid_types import RED
from logger import logger
from objects import Object
from rule_based_selector import Features


class ColorFeatures(Enum):
    COLOR = auto()  # The color of the object (int)
    COLOR_UNIQUE = auto()  # Whether the color is unique (bool)
    MAX_RED_CELLS = (
        auto()
    )  # Whether the object has the maximum number of red cells (bool)
    MAX_NON_BACKGROUND_CELLS = (
        auto()
    )  # Whether the object has the maximum number of non-background cells (bool)
    MIN_NON_BACKGROUND_CELLS = (
        auto()
    )  # Whether the object has the minimum number of non-background cells (bool)
    # TODO: this should not be hardcoded, at the very least the number of colors should be determined from the examples
    HAS_ONE_COLOR = auto()  # Whether the object has only one color (bool)


# Unpack ColorFeatures members into the local scope
(
    COLOR,
    COLOR_UNIQUE,
    MAX_RED_CELLS,
    MAX_NON_BACKGROUND_CELLS,
    MIN_NON_BACKGROUND_CELLS,
    HAS_ONE_COLOR,
) = ColorFeatures

# Functions to detect the features


def detect_color(object: Object, all_objects: List[Object]) -> int:
    return object.main_color()


# Check if the color of the object distinguishes it from the others
def detect_color_unique(object: Object, all_objects: List[Object]) -> bool:
    color = object.main_color()
    return all(obj.main_color() != color for obj in all_objects if obj != object)


def detect_has_max_red_cells(object: Object, all_objects: List[Object]) -> bool:
    def count_red_cells(obj: Object) -> int:
        return sum(
            1
            for j in range(obj.width)
            for i in range(obj.height)
            if obj[j, i] == RED
        )

    num_red_cells = count_red_cells(object)
    logger.debug(f"red_color: {RED} num_red_cells: {num_red_cells} object: {object}")
    return all(
        count_red_cells(obj) < num_red_cells for obj in all_objects if obj != object
    )


def detect_has_max_non_background_cells(
    object: Object, all_objects: List[Object]
) -> bool:
    background_color = object.main_color()

    def count_non_background_cells(obj: Object) -> int:
        return sum(
            1
            for j in range(obj.width)
            for i in range(obj.height)
            if obj[j, i] != background_color
        )

    num_non_background_cells = count_non_background_cells(object)
    logger.debug(
        f"background_color: {background_color} num_non_background_cells: {num_non_background_cells} object: {object}"
    )
    return all(
        count_non_background_cells(obj) < num_non_background_cells
        for obj in all_objects
        if obj != object
    )


def detect_has_min_non_background_cells(
    object: Object, all_objects: List[Object]
) -> bool:
    background_color = object.main_color()

    def count_non_background_cells(obj: Object) -> int:
        return sum(
            1
            for j in range(obj.width)
            for i in range(obj.height)
            if obj[j, i] != background_color
        )

    num_non_background_cells = count_non_background_cells(object)
    logger.debug(
        f"background_color: {background_color} num_non_background_cells: {num_non_background_cells} object: {object}"
    )
    return all(
        count_non_background_cells(obj) > num_non_background_cells
        for obj in all_objects
        if obj != object
    )


def detect_has_one_color(object: Object, all_objects: List[Object]) -> bool:
    colors = object.get_colors(allow_black=False)
    num_colors = len(colors)
    return num_colors == 1


def detect_color_features(object: Object, all_objects: List[Object]) -> Features:
    features: Features = {}
    features[COLOR.name] = detect_color(object, all_objects)
    features[COLOR_UNIQUE.name] = detect_color_unique(object, all_objects)
    features[MAX_RED_CELLS.name] = detect_has_max_red_cells(object, all_objects)
    features[MAX_NON_BACKGROUND_CELLS.name] = detect_has_max_non_background_cells(
        object, all_objects
    )
    features[MIN_NON_BACKGROUND_CELLS.name] = detect_has_min_non_background_cells(
        object, all_objects
    )
    features[HAS_ONE_COLOR.name] = detect_has_one_color(object, all_objects)
    return features
