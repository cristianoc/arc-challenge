from enum import Enum, auto
from typing import List

from grid_data import RED, Object
from rule_based_selector import Embedding


class ColorFeatures(Enum):
    COLOR = auto()  # The color of the object (int)
    COLOR_UNIQUE = auto()  # Whether the color is unique (bool)
    MAX_RED_CELLS = auto()  # Whether the object has the maximum number of red cells (bool)
    MAX_NON_BACKGROUND_CELLS = auto()  # Whether the object has the maximum number of non-background cells (bool)


# Unpack SymmetryType members into the local scope
COLOR, COLOR_UNIQUE, MAX_RED_CELLS, MAX_NON_BACKGROUND_CELLS = ColorFeatures


# Functions to detect the features

def detect_color(object: Object, all_objects: List[Object]) -> int:
    return object.main_color

# Check if the color of the object distinguishes it from the others
def detect_color_unique(object: Object, all_objects: List[Object]) -> bool:
    color = object.main_color
    return all(obj.main_color != color for obj in all_objects if obj != object)


def detect_has_max_red_cells(object: Object, all_objects: List[Object]) -> bool:
    def count_red_cells(obj: Object) -> int:
        return sum(1 for j in range(obj.width) for i in range(obj.height) if obj.data[i][j] == RED)
    num_red_cells = count_red_cells(object)
    print(f"red_color: {RED} num_red_cells: {num_red_cells} object: {object}")
    return all(count_red_cells(obj) < num_red_cells for obj in all_objects if obj != object)

def detect_has_max_non_background_cells(object: Object, all_objects: List[Object]) -> bool:
    background_color = object.main_color
    def count_non_background_cells(obj: Object) -> int:
        return sum(1 for j in range(obj.width) for i in range(obj.height) if obj.data[i][j] != background_color)
    num_non_background_cells = count_non_background_cells(object)
    print(f"background_color: {background_color} num_non_background_cells: {num_non_background_cells} object: {object}")
    return all(count_non_background_cells(obj) < num_non_background_cells for obj in all_objects if obj != object)

def detect_color_features(object: Object, all_objects: List[Object]) -> Embedding:
    embedding: Embedding = {}
    embedding[COLOR.name] = detect_color(object, all_objects)
    embedding[COLOR_UNIQUE.name] = detect_color_unique(object, all_objects)
    embedding[MAX_RED_CELLS.name] = detect_has_max_red_cells(object, all_objects)
    embedding[MAX_NON_BACKGROUND_CELLS.name] = detect_has_max_non_background_cells(object, all_objects)
    return embedding
    
