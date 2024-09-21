from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Union

from load_data import Example
from objects import Object

GridAndObjects = Tuple[Object, List[Object]]


T1 = TypeVar("T1", bound=Union[Object, GridAndObjects])
T2 = TypeVar("T2", bound=Union[Object, GridAndObjects])

State = str

Primitive = Callable[[Object, str, int], Object]
Match = Tuple[State, Callable[[T1], Optional[T2]]]
Xform = Callable[[List[Example[T1]], str, int], Optional[Match[T2, T2]]]


@dataclass
class XformEntry(Generic[T1, T2]):
    xform: Xform[T1, T2]
    difficulty: int


class Config:
    task_name: str | None = None
    # task_name = "e9afcf9a.json"  # map 2 colored objects
    # task_name = "0dfd9992.json"
    # task_name = "05269061.json"
    # task_name = "47996f11.json"
    # task_name = "47996f11.json"
    # task_name = "4cd1b7b2.json"  # sudoku

    # task_name = "4aab4007.json"  # diagonal pattern with shared mask
    # task_name = "1e97544e.json"  # snake-like pattern
    # task_name = "f9d67f8b.json" # maybe a mistake in the task

    task_fractal = "8f2ea7aa.json"  # fractal expansion
    task_puzzle = "97a05b5b.json"  # puzzle-like, longest in DSL (59 lines)

    task_possibly_wrong_inpainting = "f9d67f8b.json"
    task_rays_top_left_inpainting = "73251a56.json"
    inpainting_regularity_score_threshold = 0.6
    # non-inpainting tasks present at regularty threshold 0.6
    non_inpainting_tasks: List[str] = [
        "bd4472b8.json",
        "8e5a5113.json",
        "62b74c02.json",
        "ef26cbf6.json",
        "c9f8e694.json",
        "e76a88a6.json",
        "63613498.json",
        "7c8af763.json",
        "2a5f8217.json",
    ]
    find_periodic_symmetry = True
    find_non_periodic_symmetry = True
    find_cardinality_predicates = True
    find_frame_rule = True

    blacklisted_tasks: List[str] = []
    blacklisted_tasks.extend(non_inpainting_tasks)
    whitelisted_tasks: List[str] = []
    whitelisted_tasks.append(task_puzzle)

    display_not_found = True
    display_verbose = False
    only_inpainting_puzzles = False

    only_simple_examples = False
    max_size = 9
    max_colors = 4

    only_complex_examples = True
    min_size = 23
    min_colors = 10

    find_xform = True
    find_matched_objects = False
    difficulty = 1000
    display_this_task = False
