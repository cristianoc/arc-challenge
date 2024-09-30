from typing import List, Optional

import config
import xforms
from find_xform import find_xform_for_examples
from logger import logger
from match_n_objects_with_output import Examples, Match
from objects import Object, display, display_multiple
from visual_cortex import find_largest_frame, Frame
import numpy as np


def extract_subgrid(grid: Object, frame: Frame, i: int, j: int) -> Object:
    width = grid.width
    height = grid.height
    (top, left, bottom, right) = frame
    if i == 0:
        sub_left = 0
        sub_right = left - 1
    elif i == 1:
        sub_left = left
        sub_right = right
    else:
        sub_left = right + 1
        sub_right = width - 1
    if j == 0:
        sub_top = 0
        sub_bottom = top - 1
    elif j == 1:
        sub_top = top
        sub_bottom = bottom
    else:
        sub_top = bottom + 1
        sub_bottom = height - 1
    out_subgrid = grid.crop(sub_top, sub_left, sub_bottom, sub_right)
    return out_subgrid


def match_split_with_frame(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:

    logger.info(
        f"{'  ' * nesting_level}match_split_with_frame examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    subtasks: List[List[Examples[Object, Object]]] = [
        [[] for _ in range(3)] for _ in range(3)
    ]

    for input, output in examples:
        if input.size != output.size:
            return None
        frame = find_largest_frame(output, color=None, check_precise=True)
        if frame is None:
            return None

        # split the grid into 3x3 subgrids with the frame in the center
        for i in range(3):
            for j in range(3):
                in_subgrid = extract_subgrid(input, frame, i, j)
                out_subgrid = extract_subgrid(output, frame, i, j)
                if in_subgrid.width == 0 or in_subgrid.height == 0:
                    return None
                # display(in_subgrid, out_subgrid, title=f"Subgrid ({i}, {j})")
                subtasks[i][j].append(
                    (
                        in_subgrid,
                        out_subgrid,
                    )
                )
                # display(subgrid, title=f"Subgrid ({i}, {j})")

    matches = np.empty((3, 3), dtype=object)

    for i in range(3):
        for j in range(3):
            sub_examples = subtasks[i][j]
            if config.display_verbose:
                display_multiple(sub_examples, title=f"Subtask ({i}, {j})")
            # TODO: find a way to pass the color from the frame object to the xforms
            match = find_xform_for_examples(
                xforms.gridxforms, sub_examples, task_name, nesting_level + 1
            )
            if match is not None:
                matches[i][j] = match
            else:
                pass
                # return None

    def solver(input: Object) -> Optional[Object]:
        frame = None
        # TODO: need to find the frame in the input
        # One could consider a frama mask determined by an integer n
        # when n=0, it's just the 4 points at the corners
        # when n=1, it's in addition one point before and after (as in the example)
        # when n=2, it's in addition two points before and after, and so on.
        # So one could look at the outputs and the inputs, and see that the frames have the same origin and size,
        # and the output frame is complete while the input frame has n=2
        # Also, the input frame is multicolored.
        if frame is None:
            return None
        output = Object.empty(input.size)
        for i in range(3):
            for j in range(3):
                in_subgrid = extract_subgrid(input, frame, i, j)
                state_, solver_ = matches[i][j]
                out_subgrid = solver_(in_subgrid)
                if out_subgrid is None:
                    return None
                out_subgrid.origin = in_subgrid.origin
                output.add_object_in_place(out_subgrid)
        return output

    state = f"match_split_with_frame"
    return state, solver
