from typing import Optional

from logger import logger
from match_n_objects_with_output import Examples, Match
from objects import Object, display, display_multiple
from visual_cortex import find_largest_frame2, find_largest_frame, find_rectangular_objects
import config


def match_split_with_frame(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:

    logger.info(
        f"{'  ' * nesting_level}match_split_with_frame examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    for input, output in examples:
        frame = find_largest_frame2(output, None)
        if frame is None:
            return None
    return None

    for input, output in examples:

        print(f"output: {output}")
        frame = find_largest_frame2(output, None)
        if frame is None:
            return None
        g = output.crop(*frame)
        display(g, title="g")

        width = input.width
        height = input.height
        (top, left, bottom, right) = frame
        print("frame", frame)
        # split the grid into 3x3 subgrids with the frame in the center
        for i in range(3):
            for j in range(3):
                if i == 0:
                    sub_left = 0
                    sub_right = left
                elif i == 2:
                    sub_left = left
                    sub_right = right
                else:
                    sub_left = right
                    sub_right = width
                if j == 0:
                    sub_top = 0
                    sub_bottom = top
                elif j == 2:
                    sub_top = top
                    sub_bottom = bottom
                else:
                    sub_top = bottom
                    sub_bottom = height
                subgrid = output.crop(sub_top, sub_left, sub_bottom, sub_right)

                # Process the subgrid here (e.g., add to a list, analyze, etc.)
                if i==1 and j==1:
                    display(subgrid, title=f"Subgrid ({i}, {j})")

        # Continue with the rest of your logic here

    def solver(input: Object) -> Optional[Object]:
        return None

    state = f"match_split_with_frame"
    return None
