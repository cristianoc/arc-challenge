from typing import Optional, Tuple

import config
from bi_types import Examples, Match
from inpainting_match import inpainting_xform, mask_from_all_outputs
from logger import logger
from objects import Object, display


def frame_split_and_mirror_xform(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}split_and_mirror_xform examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    def get_split_masks(size: Tuple[int, int]) -> Tuple[Object, Object]:
        width, height = size
        top_right = Object.empty(size, background_color=1)
        bottom_left = Object.empty(size, background_color=1)
        for x in range(width):
            for y in range(height):
                if x - y >= 0:  # Split along the other diagonal
                    top_right[x, y] = 0
                else:
                    bottom_left[x, y] = 0
        if False:
            display(top_right, bottom_left, title=f"split_grid")
        return top_right, bottom_left

    def combine_grids(top_right: Object, bottom_left: Object) -> Object:
        size = top_right.size
        width, height = size
        combined = Object.empty(size)
        for x in range(width):
            for y in range(height):
                if x - y >= 0:
                    combined[x, y] = top_right[x, y]
                else:
                    combined[x, y] = bottom_left[x, y]
        return combined

    if mask_from_all_outputs(examples) is None:
        # abuse the function to check if the examples are valid for inpainting
        return None

    first_input = examples[0][0]
    mask_tr, mask_bl = get_split_masks(first_input.size)
    match_tr = inpainting_xform(
        examples,
        task_name + "_tr",
        nesting_level + 1,
        mask_tr,
        apply_mask_to_input=True,
        output_is_block=False,
    )
    match_bl = inpainting_xform(
        examples,
        task_name + "_bl",
        nesting_level + 1,
        mask_bl,
        apply_mask_to_input=True,
        output_is_block=False,
    )

    if match_tr is None or match_bl is None:
        return None

    state_tr, solve_tr = match_tr
    state_bl, solve_bl = match_bl

    def solve(input: Object) -> Optional[Object]:
        output_tr = solve_tr(input)
        output_bl = solve_bl(input)
        if output_tr is None or output_bl is None:
            return None
        if config.display_verbose:
            display(
                output_tr,
                output_bl,
                title="TestOutput tr+bl",
                left_title="tr",
                right_title="bl",
            )
        combined = combine_grids(output_tr, output_bl)
        if config.display_verbose:
            display(combined, title="Combined")
        return combined

    return (f"split_and_mirror({state_tr}, {state_bl})", solve)
