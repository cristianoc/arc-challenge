from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from bi_types import Config, Match
from cardinality_predicates import (
    CardinalityPredicate,
    find_cardinality_predicates,
    predicates_intersection,
)
from load_data import Example
from logger import logger
from objects import Object, display
from predict_size import get_largest_block_object
from symmetry import (
    NonPeriodicGridSymmetry,
    PeriodicGridSymmetry,
    fill_grid,
    find_non_periodic_symmetry_predicates,
    find_periodic_symmetry_predicates,
)
from visual_cortex import regularity_score


def is_inpainting_puzzle(examples: List[Example[Object]]) -> bool:
    # check the inpainting conditions on all examples
    for input, output in examples:
        if check_inpainting_conditions(input, output) is None:
            return False
    return True


def check_inpainting_conditions(input: Object, output: Object) -> Optional[int]:
    if input.size != output.size:
        largest_block_object = get_largest_block_object(input)
        if largest_block_object is None:
            return None
        if largest_block_object.size != output.size:
            return None
        if largest_block_object.width < 2 or largest_block_object.height < 2:
            return None
        return largest_block_object.main_color()

    # check if input has one more color than output
    if len(input.get_colors(allow_black=True)) - 1 != len(
        output.get_colors(allow_black=True)
    ):
        return None
    colors_only_in_input = set(input.get_colors(allow_black=True)) - set(
        output.get_colors(allow_black=True)
    )
    if len(colors_only_in_input) != 1:
        return None
    color = colors_only_in_input.pop()

    # check if input and output are the same except for the color
    for x in range(input.width):
        for y in range(input.height):
            if input[x, y] == color:
                continue
            if input[x, y] != output[x, y]:
                return None

    # check if output has high regularity score
    if regularity_score(output) >= Config.inpainting_regularity_score_threshold:
        return None

    return color


def update_mask(input: Object, output: Object, mask: Object) -> None:
    """
    Update the mask grid with 0s where the current object and the other object differ.
    """
    for x in range(output.width):
        for y in range(output.height):
            if input[x, y] != output[x, y]:
                mask[x, y] = 0


def mask_from_all_outputs(examples: List[Example[Object]]) -> Optional[Object]:
    """
    Find the mask for a set of examples where all examples have the same size.
    """
    if not examples:
        return None

    _, first_output = examples[0]
    mask = Object.empty(first_output.size, background_color=1)

    # Update the mask with each example
    for input, output in examples:
        color = check_inpainting_conditions(input, output)
        if color is None:
            return None
        if mask.size != output.size or input.size != output.size:
            return None
        update_mask(first_output, output, mask)
    return mask


# check that the color only in input is the same for all examples


def get_inpainting_color(examples: List[Example[Object]]) -> Optional[int]:
    color_only_in_input = None
    for input, output in examples:
        color = check_inpainting_conditions(input, output)
        if color is None:
            return None
        if color_only_in_input is None:
            color_only_in_input = color
        else:
            if color != color_only_in_input:
                logger.info(f"Color mismatch: {color} != {color_only_in_input}")
                return None
    return color_only_in_input


def compute_shared_symmetries(
    examples: List[Example[Object]],
    mask: Optional[Object],
    color_only_in_input: int,
    output_is_largest_block_object: bool,
) -> Optional[
    Tuple[
        NonPeriodicGridSymmetry,
        PeriodicGridSymmetry,
        List[CardinalityPredicate],
        int,
    ]
]:
    non_periodic_shared = None
    periodic_shared = None
    cardinality_shared: Optional[List[CardinalityPredicate]] = None

    for i, (input, output) in enumerate(examples):
        if Config.find_non_periodic_symmetry:
            if output_is_largest_block_object:
                non_periodic_symmetry_output = find_non_periodic_symmetry_predicates(
                    input, color_only_in_input
                )
            else:
                non_periodic_symmetry_output = find_non_periodic_symmetry_predicates(
                    output, color_only_in_input
                )
        else:
            non_periodic_symmetry_output = NonPeriodicGridSymmetry()
        if non_periodic_shared is None:
            non_periodic_shared = non_periodic_symmetry_output
        else:
            non_periodic_shared = non_periodic_shared.intersection(
                non_periodic_symmetry_output
            )

        if Config.find_cardinality_predicates:
            cardinality_shared_output = find_cardinality_predicates(output)
        else:
            cardinality_shared_output = []
        if cardinality_shared is None:
            cardinality_shared = cardinality_shared_output
        else:
            cardinality_shared = predicates_intersection(
                cardinality_shared, cardinality_shared_output
            )

        if Config.find_periodic_symmetry:
            periodic_symmetry_output = find_periodic_symmetry_predicates(
                output, color_only_in_input, mask
            )
        else:
            periodic_symmetry_output = PeriodicGridSymmetry()
        if periodic_shared is None:
            periodic_shared = periodic_symmetry_output
        else:
            periodic_shared = periodic_shared.intersection(periodic_symmetry_output)

        logger.info(
            f"#{i} From Output {non_periodic_symmetry_output} {periodic_symmetry_output} {cardinality_shared}"
        )

    if (
        periodic_shared is None
        or non_periodic_shared is None
        or cardinality_shared is None
        or color_only_in_input is None
    ):
        return None

    return (
        non_periodic_shared,
        periodic_shared,
        cardinality_shared,
        color_only_in_input,
    )


def apply_shared(
    input: Object,
    mask: Optional[Object],
    non_periodic_shared: NonPeriodicGridSymmetry,
    periodic_shared: PeriodicGridSymmetry,
    color: int,
) -> Object:
    filled_grid = fill_grid(
        input,
        mask,
        non_periodic_symmetry=non_periodic_shared,
        periodic_symmetry=periodic_shared,
        unknown=color,
    )
    return filled_grid


def check_equality_modulo_mask(
    grid1: Object, grid2: Object, mask: Optional[Object]
) -> bool:
    for x in range(grid1.width):
        for y in range(grid1.height):
            if grid1[x, y] != grid2[x, y] and (mask is None or mask[x, y] == 0):
                return False
    return True


def inpainting_xform_no_mask(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    return inpainting_xform(
        examples, task_name, nesting_level, mask=None, apply_mask_to_input=False
    )


def inpainting_xform_with_mask(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    mask = mask_from_all_outputs(examples)
    if mask is not None:
        if Config.display_verbose:
            display(mask, title=f"Mask")
    return inpainting_xform(
        examples, task_name, nesting_level, mask=mask, apply_mask_to_input=False
    )


def extract_largest_block(input: Object, filled_grid: Object) -> Object:
    largest_block_object = get_largest_block_object(input)
    assert largest_block_object is not None
    origin = largest_block_object.origin
    width, height = largest_block_object.size
    data = filled_grid._data

    # Extract the subgrid from output corresponding to the largest block object
    largest_block_object_data = data[
        origin[1] : origin[1] + height, origin[0] : origin[0] + width
    ]
    return Object(largest_block_object_data)


def inpainting_xform(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
    mask: Optional[Object],
    apply_mask_to_input: bool,
) -> Optional[Match[Object, Object]]:

    def apply_mask_to_filled_grid(filled_grid, input, mask, color_only_in_input):
        if mask is not None:
            if apply_mask_to_input:
                source = input
            else:
                first_output = examples[0][1]
                source = first_output
            filled_grid.add_object_in_place(
                source.apply_mask(mask, background_color=color_only_in_input),
                background_color=color_only_in_input,
            )
        return filled_grid

    color = get_inpainting_color(examples)
    if color is None:
        return None

    output_is_largest_block_object = any(
        input.size != output.size for input, output in examples
    ) if mask is None else False

    shared_symmetries = compute_shared_symmetries(
        examples, mask, color, output_is_largest_block_object
    )
    if shared_symmetries is None:
        return None
    (
        non_periodic_shared,
        periodic_shared,
        cardinality_shared,
        color_only_in_input,
    ) = shared_symmetries

    logger.info(
        f"inpainting_xform examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level} non_periodic_symmetries:{non_periodic_shared} cardinality_shared:{cardinality_shared}"
    )

    for i, (input, output) in enumerate(examples):
        filled_grid = fill_grid(
            input,
            mask,
            periodic_shared,
            non_periodic_shared,
            cardinality_shared,
            color_only_in_input,
        )

        filled_grid = apply_mask_to_filled_grid(
            filled_grid, input, mask, color_only_in_input
        )

        if output_is_largest_block_object:
            candidate_output = extract_largest_block(input, filled_grid)
            is_correct = candidate_output == output
        else:
            is_correct = check_equality_modulo_mask(filled_grid, output, mask)
        logger.info(
            f"#{i} Shared {non_periodic_shared} {periodic_shared} {cardinality_shared} is_correct: {is_correct}"
        )
        if Config.display_verbose and non_periodic_shared.dgm is not None:
            display(
                output,
                non_periodic_shared.dgm,
                title=f"Shared Diagonal Symmetry",
                left_title=f"output",
                right_title=f"diagonal symmetry",
            )

        if not is_correct:
            if Config.display_verbose:
                display(input, filled_grid, title=f"{is_correct} Shared Symm")
                if mask is not None:
                    display(mask, title=f"{is_correct} Mask")
        if is_correct:
            logger.info(f"#{i} Found correct solution using shared symmetries")
            pass
        else:
            break
    else:
        state = f"symmetry({non_periodic_shared}, {periodic_shared})"

        def solve_shared(input: Object) -> Object:
            input_filled = fill_grid(
                input,
                mask,
                periodic_shared,
                non_periodic_shared,
                cardinality_shared,
                color_only_in_input,
            )
            input_filled = apply_mask_to_filled_grid(
                input_filled, input, mask, color_only_in_input
            )
            logger.info(
                f"Test Shared {non_periodic_shared} {periodic_shared} {cardinality_shared}"
            )
            if Config.display_verbose:
                display(input, input_filled, title=f"Test Shared")

            if output_is_largest_block_object:
                return extract_largest_block(input, input_filled)
            else:
                return input_filled

        return (state, solve_shared)

    def solve_find_symmetry(input: Object) -> Optional[Object]:
        if Config.find_periodic_symmetry:
            periodic_symmetry_input = find_periodic_symmetry_predicates(
                input, color_only_in_input, mask
            )
        else:
            periodic_symmetry_input = PeriodicGridSymmetry()
        input_filled = fill_grid(
            input,
            mask,
            periodic_symmetry=periodic_symmetry_input,
            unknown=color_only_in_input,
        )
        if Config.find_non_periodic_symmetry:
            non_periodic_symmetry_input = find_non_periodic_symmetry_predicates(
                input, color_only_in_input
            )
        else:
            non_periodic_symmetry_input = NonPeriodicGridSymmetry()

        logger.info(
            f"Test NonShared {non_periodic_symmetry_input} {periodic_symmetry_input}"
        )

        input_filled = fill_grid(
            input,
            mask,
            periodic_symmetry=periodic_symmetry_input,
            non_periodic_symmetry=non_periodic_symmetry_input,
            unknown=color_only_in_input,
        )

        if Config.display_verbose:
            display(
                input,
                input_filled,
                title=f"Test: NonShared",
                left_title=f"input",
                right_title=f"filled",
            )

        if mask is not None:
            input_filled.add_object_in_place(
                input.apply_mask(mask, background_color=color_only_in_input),
                background_color=color_only_in_input,
            )
        # check if there are leftover unknown colors
        data = input_filled._data
        if np.any(data == color_only_in_input):
            logger.info(f"Test: Leftover unknown color: {color_only_in_input}")
            if Config.display_verbose:
                display(input, input_filled, title=f"Test: Leftover covered cells")
            return None
        if output_is_largest_block_object:
            return extract_largest_block(input, input_filled)
        else:
            return input_filled

    # Config.display_this_task = True
    state = "find_symmetry_for_each_input"
    return (state, solve_find_symmetry)
