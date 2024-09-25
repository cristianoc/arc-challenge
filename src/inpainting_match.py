from typing import List, Optional, Tuple

import numpy as np

import config
from bi_types import Examples, Match
from cardinality_predicates import (
    CardinalityPredicate,
    find_cardinality_predicates,
    predicates_intersection,
)
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


class InpaintingTransform:
    def __init__(
        self,
        examples: Examples[Object, Object],
        task_name: str,
        nesting_level: int,
        mask: Optional[Object],
        apply_mask_to_input: bool,
        output_is_block: bool,
    ):
        self.examples = examples
        self.task_name = task_name
        self.nesting_level = nesting_level
        self.mask = mask
        self.apply_mask_to_input = apply_mask_to_input
        self.output_is_block = output_is_block

        self.color: Optional[int] = None
        self.output_is_largest_block_object = False
        self.shared_symmetries: Optional[
            Tuple[
                NonPeriodicGridSymmetry,
                PeriodicGridSymmetry,
                List[CardinalityPredicate],
                int,
            ]
        ] = None
        self.state = ""

    def analyze(self) -> bool:
        """First phase: Analyze examples and determine strategy."""
        self.color = self.get_inpainting_color()
        if self.color is None:
            return False

        self.output_is_largest_block_object = (
            self.determine_output_is_largest_block_object()
        )

        shared_symmetries_result = self.compute_shared_symmetries()
        if shared_symmetries_result is None:
            return False

        self.shared_symmetries, num_correct = shared_symmetries_result
        if not self.check_shared_symmetries(num_correct):
            self.shared_symmetries = None

        if self.shared_symmetries is not None:
            self.state = (
                f"symmetry({self.shared_symmetries[0]}, {self.shared_symmetries[1]})"
            )
        else:
            if num_correct == len(self.examples):
                self.state = "find_symmetry_for_each_input"
            else:
                return False

        return True

    def solve(self, input_obj: Object) -> Optional[Object]:
        """Second phase: Apply the determined strategy to solve."""
        if self.shared_symmetries is not None:
            return self.solve_shared(input_obj)
        else:
            return self.solve_find_symmetry(input_obj)

    def get_inpainting_color(self) -> Optional[int]:
        """Determine the color used only in inputs."""
        color_only_in_input = None
        for input_obj, output_obj in self.examples:
            color = check_inpainting_conditions(
                input_obj, output_obj, self.output_is_block
            )
            if color is None:
                return None
            if color_only_in_input is None:
                color_only_in_input = color
            elif color != color_only_in_input:
                logger.info(f"Color mismatch: {color} != {color_only_in_input}")
                return None
        return color_only_in_input

    def determine_output_is_largest_block_object(self) -> bool:
        """Determine if the output is the largest block object."""
        if self.output_is_block:
            return (
                any(
                    input_obj.size != output_obj.size
                    for input_obj, output_obj in self.examples
                )
                if self.mask is None
                else False
            )
        else:
            return False

    def compute_shared_symmetries(
        self,
    ) -> Optional[
        Tuple[
            Tuple[
                NonPeriodicGridSymmetry,
                PeriodicGridSymmetry,
                List[CardinalityPredicate],
                int,
            ],
            int,
        ]
    ]:
        """
        Compute shared symmetries (non-periodic, periodic, cardinality) across all examples
        and count how many examples match.
        """
        assert self.color is not None

        num_correct = 0
        non_periodic_shared = None
        periodic_shared = None
        cardinality_shared: Optional[List[CardinalityPredicate]] = None

        for i, (input_obj, output_obj) in enumerate(self.examples):
            # Non-periodic symmetry
            if config.find_non_periodic_symmetry:
                if self.output_is_largest_block_object:
                    non_periodic_symmetry_output = (
                        find_non_periodic_symmetry_predicates(input_obj, self.color)
                    )
                else:
                    non_periodic_symmetry_output = (
                        find_non_periodic_symmetry_predicates(output_obj, self.color)
                    )
            else:
                non_periodic_symmetry_output = NonPeriodicGridSymmetry()
            non_periodic_shared = (
                non_periodic_symmetry_output
                if non_periodic_shared is None
                else non_periodic_shared.intersection(non_periodic_symmetry_output)
            )

            # Cardinality predicates
            if config.find_cardinality_predicates:
                cardinality_shared_output = find_cardinality_predicates(output_obj)
            else:
                cardinality_shared_output = []
            cardinality_shared = (
                cardinality_shared_output
                if cardinality_shared is None
                else predicates_intersection(
                    cardinality_shared, cardinality_shared_output
                )
            )

            # Periodic symmetry
            if config.find_periodic_symmetry:
                periodic_symmetry_output = find_periodic_symmetry_predicates(
                    output_obj, self.color, self.mask
                )
            else:
                periodic_symmetry_output = PeriodicGridSymmetry()
            periodic_shared = (
                periodic_symmetry_output
                if periodic_shared is None
                else periodic_shared.intersection(periodic_symmetry_output)
            )

            output_symmetries = (
                non_periodic_symmetry_output,
                periodic_symmetry_output,
                cardinality_shared_output,
                self.color,
            )
            is_correct = check_correctness(
                self.examples,
                input_obj,
                output_obj,
                output_symmetries,
                self.mask,
                self.output_is_largest_block_object,
                self.apply_mask_to_input,
            )
            if is_correct:
                num_correct += 1

            logger.info(
                f"#{i} From Output {non_periodic_symmetry_output} {periodic_symmetry_output} {cardinality_shared} is_correct: {is_correct}"
            )

        if (
            periodic_shared is None
            or non_periodic_shared is None
            or cardinality_shared is None
            or self.color is None
        ):
            return None

        return (
            (
                non_periodic_shared,
                periodic_shared,
                cardinality_shared,
                self.color,
            ),
            num_correct,
        )

    def check_shared_symmetries(self, num_correct: int) -> bool:
        """Check if shared symmetries lead to correct outputs."""
        assert self.shared_symmetries is not None
        assert self.color is not None

        use_shared_symmetries_in_test = True
        for i, (input_obj, output_obj) in enumerate(self.examples):
            filled_grid = fill_grid(
                input_obj,
                self.mask,
                periodic_symmetry=self.shared_symmetries[1],
                non_periodic_symmetry=self.shared_symmetries[0],
                cardinality_predicates=self.shared_symmetries[2],
                unknown=self.shared_symmetries[3],
            )

            filled_grid = apply_mask_to_filled_grid(
                self.examples,
                filled_grid,
                input_obj,
                self.mask,
                self.color,
                self.apply_mask_to_input,
            )

            if self.output_is_largest_block_object:
                candidate_output = extract_largest_block(input_obj, filled_grid)
                is_correct = candidate_output == output_obj
            else:
                is_correct = check_equality_modulo_mask(
                    filled_grid, output_obj, self.mask
                )
            logger.info(
                f"#{i} Shared {self.shared_symmetries[0]} {self.shared_symmetries[1]} {self.shared_symmetries[2]} is_correct: {is_correct}"
            )
            if config.display_verbose and self.shared_symmetries[0].dgm is not None:
                display(
                    output_obj,
                    self.shared_symmetries[0].dgm,
                    title=f"Shared Diagonal Symmetry",
                    left_title=f"output",
                    right_title=f"diagonal symmetry",
                )

            if not is_correct:
                if config.display_verbose:
                    display(input_obj, filled_grid, title=f"{is_correct} Shared Symm")
                    if self.mask is not None:
                        display(self.mask, title=f"{is_correct} Mask")
                if num_correct == len(self.examples):
                    logger.info(
                        f"#{i} Shared symmetries are not correct, but each symmetry is correct"
                    )
                    use_shared_symmetries_in_test = False
                    break
                else:
                    logger.info(f"#{i} Shared symmetries are not correct")
                    return False

        return use_shared_symmetries_in_test

    def solve_shared(self, input_obj: Object) -> Optional[Object]:
        """Solve using shared symmetries."""
        assert self.shared_symmetries is not None
        assert self.color is not None

        input_filled = fill_grid(
            input_obj,
            self.mask,
            periodic_symmetry=self.shared_symmetries[1],
            non_periodic_symmetry=self.shared_symmetries[0],
            cardinality_predicates=self.shared_symmetries[2],
            unknown=self.shared_symmetries[3],
        )
        input_filled = apply_mask_to_filled_grid(
            self.examples,
            input_filled,
            input_obj,
            self.mask,
            self.color,
            self.apply_mask_to_input,
        )
        logger.info(
            f"Test Shared {self.shared_symmetries[0]} {self.shared_symmetries[1]} {self.shared_symmetries[2]}"
        )
        if config.display_verbose:
            display(input_obj, input_filled, title=f"Test Shared")

        if self.output_is_largest_block_object:
            return extract_largest_block(input_obj, input_filled)
        else:
            return input_filled

    def solve_find_symmetry(self, input_obj: Object) -> Optional[Object]:
        """Solve by finding symmetry for each input."""
        assert self.color is not None

        if config.find_periodic_symmetry:
            periodic_symmetry_input = find_periodic_symmetry_predicates(
                input_obj, self.color, self.mask
            )
        else:
            periodic_symmetry_input = PeriodicGridSymmetry()
        input_filled = fill_grid(
            input_obj,
            self.mask,
            periodic_symmetry=periodic_symmetry_input,
            unknown=self.color,
        )
        if config.find_non_periodic_symmetry:
            non_periodic_symmetry_input = find_non_periodic_symmetry_predicates(
                input_obj, self.color
            )
        else:
            non_periodic_symmetry_input = NonPeriodicGridSymmetry()

        logger.info(
            f"Test NonShared {non_periodic_symmetry_input} {periodic_symmetry_input}"
        )

        input_filled = fill_grid(
            input_obj,
            self.mask,
            periodic_symmetry=periodic_symmetry_input,
            non_periodic_symmetry=non_periodic_symmetry_input,
            unknown=self.color,
        )

        if config.display_verbose:
            display(
                input_obj,
                input_filled,
                title=f"Test: NonShared",
                left_title=f"input",
                right_title=f"filled",
            )

        if self.mask is not None:
            input_filled.add_object_in_place(
                input_obj.apply_mask(self.mask, background_color=self.color),
                background_color=self.color,
            )
        # Check if there are leftover unknown colors
        data = input_filled._data
        if np.any(data == self.color):
            logger.info(f"Test: Leftover unknown color: {self.color}")
            if config.display_verbose:
                display(input_obj, input_filled, title=f"Test: Leftover covered cells")
            return None
        if self.output_is_largest_block_object:
            return extract_largest_block(input_obj, input_filled)
        else:
            return input_filled


def is_inpainting_puzzle(
    examples: Examples[Object, Object], output_is_block: bool
) -> bool:
    # Check if all examples satisfy inpainting conditions. Returns False if any example does not.
    for input, output in examples:
        if check_inpainting_conditions(input, output, output_is_block) is None:
            return False
    return True


def check_inpainting_conditions(
    input_obj: Object, output_obj: Object, output_is_block: bool
) -> Optional[int]:
    """
    Validate input-output size, color match, and overall similarity with allowed discrepancies.
    """
    if input_obj.size != output_obj.size:
        if output_is_block:
            largest_block_object = get_largest_block_object(input_obj)
            if (
                largest_block_object is None
                or largest_block_object.size != output_obj.size
                or largest_block_object.width < 2
                or largest_block_object.height < 2
            ):
                return None
            return largest_block_object.main_color()
        else:
            return None
    else:
        if output_is_block:
            return None

    # Check if input has one more color than output
    input_colors = input_obj.get_colors(allow_black=True)
    output_colors = output_obj.get_colors(allow_black=True)
    if len(input_colors) - 1 != len(output_colors):
        return None
    colors_only_in_input = set(input_colors) - set(output_colors)
    if len(colors_only_in_input) != 1:
        return None
    color = colors_only_in_input.pop()

    # Check if input and output are the same except for the color
    for x in range(input_obj.width):
        for y in range(input_obj.height):
            if input_obj[x, y] == color:
                continue
            if input_obj[x, y] != output_obj[x, y]:
                return None

    # Check if output has high regularity score
    if regularity_score(output_obj) >= config.inpainting_regularity_score_threshold:
        return None

    return color


def update_mask(input_obj: Object, output_obj: Object, mask: Object) -> None:
    """
    Update the mask grid in-place with 0s where input and output objects differ.
    """
    for x in range(output_obj.width):
        for y in range(output_obj.height):
            if input_obj[x, y] != output_obj[x, y]:
                mask[x, y] = 0


def mask_from_all_outputs(examples: Examples[Object, Object]) -> Optional[Object]:
    """
    Find the mask for a set of examples where all examples have the same size.
    Returns None if any example doesn't satisfy inpainting conditions or has mismatched sizes.
    """
    if not examples:
        return None

    _, first_output = examples[0]
    mask = Object.empty(first_output.size, background_color=1)

    # Update the mask with each example
    for input_obj, output_obj in examples:
        color = check_inpainting_conditions(
            input_obj, output_obj, output_is_block=False
        )
        if color is None:
            return None
        if mask.size != output_obj.size or input_obj.size != output_obj.size:
            return None
        update_mask(first_output, output_obj, mask)
    return mask


def apply_mask_to_filled_grid(
    examples: Examples[Object, Object],
    filled_grid: Object,
    input_obj: Object,
    mask: Optional[Object],
    color_only_in_input: int,
    apply_mask_to_input: bool,
) -> Object:
    """
    Apply the mask to the filled grid, using input or the first output as the source.
    If mask is not None, mask out areas where input and output differ.
    """
    if mask is not None:
        source = input_obj if apply_mask_to_input else examples[0][1]
        filled_grid.add_object_in_place(
            source.apply_mask(mask, background_color=color_only_in_input),
            background_color=color_only_in_input,
        )
    return filled_grid


def check_correctness(
    examples: Examples[Object, Object],
    input_obj: Object,
    output_obj: Object,
    shared_symmetries: Tuple[
        NonPeriodicGridSymmetry, PeriodicGridSymmetry, List[CardinalityPredicate], int
    ],
    mask: Optional[Object],
    output_is_largest_block_object: bool,
    apply_mask_to_input: bool,
) -> bool:
    """
    Validate whether the filled grid, after applying shared symmetries and mask, matches the output.
    """
    (
        non_periodic_shared,
        periodic_shared,
        cardinality_shared,
        color_only_in_input,
    ) = shared_symmetries
    filled_grid = fill_grid(
        input_obj,
        mask,
        periodic_shared,
        non_periodic_shared,
        cardinality_shared,
        color_only_in_input,
    )
    filled_grid = apply_mask_to_filled_grid(
        examples,
        filled_grid,
        input_obj,
        mask,
        color_only_in_input,
        apply_mask_to_input,
    )

    if output_is_largest_block_object:
        candidate_output = extract_largest_block(input_obj, filled_grid)
        return candidate_output == output_obj
    else:
        return check_equality_modulo_mask(filled_grid, output_obj, mask)


def check_equality_modulo_mask(
    grid1: Object, grid2: Object, mask: Optional[Object]
) -> bool:
    """Check if two grids are equal, ignoring differences where mask is applied."""
    for x in range(grid1.width):
        for y in range(grid1.height):
            if grid1[x, y] != grid2[x, y] and (mask is None or mask[x, y] == 0):
                return False
    return True


def extract_largest_block(input_obj: Object, filled_grid: Object) -> Object:
    """Extract the largest block object from the filled grid."""
    largest_block_object = get_largest_block_object(input_obj)
    assert largest_block_object is not None
    origin = largest_block_object.origin
    width, height = largest_block_object.size
    data = filled_grid._data

    # Extract the subgrid corresponding to the largest block object
    largest_block_data = data[
        origin[1] : origin[1] + height, origin[0] : origin[0] + width
    ]
    return Object(largest_block_data)


def inpainting_xform(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
    mask: Optional[Object],
    apply_mask_to_input: bool,
    output_is_block: bool,
) -> Optional[Match[Object, Object]]:
    """Main inpainting transform function."""
    transform = InpaintingTransform(
        examples, task_name, nesting_level, mask, apply_mask_to_input, output_is_block
    )

    # First phase: Analyze
    if not transform.analyze():
        return None  # Strategy determination failed

    # Second phase: Return the state and solve function
    return (transform.state, transform.solve)


def inpainting_xform_no_mask(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    """Inpainting transform without mask."""
    return inpainting_xform(
        examples,
        task_name,
        nesting_level,
        mask=None,
        apply_mask_to_input=False,
        output_is_block=False,
    )


def inpainting_xform_output_is_block(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    """Inpainting transform where output is the largest block."""
    return inpainting_xform(
        examples,
        task_name,
        nesting_level,
        mask=None,
        apply_mask_to_input=False,
        output_is_block=True,
    )


def inpainting_xform_with_mask(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    """Inpainting transform with mask."""
    mask = mask_from_all_outputs(examples)
    if mask is not None and config.display_verbose:
        display(mask, title=f"Mask")
    return inpainting_xform(
        examples,
        task_name,
        nesting_level,
        mask=mask,
        apply_mask_to_input=False,
        output_is_block=False,
    )
