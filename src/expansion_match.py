from typing import List, Optional
from objects import Object, display_multiple, display
from load_data import Example
from bi_types import Match, GridAndObjects
from logger import logger
import numpy as np
from bi_types import XformEntry


def check_fractal_expansion_sizes(examples: List[Example[GridAndObjects]]):
    """
    Check if every input is NxN and the output's size is N^2xN^2
    """
    for (input_grid, input_objects), (output_grid, output_objects) in examples:
        if len(input_objects) == 0 or len(output_objects) == 0:
            return False
    for input_obj, output_obj in zip(input_objects, output_objects):
        # Ensure input is NxN (i.e., square)
        if input_obj.width != input_obj.height:
            return False
        # Ensure output is N^2xN^2
        if (
            output_obj.width != input_obj.width**2
            or output_obj.height != input_obj.height**2
        ):
            return False
    return True


# TODDO: replace this with inferring a function from (grid, pixel coordinates) to output grid (of the same size)


def fractal_expansion(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object]]:

    def map_function_numpy_inplace(
        output_grid: np.ndarray,
        input_grid: np.ndarray,
        original_image: np.ndarray,
        background_color,
    ) -> None:
        width, height = input_grid.shape
        for x in range(width):
            for y in range(height):
                color = input_grid[x, y]
                if color != background_color:
                    output_grid[
                        x * width : (x + 1) * width, y * height : (y + 1) * height
                    ] = original_image

    def apply_recursive_expansion_numpy_inplace(
        original_image: np.ndarray, background_color
    ) -> np.ndarray:
        width, height = original_image.shape
        output_grid = np.full((width * width, height * height), background_color)
        map_function_numpy_inplace(
            output_grid, original_image, original_image, background_color
        )
        return output_grid

    if False:
        display_multiple(examples, title=f"Fractal Expansion Examples")

    for input, output in examples:
        input_image = input._data
        output_image = output._data
        expanded_image = apply_recursive_expansion_numpy_inplace(input_image, 0)
        if False:
            display(
                Object(expanded_image),
                Object(output_image),
                title=f"Expanded vs Output",
            )
        if not np.array_equal(expanded_image, output_image):
            return None

    state = "fractal_expansion"

    def solve(input: Object) -> Object:
        if isinstance(input, Object):
            return Object(apply_recursive_expansion_numpy_inplace(input._data, 0))
        else:
            assert False

    return (state, solve)


def stretch_height(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object]]:
    # TODO: implement the inference of the boolean function
    for i, (input, output) in enumerate(examples):
        if output.height != 2 or output.width != input.width:
            return None
        color = input.first_color
        for x in range(input.width):
            for y in range(input.height):
                is_filled = x % 2 == 0 if y == 0 else x % 2 == 1
                if input.origin != (0, 0):
                    # TODO: fix this
                    is_filled = not is_filled
                output_color = output[x, y]
                predicted_color = color if is_filled else 0
                if output_color != predicted_color:
                    logger.info(
                        f"Example {i} failed: output_color {output_color} != predicted_color {predicted_color}"
                    )
    state = "stretch_height"

    def solve(input: Object) -> Object:
        output = Object.empty((input.width, input.height * 2))
        color = input.first_color
        for x in range(input.width):
            for y in range(input.height * 2):
                is_filled = x % 2 == 0 if y == 0 else x % 2 == 1
                if input.origin != (0, 0):
                    # TODO: fix this
                    is_filled = not is_filled
                if is_filled:
                    output[x, y] = color
        return output

    match = (state, solve)
    return match


