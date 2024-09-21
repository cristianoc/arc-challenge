from typing import List, Optional, Tuple

from bi_types import Match, XformEntry
from expansion_match import fractal_expansion
from load_data import Example
from logger import logger
from objects import Object, display, display_multiple


def out_objects_are_a_subset(
    inputs: List[Object], outputs: List[Object]
) -> Optional[List[Tuple[int, int]]]:
    """
    Determines if the output objects are a subset of the input objects based on their colors.

    Checks if each color in the output set is present in the input set. Returns a mapping
    of input indices to output indices if the subset condition is met, or None if not satisfied
    or if any output color is not present in the input.
    """
    input_colors = [input_obj.first_color for input_obj in inputs]
    output_colors = [output_obj.first_color for output_obj in outputs]

    input_to_output_indices = []

    for ic in input_colors:
        if ic in output_colors:
            input_to_output_indices.append(
                (input_colors.index(ic), output_colors.index(ic))
            )
    for oc in output_colors:
        if oc not in input_colors and False:
            display_multiple(
                [
                    (input_obj, output_obj)
                    for input_obj, output_obj in zip(inputs, outputs)
                ],
                title=f"Input vs Output",
            )
            return None  # Output color not in input

    return input_to_output_indices


def stretch_height(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"stretch_height examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )
    origin = None
    for input, output in examples:
        if origin is None:
            origin = output.origin
        if origin != output.origin:
            logger.info(
                f"Output origin: {output.origin} != Expected origin: {origin}"
            )
            return None
        if input.width != output.width:
            logger.info(
                f"Input width: {input.width} != Output width: {output.width}"
            )
            return None
        if input.height * 2 != output.height:
            logger.info(
                f"Input height * 2: {input.height * 2} != Output height: {output.height}"
            )
            return None
        logger.info(
            f"stretch_height origin:{output.origin} width:{output.width} height:{output.height}"
        )
        if False:
            display(input, output, title=f"stretch_height")
    # TODO: need to adjust the origin from the call to the expansion xform
    for xform in expansion_xforms:
        match = xform.xform(examples, task_name, nesting_level)
        if match is not None:
            return match
    return None

expansion_xforms: List[XformEntry[Object, Object]] = [
    XformEntry(fractal_expansion, 1),
    XformEntry(stretch_height, 1),
]
