from typing import List, Optional, Tuple

from bi_types import Example, Examples, Match, XformEntry
from expansion_match import stretch_height
from find_xform import find_xform_for_examples
from logger import logger
from map_function_match import expansion_xforms, out_objects_are_a_subset
from objects import Object, display_multiple

map_xforms: List[XformEntry[Object, Object]] = [XformEntry(stretch_height, 1)]


def check_list_of_objects_subset(
    examples: Examples[List[Object], List[Object]],
) -> Optional[List[Tuple[int, int]]]:
    """
    Check if the output objects are a subset of the input objects based on their colors.
    Returns a list of indices of the input objects that correspond to the output objects.
    The same list must apply to all examples.
    """
    input_to_output_indices_list = []
    for input_objects, output_objects in examples:
        if len(input_objects) < 2:
            return None
        input_to_output_indices = out_objects_are_a_subset(
            input_objects, output_objects
        )
        if input_to_output_indices is None:
            return None
        # store the indices
        input_to_output_indices_list.append(input_to_output_indices)
    # check if they are all the same
    if len(set(tuple(indices) for indices in input_to_output_indices_list)) != 1:
        return None
    logger.info(f"input_to_output_indices_list: {input_to_output_indices_list}")
    input_to_output_indices = input_to_output_indices_list[0]
    if len(input_to_output_indices) == 0:
        return None
    return input_to_output_indices


def map_first_input_to_output_grid(
    examples: Examples[List[Object], List[Object]],
) -> Examples[Object, Object]:
    input_output_objects_examples: Examples[Object, Object] = []
    for input_objects, output_objects in examples:
        input_output_objects_examples.append((input_objects[0], output_objects[0]))

    return input_output_objects_examples


def check_fractal_expansion_sizes(
    examples: Examples[List[Object], List[Object]],
) -> bool:
    """
    Check if every input is NxN and the output's size is N^2xN^2
    """
    for input_objects, output_objects in examples:
        if len(input_objects) != 1 or len(output_objects) != 1:
            return False
    output_obj = output_objects[0]
    input_obj = input_objects[0]
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


def match_object_list(
    examples: Examples[List[Object], List[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[List[Object], List[Object]]]:
    logger.info(
        f"{'  ' * nesting_level}match_object_list examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    if check_fractal_expansion_sizes(examples):
        input_output_objects_examples = map_first_input_to_output_grid(examples)

        # now pattern match recursively
        match: Optional[Match[Object, Object]] = find_xform_for_examples(
            expansion_xforms,
            input_output_objects_examples,
            task_name,
            nesting_level + 1,
        )
        if match is not None:
            state, solve = match

            def solve_grid_and_objects(
                objects: List[Object],
            ) -> Optional[List[Object]]:
                solved_objects_ = [solve(obj) for obj in objects]
                solved_objects = [obj for obj in solved_objects_ if obj is not None]
                if len(solved_objects) != len(objects):
                    return None
                return solved_objects

            return state, solve_grid_and_objects

    # check if the input objects can be matched to the output objects
    input_to_output_indices = check_list_of_objects_subset(examples)
    if input_to_output_indices is not None:
        logger.info(
            f"{'  ' * nesting_level}Found input_to_output_indices: {input_to_output_indices}"
        )

        new_examples_train: List[Examples[Object, Object]] = [
            [] for _ in input_to_output_indices
        ]
        for e_inputs, e_outputs in examples:
            for i, (input_index, output_index) in enumerate(input_to_output_indices):
                new_examples_train[i].append(
                    (e_inputs[input_index], e_outputs[output_index])
                )

        for xform in map_xforms:
            matches = []  # for each input/output index pair, the match
            for i, (input_index, output_index) in enumerate(input_to_output_indices):
                match = xform.xform(
                    new_examples_train[i],
                    task_name,
                    nesting_level,
                )
                if match is None:
                    logger.info(
                        f"Xform {xform.xform.__name__} index:{output_index} failed: no match"
                    )
                    return None
                else:
                    matches.append(match)

        logger.info(f"Xform {xform.xform.__name__} succeeded")

        new_state = "{"
        for i, (s, _) in enumerate(matches):
            new_state += f"{i}:{s}, "
        new_state += "}"

        def solve_grid_and_objects(
            objects: List[Object],
        ) -> Optional[List[Object]]:
            outputs = []
            assert input_to_output_indices is not None
            for i, (input_index, output_index) in enumerate(input_to_output_indices):
                state, solve = matches[i]
                output = solve(objects[input_index])
                if output is None:
                    return None
                outputs.append(output)
            return outputs

        return new_state, solve_grid_and_objects

    logger.info(f"{'  ' * nesting_level}TODO: more cases of match_list_of_objects")

    return None
