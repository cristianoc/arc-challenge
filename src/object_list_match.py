from typing import List, Optional, Tuple
from load_data import Example
from bi_types import GridAndObjects, Match, XformEntry
from logger import logger
from objects import Object
from map_function_match import expansion_xforms, out_objects_are_a_subset
from expansion_match import check_fractal_expansion_sizes, stretch_height
from find_xform import find_xform_for_examples


map_xforms: List[XformEntry[Object]] = [XformEntry(stretch_height, 1)]

def check_list_of_objects_subset(
    examples: List[Example[GridAndObjects]],
) -> Optional[List[Tuple[int, int]]]:
    """
    Check if the output objects are a subset of the input objects based on their colors.
    Returns a list of indices of the input objects that correspond to the output objects.
    The same list must apply to all examples.
    """
    input_to_output_indices_list = []
    for (_, input_objects), (_, output_objects) in examples:
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
    examples: List[Example[GridAndObjects]],
) -> List[Example[Object]]:
    input_output_objects_examples: List[Example[Object]] = []
    for (input_grid, input_objects), (output_grid, output_objects) in examples:
        input_output_objects_examples.append((input_objects[0], output_grid))

    return input_output_objects_examples


def match_list_of_objects(
    examples: List[Example[GridAndObjects]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[GridAndObjects]]:
    logger.info(
        f"{'  ' * nesting_level}match_list_of_objects examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    if check_fractal_expansion_sizes(examples):
        input_output_objects_examples = map_first_input_to_output_grid(
            examples
        )

        # now pattern match recursively
        match: Optional[Match[Object]] = find_xform_for_examples(
            expansion_xforms,
            input_output_objects_examples,
            task_name,
            nesting_level + 1,
        )
        if match is not None:
            state, solve = match

            def solve_grid_and_objects(
                grid_and_objects: GridAndObjects,
            ) -> Optional[GridAndObjects]:
                grid, objects = grid_and_objects
                solved_objects_ = [solve(obj) for obj in objects]
                solved_objects = [obj for obj in solved_objects_ if obj is not None]
                if len(solved_objects) != len(objects):
                    return None
                return (grid, solved_objects)

            return state, solve_grid_and_objects

    # check if the input objects can be matched to the output objects
    input_to_output_indices = check_list_of_objects_subset(examples)
    if input_to_output_indices is not None:
        logger.info(
            f"{'  ' * nesting_level}Found input_to_output_indices: {input_to_output_indices}"
        )

        new_examples_train: List[List[Example[Object]]] = [
            [] for _ in input_to_output_indices
        ]
        for (_, e_inputs), (_, e_outputs) in examples:
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
            grid_and_objects: GridAndObjects,
        ) -> Optional[GridAndObjects]:
            input_grid, input_objects = grid_and_objects
            outputs = []
            assert input_to_output_indices is not None
            for i, (input_index, output_index) in enumerate(input_to_output_indices):
                state, solve = matches[i]
                output = solve(input_objects[input_index])
                if output is None:
                    return None
                outputs.append(output)
            return (input_grid, outputs)

        return new_state, solve_grid_and_objects

    logger.info(f"{'  ' * nesting_level}TODO: more cases of match_list_of_objects")

    return None
