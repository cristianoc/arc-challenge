from typing import List, Optional, Tuple, Callable

import config
from bi_types import Examples, Match, Object, Xform, XformEntry
from inpainting_match import inpainting_xform
from logger import logger
from objects import display, display_multiple


def matcher_from_primitive(
    get_index: Callable[[List[Object]], int]
) -> Xform[List[Object], int]:
    def matcher(
        examples: Examples[List[Object], int], task_name: str, nesting_level: int
    ) -> Optional[Match[List[Object], int]]:
        for inputs, index in examples:
            if index != get_index(inputs):
                return None
        state = f"matcher_from_primitive({get_index.__name__})"
        return (state, lambda inputs: get_index(inputs))

    return matcher


def feat_smallest_area(inputs: List[Object]) -> int:
    smallest_index, _ = min(
        [(i, o.area) for i, o in enumerate(inputs)],
        key=lambda x: x[1],
    )
    return smallest_index


def feat_largest_num_cells(inputs: List[Object]) -> int:
    index, _ = max(
        [(i, o.num_cells(None)) for i, o in enumerate(inputs)],
        key=lambda x: x[1],
    )
    return index


def feat_is_square(inputs: List[Object]) -> int:
    for i, o in enumerate(inputs):
        if o.width == o.height:
            return i
    return 0


select_object_xforms: List[XformEntry[List[Object], int]] = [
    XformEntry(matcher_from_primitive(feat_smallest_area), 3),
    XformEntry(matcher_from_primitive(feat_largest_num_cells), 3),
    XformEntry(matcher_from_primitive(feat_is_square), 3),
    # TODO: use detect_common_features instead of listing the features explicitly here
]


def match2_object_is_output(
    examples: Examples[Tuple[Object, Object], Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Tuple[Object, Object], Object]]:
    for input, output in examples:
        if input[0] != output:
            return None

    def solver(inputs: Tuple[Object, Object]) -> Optional[Object]:
        return inputs[0]

    return ("match2_object_is_output", solver)

def mask_right(inputs: Tuple[Object, Object], background_color: int) -> Object:
    """
    o1 is the main object, o2 is the mask
    """
    (o1, o2) = inputs
    width, height = o1.size
    o = Object.empty(o1.size, background_color)
    for x in range(width):
        for y in range(height):
            if o2[x, y] != background_color:
                o[x, y] = o1[x, y]
    return o


def mask_left(inputs: Tuple[Object, Object], background_color: int) -> Object:
    """
    o2 is the main object, o1 is the mask
    """
    (o1, o2) = inputs
    width, height = o1.size
    o = Object.empty(o1.size, background_color)
    for x in range(width):
        for y in range(height):
            if o2[x, y] != background_color:
                o[x, y] = o1[x, y]
    return o

binary_operations = [mask_right, mask_left] # TODO: add more, and/or infer them

def match2_binary_operation(
    examples: Examples[Tuple[Object, Object], Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Tuple[Object, Object], Object]]:
    for (i1, i2), o in examples:
        if i1.size != o.size or i2.size != o.size:
            return None
    first_output_size = examples[0][0][1].size
    all_outputs_have_same_size = all(o.size == first_output_size for _, o in examples)
    if all_outputs_have_same_size: # TODO
        logger.info("TODO: can have a different binary operation for each cell")
    background_color = 0 # Todo: find a way to send it in from the outside
    binop = None
    for xform in binary_operations:
        found = True
        for (i1, i2), o in examples:
            if xform((i1, i2), background_color) != o:
                found = False
                break
        if found:
            binop = xform
            break
    if binop is None:
        return None
    def solver(inputs: Tuple[Object, Object]) -> Optional[Object]:
        return binop(inputs, background_color)
    return (f"match2_binary_operation({binop.__name__})", solver)


def match2_downscale_square_object(
    examples: Examples[Tuple[Object, Object], Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Tuple[Object, Object], Object]]:
    def get_multiple(input: Tuple[Object, Object]) -> Optional[int]:
        main_object = input[0]
        other_object = input[1]
        main_width = main_object.width
        main_height = main_object.height
        other_width = other_object.width
        other_height = other_object.height
        # check if other_width is a multiple of main_width and other_height is a multiple of main_height
        if other_width % main_width == 0 and other_height % main_height == 0:
            multiple = other_width // main_width
            if multiple == 1:
                return None # recursion
            return multiple
        return None

    downscaled_examples: Examples[Tuple[Object, Object], Object] = []
    for inputs, output in examples:
        multiple = get_multiple(inputs)
        if multiple is None:
            return None
        downscaled_object = inputs[1].downscale(multiple)
        if downscaled_object.size != inputs[0].size:
            return None
        downscaled_examples.append(((inputs[0], downscaled_object), output))

    found_xform = None
    for xform in combine_two_objects_xforms:
        result = xform.xform(downscaled_examples, task_name, nesting_level)
        if result is None:
            continue
        state_, solver_ = result
        found_xform = (state_, solver_)
        break
    if found_xform is None:
        return None

    state_, solver_ = found_xform

    def solver(inputs: Tuple[Object, Object]) -> Optional[Object]:
        multiple = get_multiple(inputs)
        if multiple is None:
            return None
        downscaled_object = inputs[1].downscale(multiple)
        return solver_((inputs[0], downscaled_object))

    state = state_ + f"downscale(multiple={multiple})"
    return (state, solver)


combine_two_objects_xforms: List[XformEntry[Tuple[Object, Object], Object]] = [
    XformEntry(match2_object_is_output, 2),
    XformEntry(match2_downscale_square_object, 2),
    XformEntry(match2_binary_operation, 3),
]


def match3plus_object_is_output(
    examples: Examples[Tuple[Object, ...], Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Tuple[Object, ...], Object]]:
    for input, output in examples:
        if input[0] != output:
            return None

    def solver(inputs: Tuple[Object, ...]) -> Optional[Object]:
        return inputs[0]

    return ("match3plus_object_is_output", solver)


combine_threeplus_objects_xforms: List[
    XformEntry[Tuple[Object, ...], Object]
] = [
    XformEntry(match3plus_object_is_output, 3),
]


def match_n_objects_with_output(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    logger.info(
        f"{'  ' * nesting_level}match_n_objects_with_output examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )
    objext_and_index_list: List[Tuple[List[Object], int]] = []
    def get_objects(input: Object) -> Optional[List[Object]]:
        background_color = input.main_color(allow_black=True)
        objects = input.detect_objects(
            diagonals=True,
            background_color=background_color,
            multicolor=True,
            keep_origin=False,
        )
        n = len(objects)
        if n <= 1:
            return None
        return objects
    def get_objects_and_index(input: Object, output: Object) -> Optional[Tuple[List[Object], int]]:
        objects = get_objects(input)
        if objects is None:
            return None
        output_indices = [i for i, o in enumerate(objects) if o.size == output.size]
        if len(output_indices) != 1:
            return None
        output_index = output_indices[0]
        return (objects, output_index)

    for input, output in examples:
        if input.size == output.size:
            return None
        objects_and_index = get_objects_and_index(input, output)
        if objects_and_index is None:
            return None
        objects, output_index = objects_and_index
        objext_and_index_list.append((objects, output_index))

    found_select_xform = None
    for xform in select_object_xforms:
        result = xform.xform(objext_and_index_list, task_name, nesting_level)
        if result is None:
            continue
        found_select_xform = result
        break
    if found_select_xform is None:
        return None
    select_state, select_solver = found_select_xform

    num_other_objects = len(objext_and_index_list[0][0]) - 1
    two_object_list: List[Tuple[Object, Object]] = []
    three_object_list: List[Tuple[Object, Object, Object]] = []
    for objects, output_index in objext_and_index_list:
        other_objects = [o for i, o in enumerate(objects) if i != output_index]
        if len(other_objects) != num_other_objects:
            return None
        if num_other_objects == 1:
            two_object_list.append((objects[output_index], other_objects[0]))
        elif num_other_objects == 2:
            three_object_list.append(
                (objects[output_index], other_objects[0], other_objects[1])
            )
        else:
            return None

    two_object_examples: Examples[Tuple[Object, Object], Object] = []
    threeplus_object_examples: Examples[Tuple[Object, ...], Object] = []

    if num_other_objects == 1:
        two_object_examples = [(x, e[1]) for x, e in zip(two_object_list, examples)]
    elif num_other_objects >= 2:
        threeplus_object_examples = [(x, e[1]) for x, e in zip(three_object_list, examples)]
    else:
        return None

    if num_other_objects == 1:
        found_xform2 = None
        for xform2 in combine_two_objects_xforms:
            match2 = xform2.xform(two_object_examples, task_name, nesting_level)
            if match2 is None:
                continue
            found_xform2 = match2
            break
        if found_xform2 is None:
            config.display_this_task = True
            return None
        match2_state, match2_solver = found_xform2
        def solver2(input: Object) -> Optional[Object]:
            objects = get_objects(input)
            if objects is None:
                return None
            index = select_solver(objects)
            if index is None:
                return None
            main_object = objects[index]
            other_objects = [o for o in objects if o != main_object]
            if len(other_objects) != 1:
                return None
            other_object = other_objects[0]
            output = match2_solver((main_object, other_object))
            return output
        state = f"match_n_objects_with_output({match2_state},{select_state})"
        return (state, solver2)
    elif num_other_objects == 2:
        found_xform3 = None
        for xform3 in combine_threeplus_objects_xforms:
            match3 = xform3.xform(threeplus_object_examples, task_name, nesting_level)
            if match3 is None:
                continue
            found_xform3 = match3
            break
        if found_xform3 is None:
            config.display_this_task = True
            return None
        match3_state, match3_solver = found_xform3

        def solver3(input: Object) -> Optional[Object]:
            objects = get_objects(input)
            if objects is None:
                return None
            index = select_solver(objects)
            if index is None:
                return None
            main_object = objects[index]
            other_objects = [o for o in objects if o != main_object]
            if len(other_objects) < 2:
                return None
            output = match3_solver((main_object, *other_objects))
            return output
        state = f"match_n_objects_with_output({match3_state},{select_state})"
        return (state, solver3)
    else:
        return None
