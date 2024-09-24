from typing import List, Optional, Tuple

from bi_types import Examples, Match
from grid_types import ClockwiseRotation, RigidTransformation, XReflection
from logger import logger
from objects import Object, display, display_multiple


def find_canvas_objects(
    inputs: List[Object], outputs: Optional[List[Object]]
) -> Optional[List[Object]]:
    """Finds the largest objects in each example's input. Check the size is the same as the output's size, if provided."""
    canvas_objects = []
    for i, input in enumerate(inputs):
        objects = input.detect_objects()
        canvas = max(objects, key=lambda obj: obj.area, default=None)
        if canvas is None:
            return None
        if outputs is not None:
            output = outputs[i]
            if canvas.size != output.size:
                return None
        canvas_objects.append(canvas)

    return canvas_objects


def equal_modulo_rigid_transformation(
    examples: Examples[Object, Object], task_name: str, nesting_level: int
) -> Optional[Match[Object, Object]]:
    for x_reflection in XReflection:
        for rotation in ClockwiseRotation:
            rigid_transformation = RigidTransformation(rotation, x_reflection)
            all_examples_correct = True
            for input, output in examples:
                trasformed_input = input.apply_rigid_xform(rigid_transformation)
                if trasformed_input != output:
                    all_examples_correct = False
                    break
            if all_examples_correct:
                state = f"({rigid_transformation})"
                solve = lambda input: input.apply_rigid_xform(rigid_transformation)
                return (state, solve)
    return None


def solve_puzzle(
    input: Object, task_name: str, nesting_level: int, canvas: Object
) -> Optional[Tuple[Object, List[Object]]]:

    canvas_color = canvas.main_color()
    compound_objects = input.detect_objects(
        background_color=canvas_color, multicolor=True
    )
    # remove canvas and input grid from grid_objects
    compound_objects = [obj for obj in compound_objects if obj.area < canvas.area]

    def compound_object_get_handle(cobj: Object) -> Optional[Object]:
        colors = set(cobj.get_colors(allow_black=True))
        if canvas_color not in colors:
            logger.debug(f"Canvas color: {canvas_color} not in colors: {colors}")
            return None
        colors.remove(canvas_color)
        if len(colors) != 1:
            logger.debug(f"Canvas color: {canvas_color} colors: {colors}")
            return None
        other_color = colors.pop()
        handle = cobj.detect_colored_objects(background_color=other_color)[0]
        logger.debug(f"Canvas color: {canvas_color} Handle color: {other_color}")

        return handle

    handles = []
    handles_shapes = []
    for cobj in compound_objects:
        handle = compound_object_get_handle(cobj)
        if handle is None:
            return None
        handles.append(handle)
        handles_shapes.append(handle.get_shape())
    holes = canvas.detect_objects(allow_black=True, background_color=canvas_color)
    # remove object of size the entire max_area_object
    holes = [obj for obj in holes if obj.area < canvas.area]
    holes_shapes = [obj.get_shape(background_color=canvas_color) for obj in holes]

    logger.debug(f"compound_objects:{len(compound_objects)} holes:{len(holes)}")

    if len(compound_objects) != len(holes):
        return None
    if len(compound_objects) == 0:
        return None

    # display grid_objects alongsize holes
    if False:
        display_multiple(
            [(o1, o2) for o1, o2 in zip(compound_objects, holes)],
            title=f"Compound Objects and Canvas Holes",
        )

    # display objects_holes alonside canvas_holes
    if False:
        display_multiple(
            [
                (o1, o2)
                for o1, o2 in zip(
                    handles,
                    holes,
                )
            ],
            title=f"Handles and Holes",
        )

    if False:
        display_multiple(
            [(o1, o2) for o1, o2 in zip(handles_shapes, holes_shapes)],
            title=f"Handles Shapes and Holes Shapes",
        )

    matches = []
    num_cases = len(handles_shapes)
    xform = equal_modulo_rigid_transformation
    matched_handles = set()
    matched_holes = set()
    transformed_compound_objects = compound_objects
    holes_origins = [(0, 0)] * num_cases
    for i in range(num_cases):
        match = None
        for j in range(num_cases):
            if i in matched_handles or j in matched_holes:
                continue
            handle = handles[i]
            hole = holes[j]
            handle_shape = handles_shapes[i]
            hole_shape = holes_shapes[j]
            match = xform(
                [
                    (
                        handle_shape,
                        hole_shape,
                    )
                ],
                task_name,
                nesting_level + 1,
            )
            if match is None:
                continue
            matched_handles.add(i)
            matched_holes.add(j)
            state, solve = match

            transformed_compound_object = solve(compound_objects[i])
            if transformed_compound_object is None:
                return None
            transformed_compound_objects[i] = transformed_compound_object
            holes_origins[i] = hole.origin

            matches.append(match)
        if match is None:
            logger.info(f"No match found for handle {i}")
            return None
    logger.debug(f"matches found:{len(matches)}")

    if False:
        display_multiple(
            [(obj, obj) for obj in transformed_compound_objects],
            title=f"Transformed Compound Objects",
        )

    new_objects = []
    for i, obj in enumerate(transformed_compound_objects):
        new_handle = compound_object_get_handle(obj)
        if new_handle is None:
            return None
        ox, oy = holes_origins[i]
        ox -= new_handle.origin[0]
        oy -= new_handle.origin[1]
        obj.origin = (ox, oy)
        logger.debug(f"new origin: {obj.origin}")
        new_objects.append(obj)

    return canvas, new_objects


def canvas_grid_xform(
    examples: Examples[Object, Object],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object, Object]]:
    # every example has a canvas
    canvas_objects = find_canvas_objects(
        inputs=[input for input, _ in examples],
        outputs=[output for _, output in examples],
    )
    if canvas_objects is None:
        return None
    # Config.display_this_task = True

    for i, (input, output) in enumerate(examples):
        canvas = canvas_objects[i]
        solution = solve_puzzle(input, task_name, nesting_level, canvas)
        if solution is None:
            return None
        canvas, new_objects = solution
        new_input = input.copy()
        for obj in new_objects:
            ox, oy = obj.origin
            ox += canvas.origin[0]
            oy += canvas.origin[1]
            obj.origin = (ox, oy)
            logger.debug(f"new origin: {obj.origin}")
            new_input.add_object_in_place(obj)

    state = "canvas_grid_xform"

    def solve(input: Object) -> Optional[Object]:
        canvas_objects = find_canvas_objects([input], None)
        if canvas_objects is None:
            return None
        canvas = canvas_objects[0]
        solution = solve_puzzle(input, task_name, nesting_level, canvas)
        if solution is None:
            logger.info(f"No solution found for input")
            return None
        canvas, new_objects = solution
        output = canvas.copy()
        output.origin = (0, 0)
        for obj in new_objects:
            output.add_object_in_place(obj)
        if False:
            display(output, title=f"Output")
        return output

    match = (state, solve)
    return match
