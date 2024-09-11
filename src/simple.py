from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Generic,
    Union,
    Set,
)

from color_features import detect_color_features
from logger import logger
from objects import Object, display, display_multiple
from load_data import Example, Task, Tasks, training_data, evaluation_data
from rule_based_selector import DecisionRule, select_object_minimal
from shape_features import detect_shape_features
from symmetry_features import detect_symmetry_features
from symmetry import find_periodic_symmetry_with_unknowns, fill_grid, GridSymmetry
from visual_cortex import find_rectangular_objects
import numpy as np
from dataclasses import dataclass
from grid_normalization import ClockwiseRotation, XReflection, RigidTransformation

# returns the index of the object to pick
ObjectPicker = Callable[[List[Object]], int]


class Config:
    find_xform = True
    find_matched_objects = False
    try_remove_main_color = False
    difficulty = 1000
    task_name: str | None = None
    # task_name = "e9afcf9a.json"  # map 2 colored objects
    # task_name = "0dfd9992.json"
    # task_name = "05269061.json"
    task_fractal = "8f2ea7aa.json"  # fractal expansion
    task_puzzle = "97a05b5b.json"  # puzzle-like, longest in DSL (59 lines)
    whitelisted_tasks: List[str] = []
    whitelisted_tasks.append(task_puzzle)
    # find_xform_color = True
    display_not_found = False
    display_this_task = False
    only_simple_examples = False
    max_size = 9
    max_colors = 4


def filter_simple_xforms(task: Task, task_name: str):
    examples = task.train
    tests = task.test
    for example in examples:
        input = example[0]
        output = example[1]
        if (
            input.width > Config.max_size
            or input.height > Config.max_size
            or input.size != output.size
            or input.get_colors(allow_black=True) != output.get_colors(allow_black=True)
            or len(input.get_colors(allow_black=True)) > Config.max_colors
        ):
            return False
    return True


GridAndObjects = Tuple[Object, List[Object]]

T = TypeVar("T", bound=Union[Object, GridAndObjects])
State = TypeVar("State")

ApplyState = Callable[[State, T], T]
Primitive = Callable[[Object, str, int], Object]
Match = Tuple[State, ApplyState[State, T]]
Xform = Callable[[List[Example[T]], str, int], Optional[Match[State, T]]]


@dataclass
class XformEntry(Generic[T, State]):
    xform: Xform[T, State]
    difficulty: int


def check_primitive_on_examples(
    prim: Callable[[Object, str, int], Object],
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[str, Object]]:
    logger.debug(f"{'  ' * nesting_level}Checking primitive {prim.__name__}")
    for i, example in enumerate(examples):
        logger.debug(f"{'  ' * nesting_level}  Example {i+1}/{len(examples)}")
        input = example[0]
        output = example[1]
        new_output = prim(input, task_name, nesting_level)
        if new_output != output:
            logger.debug(f"{'  ' * nesting_level}  Example {i+1} failed")
            return None
    state = "prim"
    apply_state: ApplyState = lambda s, i: prim(i, task_name, nesting_level)
    return (state, apply_state)


def primitive_to_xform(primitive: Primitive) -> Xform[Object, str]:
    def xform(
        examples: List[Example],
        task_name: str,
        nesting_level: int,
    ) -> Optional[Match]:
        result = check_primitive_on_examples(
            primitive, examples, task_name, nesting_level
        )
        if result is None:
            return None
        state, apply_state = result
        return (state, apply_state)

    xform.__name__ = primitive.__name__
    return xform


def translate_down_1(input: Object, task_name: str, nesting_level: int):
    obj = input.copy()
    obj.translate_in_place(dx=0, dy=1)
    result = Object.empty(obj.size)
    result.add_object_in_place(obj)
    return result


primitives: List[Primitive] = [
    translate_down_1,
]


def xform_identity(
    examples: List[Example], task_name: str, nesting_level: int
) -> Optional[Match]:
    def identity(input: Object, task_name: str, nesting_level: int):
        return input

    return check_primitive_on_examples(identity, examples, task_name, nesting_level)


# TODO: This is currently not used but it illustrates how to compose primitives
def xform_two_primitives_in_sequence(
    examples: List[Example], task_name: str, nesting_level: int
) -> Optional[Match]:
    # try to apply two primitives in sequence, and return the first pair that works
    for p1 in primitives:
        for p2 in primitives:

            def composed_primitive(input: Object, task_name: str, nesting_level: int):
                r1 = p1(input, task_name, nesting_level + 1)
                return p2(r1, task_name, nesting_level + 1)

            if check_primitive_on_examples(
                composed_primitive, examples, task_name, nesting_level
            ):
                state = (p1.__name__, p2.__name__)
                apply_state = lambda state, input: p2(
                    p1(input, task_name, nesting_level + 1),
                    task_name,
                    nesting_level + 1,
                )
                return (state, apply_state)
    return None


def check_matching_colored_objects_count_and_color(examples: List[Example]) -> bool:
    for input, output in examples:
        input_objects = input.detect_colored_objects(background_color=0)
        output_objects = output.detect_colored_objects(background_color=0)
        if len(input_objects) != len(output_objects):
            return False

        different_color = any(
            input_object.first_color != output_object.first_color
            for input_object, output_object in zip(input_objects, output_objects)
        )

        if different_color:
            return False
    return True


def match_colored_objects(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[str, Object]]:

    logger.info(
        f"{'  ' * nesting_level}match_colored_objects examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    color_match = check_matching_colored_objects_count_and_color(examples)
    if color_match is None:
        return None
    # now the colored input

    # each example has the same number of input and output objects
    # so we can turn those lists into and ObjectListExample
    object_list_examples: List[Example[GridAndObjects]] = []

    def get_background_color(input: Object) -> int:
        background_color = 0  # TODO: determine background color
        return background_color

    def get_grid_and_objects(input: Object) -> GridAndObjects:
        background_color = get_background_color(input)
        input_objects: List[Object] = input.detect_colored_objects(background_color)
        return (input, input_objects)

    input_grid_and_objects: GridAndObjects
    output_grid_and_objects: GridAndObjects
    for input, output in examples:
        input_grid_and_objects = get_grid_and_objects(input)
        output_grid_and_objects = get_grid_and_objects(output)
        input_objects = input_grid_and_objects[1]
        output_objects = output_grid_and_objects[1]

        if len(input_objects) == 0 or len(output_objects) == 0:
            return None

        if False:
            display_multiple(
                [
                    (input_object, output_object)
                    for input_object, output_object in zip(
                        input_objects, output_objects
                    )
                ],
                title=f"Colored Objects [Exam]",
            )

        object_list_example: Example[GridAndObjects] = (
            input_grid_and_objects,
            output_grid_and_objects,
        )
        object_list_examples.append(object_list_example)

    for list_xform in list_xforms:
        match: Optional[Match[str, GridAndObjects]] = list_xform.xform(
            object_list_examples, task_name, nesting_level + 1
        )
        if match is not None:
            apply_state_list_xform: ApplyState[str, GridAndObjects]
            state_list_xform, apply_state_list_xform = match

            def apply_state_object(state, input: Object) -> Object:
                background_color = get_background_color(input)
                input_objects = input.detect_colored_objects(background_color)
                grid_and_objects: GridAndObjects = (input, input_objects)
                _, output_objects = apply_state_list_xform(
                    state_list_xform, grid_and_objects
                )
                output_grid = None
                if False:
                    display_multiple(
                        list(zip(input_objects, output_objects)),
                        title=f"Output Objects",
                    )
                for output in output_objects:
                    if output_grid is None:
                        output_grid = output.copy()
                    else:
                        output_grid.add_object_in_place(output)
                assert output_grid is not None
                if False:
                    display(output_grid, title=f"Output Grid")
                return output_grid

            return (
                f"{list_xform.xform.__name__}({state_list_xform})",
                apply_state_object,
            )
        else:
            logger.info(
                f"{'  ' * nesting_level}Xform {list_xform.xform.__name__} is not applicable"
            )

    return None


class CanvasGridMatch:
    @staticmethod
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

    @staticmethod
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
                state, apply_state = match

                cobj = compound_objects[i]

                transformed_compound_objects[i] = apply_state(
                    state, compound_objects[i]
                )
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

    @staticmethod
    def canvas_grid_xform(
        examples: List[Example[Object]],
        task_name: str,
        nesting_level: int,
    ) -> Optional[Match[str, Object]]:
        # every example has a canvas
        canvas_objects = CanvasGridMatch.find_canvas_objects(
            inputs=[input for input, _ in examples],
            outputs=[output for _, output in examples],
        )
        if canvas_objects is None:
            return None
        # Config.display_this_task = True

        for i, (input, output) in enumerate(examples):
            canvas = canvas_objects[i]
            solution = CanvasGridMatch.solve_puzzle(
                input, task_name, nesting_level, canvas
            )
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

        def apply_state(state, input: Object) -> Object:
            canvas_objects = CanvasGridMatch.find_canvas_objects([input], None)
            if canvas_objects is None:
                return input
            canvas = canvas_objects[0]
            solution = CanvasGridMatch.solve_puzzle(
                input, task_name, nesting_level, canvas
            )
            if solution is None:
                logger.info(f"No solution found for input")
                return input
            canvas, new_objects = solution
            output = canvas.copy()
            output.origin = (0, 0)
            for obj in new_objects:
                output.add_object_in_place(obj)
            if False:
                display(output, title=f"Output")
            return output

        match = (state, apply_state)
        return match


def equal_modulo_rigid_transformation(
    examples: List[Example], task_name: str, nesting_level: int
) -> Optional[Match]:
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
                state = rigid_transformation
                apply_state: ApplyState = lambda state, input: input.apply_rigid_xform(
                    state
                )
                return (state, apply_state)
    return None


def inpainting_xform(
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[str, Object]]:
    # check if the input has one more color than the output
    # and the rest of the input is identical to the output
    # then return the inpainting xform

    regularity_scores = []
    symmetries: Set[GridSymmetry] = set()
    periodic_symmetries: Set[GridSymmetry] = set()
    incorrect_periodic_found = False
    for i, (input, output) in enumerate(examples):
        if input.size != output.size:
            return None
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
        for x in range(input.width):
            for y in range(input.height):
                if input[x, y] == color:
                    continue
                if input[x, y] != output[x, y]:
                    return None
        from visual_cortex import regularity_score

        output_symmetries = set(output.find_symmetries())
        logger.info(f"output_symmetries example {i}:{output_symmetries}")
        if len(output_symmetries) > 0:
            has_symmetries = True

        periodic_symmetry = find_periodic_symmetry_with_unknowns(input, color)
        filled_grid = fill_grid(input, periodic_symmetry, color)
        is_correct = filled_grid == output
        logger.info(
            f"#{i} {periodic_symmetry} is_correct: {is_correct}"
        )
        if is_correct:
            periodic_symmetries.add(periodic_symmetry)
        else:
            incorrect_periodic_found = True
            # Config.display_this_task = True
            # display(input, filled_grid, title=f"{is_correct} Filled Grid")

        symmetries = symmetries.intersection(output_symmetries)
        regularity_scores.append(regularity_score(output))

    average_regularity_score = sum(regularity_scores) / len(regularity_scores)
    logger.info(
        f"inpainting_xform examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level} average_regularity_score:{average_regularity_score:.2f} symmetries:{symmetries}"
    )
    if average_regularity_score < 0.5:
        if incorrect_periodic_found: # and len(symmetries) == 0:
            Config.display_this_task = True
        if len(periodic_symmetries) == 1:
            logger.info(f"periodic_symmetries: {periodic_symmetries}")
        if len(symmetries) == 0:
            logger.error(f"CHECK: {task_name}")
            # Config.display_this_task = True
    # Web view: open -a /Applications/Safari.app "https://arcprize.org/play?task=484b58aa"
    # Tasks with average_regularity_score < 0.5:
    # 484b58aa # sudoku
    # c3f564a4
    # bd4472b8
    # b8825c91
    # 05269061
    # 0dfd9992
    # 3631a71a
    # 8e5a5113
    # 29ec7d0e
    # f9d67f8b # some uncentered symmetry, to investigate
    # 7c8af763
    # 929ab4e9
    # 47996f11
    # 62b74c02
    # c663677b
    # ca8f78db  # periodic symmetry x and y (different px and py)
    # 4aab4007
    # af22c60d
    # e95e3d8e
    # 1d0a4b61  # all have vertical and horizontal symmetry (sometimes need find center), all have periodic symmetry (px and py different)
    # ef26cbf6
    # 1e97544e
    # 4cd1b7b2
    # 981571dc
    # 903d1b4a
    return None


gridxforms: List[XformEntry[Object, str]] = [
    # XformEntry(match_colored_objects, 3),
    # XformEntry(xform_identity, 1),
    # XformEntry(equal_modulo_rigid_transformation, 2),
    # XformEntry(primitive_to_xform(translate_down_1), 2),
    # XformEntry(CanvasGridMatch.canvas_grid_xform, 2),
    XformEntry(inpainting_xform, 2),
]


class ExpansionMatch:

    @staticmethod
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
    @staticmethod
    def fractal_expansion(
        examples: List[Example[Object]],
        task_name: str,
        nesting_level: int,
    ) -> Optional[Match[str, Object]]:

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

        def apply_state(state, input: Object) -> Object:
            if isinstance(input, Object):
                return Object(apply_recursive_expansion_numpy_inplace(input._data, 0))
            else:
                assert False

        return (state, apply_state)

    @staticmethod
    def stretch_height(
        examples: List[Example[Object]],
        task_name: str,
        nesting_level: int,
    ) -> Optional[Match[str, Object]]:
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
                        assert False
        state = "stretch_height"

        def apply_state(state, input: Object) -> Object:
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

        match = (state, apply_state)
        return match


expansion_xforms: List[XformEntry[Object, str]] = [
    XformEntry(ExpansionMatch.fractal_expansion, 1),
    XformEntry(ExpansionMatch.stretch_height, 1),
]


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


class MapFunctionMatch:
    @staticmethod
    def stretch_height(
        examples: List[Example[Object]],
        task_name: str,
        nesting_level: int,
    ) -> Optional[Match[str, Object]]:
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


map_xforms: List[XformEntry[Object, str]] = [
    XformEntry(MapFunctionMatch.stretch_height, 1)
]


from typing import List, Tuple


class ObjectListMatch:
    @staticmethod
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

    @staticmethod
    def map_first_input_to_output_grid(
        examples: List[Example[GridAndObjects]],
    ) -> List[Example[Object]]:
        input_output_objects_examples: List[Example[Object]] = []
        for (input_grid, input_objects), (output_grid, output_objects) in examples:
            input_output_objects_examples.append((input_objects[0], output_grid))

        return input_output_objects_examples

    @staticmethod
    def match_list_of_objects(
        examples: List[Example[GridAndObjects]],
        task_name: str,
        nesting_level: int,
    ) -> Optional[Match[str, GridAndObjects]]:
        logger.info(
            f"{'  ' * nesting_level}match_list_of_objects examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
        )

        if ExpansionMatch.check_fractal_expansion_sizes(examples):
            input_output_objects_examples = (
                ObjectListMatch.map_first_input_to_output_grid(examples)
            )

            # now pattern match recursively
            match: Optional[Match[str, Object]] = find_xform_for_examples(
                expansion_xforms,
                input_output_objects_examples,
                task_name,
                nesting_level + 1,
            )
            if match is not None:
                state, apply_state = match

                def apply_state_grid_and_objects(
                    state, grid_and_objects: GridAndObjects
                ) -> GridAndObjects:
                    grid, objects = grid_and_objects
                    return (grid, [apply_state(state, obj) for obj in objects])

                return state, apply_state_grid_and_objects

        # check if the input objects can be matched to the output objects
        input_to_output_indices = ObjectListMatch.check_list_of_objects_subset(examples)
        if input_to_output_indices is not None:
            logger.info(
                f"{'  ' * nesting_level}Found input_to_output_indices: {input_to_output_indices}"
            )

            new_examples_train: List[List[Example[Object]]] = [
                [] for _ in input_to_output_indices
            ]
            for (_, e_inputs), (_, e_outputs) in examples:
                for i, (input_index, output_index) in enumerate(
                    input_to_output_indices
                ):
                    logger.info(f"inputs:{len(e_inputs)} outputs:{len(e_outputs)}")
                    new_examples_train[i].append(
                        (e_inputs[input_index], e_outputs[output_index])
                    )

            for xform in map_xforms:
                matches = []  # for each input/output index pair, the match
                for i, (input_index, output_index) in enumerate(
                    input_to_output_indices
                ):
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

            def apply_state_grid_and_objects(
                state, grid_and_objects: GridAndObjects
            ) -> GridAndObjects:
                input_grid, input_objects = grid_and_objects
                outputs = []
                assert input_to_output_indices is not None
                for i, (input_index, output_index) in enumerate(
                    input_to_output_indices
                ):
                    state, apply_state = matches[i]
                    output = apply_state(state, input_objects[input_index])
                    outputs.append(output)
                return (input_grid, outputs)

            return new_state, apply_state_grid_and_objects

        logger.info(f"{'  ' * nesting_level}TODO: more cases of match_list_of_objects")

        return None


list_xforms: List[XformEntry[GridAndObjects, str]] = [
    XformEntry(ObjectListMatch.match_list_of_objects, 4),
]


def find_xform_for_examples(
    xforms: List[XformEntry[Object, State]],
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[State, Object]]:
    logger.info(
        f"\n{'  ' * nesting_level}find_xform_for_examples examples:{len(examples)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    for xform in xforms:
        if Config.difficulty < xform.difficulty:
            continue
        func = xform.xform
        logger.debug(f"{'  ' * nesting_level}Checking xform {func.__name__}")
        match = func(examples, task_name, nesting_level + 1)
        if match is not None:
            logger.info(
                f"{'  ' * nesting_level}Xform {xform.xform.__name__} state:{match[0]} is correct for examples"
            )
            return match
        else:
            logger.info(
                f"{'  ' * nesting_level}Xform {func.__name__} is not applicable"
            )

    return None


def find_xform(
    xforms: List[XformEntry[Object, State]],
    examples: List[Example[Object]],
    tests: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[State, Object]]:
    logger.info(
        f"\n{'  ' * nesting_level}find_xform examples:{len(examples)} tests:{len(tests)} task_name:{task_name} nesting_level:{nesting_level}"
    )

    match = find_xform_for_examples(xforms, examples, task_name, nesting_level)
    if match is None:
        return None

    state, apply_state = match

    for i, test_example in enumerate(tests):
        test_input = test_example[0]
        test_output = test_example[1]
        result_on_test = apply_state(state, test_input)
        if result_on_test != test_output:
            logger.info(f"Xform state:{state} failed for test input {i}")
            if False:
                display(
                    test_output,
                    result_on_test,
                    title=f"Ex{i} state:{state}",
                )
            return None

    return match


# ObjectMatch is a type alias representing a match between a list of detected input objects
# and the index of the object within that list that is identical to the output object.
#
# The first element of the tuple (List[Object]) contains all the detected input objects,
# while the second element (int) specifies the index of the object in this list that is
# identical to the output object in terms of size and data.
ObjectMatch = Tuple[List[Object], int]


def detect_common_features(matched_objects: List[ObjectMatch], initial_difficulty: int):
    def detect_common_symmetry_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [detect_symmetry_features(obj) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Symmetry): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Symmetry)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_color_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [
                detect_color_features(obj, input_objects) for obj in input_objects
            ]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Color): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Color)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_shape_features(level: int) -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [
                detect_shape_features(obj, input_objects, level)
                for obj in input_objects
            ]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                logger.debug(f"  Decision rule (Shape): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule
                    )
                    if common_decision_rule is None:
                        break
            else:
                logger.debug(f"  No decision rule found (Shape)")
                common_decision_rule = None
                break
        return common_decision_rule

    common_decision_rule = None
    features_used = None

    # Try detecting common features in the order of shape, color, and symmetry

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 1:
        common_decision_rule = detect_common_shape_features(initial_difficulty + 1)
        features_used = "Shape"

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 2:
        common_decision_rule = detect_common_color_features()
        features_used = "Color"

    if common_decision_rule is None and Config.difficulty >= initial_difficulty + 3:
        common_decision_rule = detect_common_symmetry_features()
        features_used = "Symmetry"
    assert num_difficulties_matching == 3

    return common_decision_rule, features_used


def find_matched_objects(
    examples: List[Example], task_type: str
) -> Optional[List[ObjectMatch]]:
    """
    Identifies and returns a list of matched input objects that correspond to the output objects
    in the given examples. For each example, it detects candidate objects in the input grid
    and matches them with the output grid based on size and data. If all examples have a match,
    the function returns the list of matched objects; otherwise, it returns None.

    Args:
        examples: A list of examples, each containing an input and output grid.
        task_type: A string indicating the type of task (e.g., 'train' or 'test').

    Returns:
        A list of ObjectMatch tuples if matches are found for all examples, otherwise None.
    """

    def candidate_objects_for_matching(input: Object, output: Object) -> List[Object]:
        """
        Detects objects in the input grid that are candidates for matching the output grid.
        """
        if output.has_frame():
            # If the output is a frame, detect objects in the input as frames
            logger.debug("  Output is a frame")
        num_colors_output = len(output.get_colors(allow_black=True))
        return find_rectangular_objects(input, allow_multicolor=num_colors_output > 1)

    def find_matching_input_object(
        input_objects: List[Object], output: Object
    ) -> Optional[int]:
        for i, io in enumerate(input_objects):
            if io.size == output.size and np.array_equal(io._data, output._data):
                logger.debug(f"  Input object matching output: {io}")
                return i
        return None

    def get_matched_objects(examples: List[Example]) -> Optional[List[ObjectMatch]]:
        matched_objects: List[ObjectMatch] = []

        for example in examples:
            input = example[0]
            output = example[1]
            logger.info(f"  {task_type} {input.size} -> {output.size}")

            input_objects = candidate_objects_for_matching(input, output)
            matched_object_index = find_matching_input_object(input_objects, output)

            if matched_object_index is not None:
                matched_objects.append((input_objects, matched_object_index))

        return matched_objects if len(matched_objects) == len(examples) else None

    matched_objects = get_matched_objects(examples)
    return matched_objects


num_difficulties_xform = max(xform.difficulty for xform in gridxforms)
num_difficulties_matching = 3
num_difficulties_total = num_difficulties_xform + num_difficulties_matching


def process_tasks(tasks: Tasks, set: str):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in tasks.items():
        Config.display_this_task = False
        if Config.task_name and task_name != Config.task_name:
            continue
        if (
            filter_simple_xforms(task, task_name) == False
            and Config.only_simple_examples
            and task_name not in Config.whitelisted_tasks
        ):
            continue
        logger.info(f"\n***Task: {task_name} {set}***")

        examples = task.train

        tests = task.test
        task_type = "train"

        if True:
            current_difficulty = 0

            if Config.find_xform:
                correct_xform = find_xform(gridxforms, examples, tests, task_name, 0)
                if correct_xform is not None:
                    num_correct += 1
                    continue

            current_difficulty += num_difficulties_xform

            if Config.find_matched_objects:
                # Check if the input objects can be matched to the output objects
                logger.debug(f"Checking common features for {task_name} {set}")
                matched_objects = find_matched_objects(examples, task_type)
                if matched_objects:
                    # If the input objects can be matched to the output objects, try to detect common features
                    # to determine the correct object to pick
                    logger.debug(
                        f"XXX Matched {len(matched_objects)}/{len(examples)} {task_name} {set}"
                    )
                    common_decision_rule, features_used = detect_common_features(
                        matched_objects, current_difficulty
                    )
                    if common_decision_rule:
                        logger.info(
                            f"Common decision rule ({features_used}): {common_decision_rule}"
                        )
                        num_correct += 1
                        continue
                    else:
                        logger.warning(
                            f"Could not find common decision rule for {task_name} {set}"
                        )
            current_difficulty += num_difficulties_matching

            if Config.display_not_found:
                Config.display_this_task = True
            if Config.display_this_task:
                grids = [(example[0], example[1]) for example in examples]
                display_multiple(grids, title=f"{task_name} {set}")

            # If no valid dimensions could be determined, give up
            logger.warning(
                f"Could not find correct transformation for {task_name} {set} examples"
            )
            num_incorrect += 1

    return num_correct, num_incorrect


def compute_perc_correct(num_correct: int, num_incorrect: int) -> Optional[float]:
    if num_correct + num_incorrect > 0:
        return int(1000 * num_correct / (num_correct + num_incorrect)) / 10
    return None


def simple():
    num_correct_tr, num_incorrect_tr = process_tasks(training_data, "training_data")
    num_correct_ev, num_incorrect_ev = process_tasks(evaluation_data, "evaluation_data")
    perc_correct_tr = compute_perc_correct(num_correct_tr, num_incorrect_tr)
    perc_correct_ev = compute_perc_correct(num_correct_ev, num_incorrect_ev)

    def log_evaluation_results(set: str, num_correct: int, num_incorrect: int):
        perc_correct = compute_perc_correct(num_correct, num_incorrect)
        if perc_correct is not None:
            logger.error(
                f"{set.capitalize()} data: "
                f"Correct: {num_correct}, Incorrect: {num_incorrect}, Score: {perc_correct}%"
            )

    logger.error("\n***Summary***")
    log_evaluation_results("training", num_correct_tr, num_incorrect_tr)
    log_evaluation_results("evaluation", num_correct_ev, num_incorrect_ev)

    # Write summary of results to JSON file
    with open("simple.json", "w") as f:
        f.write(
            f'{{"training_data":{perc_correct_tr},"evaluation_data":{perc_correct_ev}}}'
        )


if __name__ == "__main__":
    simple()