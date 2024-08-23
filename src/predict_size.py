from typing import Callable, List, Optional, Tuple

from color_features import detect_color_features
from visual_cortex import find_largest_frame, is_frame_part_of_lattice
from grid import Grid
from grid_data import Object, display
from load_data import Example, Tasks, iter_tasks, training_data, evaluation_data
from numeric_features import detect_numeric_features, pretty_print_numeric_features
from rule_based_selector import DecisionRule, Features, select_object_minimal
from shape_features import detect_shape_features
from solve_integer_program import find_weights_and_bias
from symmetry_features import detect_symmetry_features


Size = Tuple[int, int]
ExampleGrids = List[Tuple[Grid, Grid]]
SizeXform = Callable[[ExampleGrids, Grid, str], Optional[Size]]

# returns the index of the object to pick
ObjectPicker = Callable[[List[Object]], int]


Debug = False


def identity_xform(grids: ExampleGrids, grid: Grid, task_name: str):
    return grid.size


def always_same_output_xform(grids: ExampleGrids, grid: Grid, task_name: str):
    return grids[0][1].size


def size_of_largest_object_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    objects = grid.detect_objects()
    if not objects:
        return (0, 0)
    largest_object = max(objects, key=lambda obj: obj.num_cells)
    return largest_object.size


def find_frame_objects(grid: Grid, objects: List[Object], task_name: str) -> List[Object]:
    frame_objects: List[Object] = []
    for obj in objects:
        if obj.size != grid.size and obj.has_frame():
            frame_objects.append(obj)
            continue
        foreground = obj.main_color
        frame = find_largest_frame(obj.data, foreground)

        if frame:
            (top, left, bottom, right) = frame
            width = right - left + 1
            height = bottom - top + 1
            if Debug and width >= 2 and height >= 2:
                print(
                    f"Frame found: {frame} height:{height} width:{width} foreground:{foreground}")
            data_for_frame = [row[left:right+1]
                              for row in obj.data[top:bottom+1]]
            obj_for_frame = Object(
                (obj.origin[0] + top, obj.origin[1] + left), data_for_frame)
            if obj_for_frame.has_frame():
                frame_objects.append(obj_for_frame)

    return frame_objects


def find_frame_objects_black(grid: Grid, objects: List[Object], task_name: str) -> List[Object]:
    frame_objects: List[Object] = []
    for obj in objects:
        if obj.is_block():
            frame_objects.append(obj)
        continue
    return frame_objects


def output_inside_largest_frame_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    largest_frame = find_largest_frame(grid.data, None)
    if largest_frame:
        (top, left, bottom, right) = largest_frame
        width = right - left + 1
        height = bottom - top + 1
        if Debug and width >= 2 and height >= 2:
            print(
                f"Largest frame found: {largest_frame} height:{height} width:{width}")
        height_without_frame = height - 2
        width_without_frame = width - 2
        return (height_without_frame, width_without_frame)
    return (0, 0)

def output_is_largest_block_object_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    objects = grid.detect_objects(allow_black=True)
    # exclude full grid size
    objects = [obj for obj in objects if obj.size != grid.size and obj.is_block()]
    if not objects:
        return None
    largest_object = max(objects, key=lambda obj: obj.size[0] * obj.size[1])
    return largest_object.size


def output_is_largest_object_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    objects = grid.detect_objects(allow_black=True)
    # exclude full grid size
    objects = [obj for obj in objects if obj.size != grid.size]
    if not objects:
        return None
    largest_object = max(objects, key=lambda obj: obj.size[0] * obj.size[1])
    return largest_object.size


def one_object_is_a_frame_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    # Check that all the output sizes are smaller than the input sizes
    for input_grid, output_grid in grids:
        if output_grid.size >= input_grid.size:
            return (0, 0)

    objects = grid.detect_objects(diagonals=False, allow_black=False)
    frame_objects = find_frame_objects(
        grid=grid, objects=objects, task_name=task_name)
    if Debug:
        print(f"  # of objects: {len(objects)}")
    if Debug:
        print(f"  # of frame objects: {len(frame_objects)}")

    if len(frame_objects) > 1:
        sorted_objects = []
        sorted_objects = sorted(
            frame_objects, key=lambda obj: obj.size[0] * obj.size[1], reverse=True)
        # if there are multiple frame objects, keep the largest one
        if Debug:
            print(f"  Sorted objects: {sorted_objects}")
        frame = sorted_objects[0]
        frame_objects = [frame]

    # Check if there's exactly one frame
    if len(frame_objects) == 1:
        frame = frame_objects[0]

        fr = (frame.origin[0], frame.origin[1], frame.origin[0] +
              frame.height - 1, frame.origin[1] + frame.width - 1)
        is_lattice = is_frame_part_of_lattice(grid.data, fr, frame.main_color)
        if False and is_lattice:
            display(grid.data, title=f"black frame: {fr} lattice")

        h, w = frame.size
        if h > 2 and w > 2:
            # check if all the elements immediately inside the frame are of a different color
            n_diff_color = 0
            for i in range(1, h - 1):
                if frame.data[i][1] == frame.first_color:
                    n_diff_color += 1
                if frame.data[i][w - 2] == frame.first_color:
                    n_diff_color += 1
            for j in range(1, w - 1):
                if frame.data[1][j] == frame.first_color:
                    n_diff_color += 1
                if frame.data[h - 2][j] == frame.first_color:
                    n_diff_color += 1
            if n_diff_color <= 1:
                # Reduce the frame by 1 cell on each side
                if Debug:
                    print(f"  Reducing frame size to {h-2}x{w-2}")
                return (h - 2, w - 2)
            else:
                if Debug:
                    print(f"  Frame size is {h}x{w}")
                return (h, w)
        elif frame.is_block():
            if Debug:
                print(f"  Frame is a block")
            return (h, w)
        else:
            # Handle case where frame is too small to reduce
            if Debug:
                print(
                    f"  Frame is too small to reduce origin:{frame.origin} size:{frame.size}")
            return (0, 0)
    if Debug:
        print("  No frame object found")
    return (0, 0)


def one_object_is_a_black_frame_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    # Check that all the output sizes are smaller than the input sizes
    for input_grid, output_grid in grids:
        if output_grid.size >= input_grid.size:
            return (0, 0)

    objects = grid.detect_objects(diagonals=False, allow_black=True)
    frame_objects = find_frame_objects_black(
        grid=grid, objects=objects, task_name=task_name)
    if Debug:
        print(f"  # of objects: {len(objects)}")
    if Debug:
        print(f"  # of frame objects: {len(frame_objects)}")

    if len(frame_objects) > 1:
        sorted_objects = []
        black_objects: List[Object] = []
        other_objects: List[Object] = []
        for obj in frame_objects:
            if obj.main_color == 0:
                black_objects.append(obj)
            else:
                other_objects.append(obj)
        sorted_objects = sorted(black_objects, key=lambda obj: obj.size[0] * obj.size[1], reverse=True) + sorted(
            other_objects, key=lambda obj: obj.size[0] * obj.size[1], reverse=True)
        # if there are multiple frame objects, keep the largest one
        if Debug:
            print(f"  Sorted objects: {sorted_objects}")
        frame = sorted_objects[0]
        frame_objects = [frame]

    # Check if there's exactly one frame
    if len(frame_objects) == 1:
        frame = frame_objects[0]
        h, w = frame.size
        if h > 2 and w > 2:
            # check if all the elements immediately inside the frame are of a different color
            n_diff_color = 0
            for i in range(1, h - 1):
                if frame.data[i][1] == frame.first_color:
                    n_diff_color += 1
                if frame.data[i][w - 2] == frame.first_color:
                    n_diff_color += 1
            for j in range(1, w - 1):
                if frame.data[1][j] == frame.first_color:
                    n_diff_color += 1
                if frame.data[h - 2][j] == frame.first_color:
                    n_diff_color += 1
            if n_diff_color <= 1:
                # Reduce the frame by 1 cell on each side
                if Debug:
                    print(f"  Reducing frame size to {h-2}x{w-2}")
                return (h - 2, w - 2)
            else:
                if Debug:
                    print(f"  Frame size is {h}x{w}")
                return (h, w)
        elif frame.is_block():
            if Debug:
                print(f"  Frame is a block")
            return (h, w)
        else:
            # Handle case where frame is too small to reduce
            if Debug:
                print(
                    f"  Frame is too small to reduce origin:{frame.origin} size:{frame.size}")
            return (0, 0)
    if Debug:
        print("  No frame object found")
    return (0, 0)


def size_is_multiple_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    """
    Determines if the given grid can be scaled by consistent ratios derived from example grids.
    The function checks if applying these ratios to the grid's size results in integer dimensions.

    If the transformation is valid and consistent across all example grids, it returns the new size.
    Otherwise, it returns (0, 0).
    """
    ratios_height: List[float] = []
    ratios_width: List[float] = []
    for input_grid, output_grid in grids:
        ratios_height.append(output_grid.size[0] / input_grid.size[0])
        ratios_width.append(output_grid.size[1] / input_grid.size[1])

    # Check if applying the ratios to the grid's size results in integers
    transformed_height = [ratio * grid.size[0] for ratio in ratios_height]
    transformed_width = [ratio * grid.size[1] for ratio in ratios_width]

    if all(height.is_integer() for height in transformed_height) and all(width.is_integer() for width in transformed_width):
        # Ensure all ratios are the same
        if all(ratio == ratios_height[0] for ratio in ratios_height) and all(ratio == ratios_width[0] for ratio in ratios_width):
            return (int(ratios_height[0] * grid.size[0]), int(ratios_width[0] * grid.size[1]))

    return (0, 0)


# check if the size is a multiple determined by the number of colors
def size_is_multiple_determined_by_colors_xform(grids: ExampleGrids, grid: Grid, task_name: str) -> Optional[Size]:
    ncolors = 0
    h = grid.height
    w = grid.width
    colors = grid.get_colors()
    ncolors = len(colors)
    return (h * ncolors, w * ncolors)


xforms = [
    identity_xform, always_same_output_xform, size_of_largest_object_xform,
    size_is_multiple_xform, size_is_multiple_determined_by_colors_xform,
    output_inside_largest_frame_xform,
    output_is_largest_block_object_xform,
    output_is_largest_object_xform,
    one_object_is_a_frame_xform,
    one_object_is_a_black_frame_xform
]


def check_xform_on_examples(xform: SizeXform, examples: List[Example], task_name: str, task_type: str) -> bool:
    grids = [(Grid(example['input']), Grid(example['output']))
             for example in examples]
    if Debug:
        print(f"Checking xform {xform.__name__} {task_type}")
    for i, example in enumerate(examples):
        if Debug:
            print(f"  Example {i+1}/{len(examples)}")
        input = Grid(example['input'])
        output = Grid(example['output'])
        new_output_size = xform(grids, input, task_name)
        if new_output_size != output.size:
            if Debug:
                print(f"  Example {i+1} failed")
            return False
    return True


# ObjectMatch is a type alias representing a match between a list of detected input objects
# and the index of the object within that list that is identical to the output object.
#
# The first element of the tuple (List[Object]) contains all the detected input objects,
# while the second element (int) specifies the index of the object in this list that is
# identical to the output object in terms of size and data.
ObjectMatch = Tuple[List[Object], int]


def detect_common_features(matched_objects: List[ObjectMatch], debug: bool = False):
    def detect_common_symmetry_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [detect_symmetry_features(
                obj.data) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                if debug:
                    print(f"  Decision rule (Symmetry): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule)
                    if common_decision_rule is None:
                        break
            else:
                if debug:
                    print(f"  No decision rule found (Symmetry)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_color_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [detect_color_features(
                obj, input_objects, debug) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                if debug:
                    print(f"  Decision rule (Color): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule)
                    if common_decision_rule is None:
                        break
            else:
                if debug:
                    print(f"  No decision rule found (Color)")
                common_decision_rule = None
                break
        return common_decision_rule

    def detect_common_shape_features() -> Optional[DecisionRule]:
        common_decision_rule = None
        for input_objects, index in matched_objects:
            embeddings = [detect_shape_features(
                obj, input_objects, debug) for obj in input_objects]
            decision_rule = select_object_minimal(embeddings, index)
            if decision_rule is not None:
                if debug:
                    print(f"  Decision rule (Shape): {decision_rule}")
                if common_decision_rule is None:
                    common_decision_rule = decision_rule
                else:
                    common_decision_rule = common_decision_rule.intersection(
                        decision_rule)
                    if common_decision_rule is None:
                        break
            else:
                if debug:
                    print(f"  No decision rule found (Shape)")
                common_decision_rule = None
                break
        return common_decision_rule

    # Try detecting common features in the order of shape, symmetry, and color
    common_decision_rule = detect_common_shape_features()
    features_used = "Shape"

    if common_decision_rule is None:
        common_decision_rule = detect_common_symmetry_features()
        features_used = "Symmetry"

    if common_decision_rule is None:
        common_decision_rule = detect_common_color_features()
        features_used = "Color"

    return common_decision_rule, features_used


def process_tasks(tasks: Tasks, set: str):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in iter_tasks(tasks):
        # if task_name != "9f236235.json":
        #     continue
        print(f"\n***Task: {task_name} {set}***")

        for task_type, examples in task.items():
            if task_type not in ['train', 'test']:
                continue
            if task_type == 'test':
                continue
            # check if at least one xform is correct
            correct_xform = None
            for xform in xforms:
                if check_xform_on_examples(xform, examples, task_name, task_type):
                    if False and xform == output_is_largest_object_xform:
                        title = f"{xform.__name__} ({task_name})"
                        print(title)
                        for i, e in enumerate(examples):
                            display(e['input'], output=e['output'],
                                    title=f"Ex{i+1} " + title)
                    correct_xform = xform
                    break
            if correct_xform:
                print(
                    f"Xform {correct_xform.__name__} is correct for all examples in {task_type}")
                test_examples = [examples for task_type,
                                 examples in task.items() if task_type == 'test']
                for i, test_example in enumerate(test_examples):
                    if not check_xform_on_examples(correct_xform, test_example, task_name, 'test'):
                        print(
                            f"Xform {correct_xform.__name__} failed for test example {i}")
                        correct_xform = None
                        break
            if correct_xform:
                print(
                    f"Xform {correct_xform.__name__} is correct for all examples in {task_type} and test")
                num_correct += 1
                continue

            print(f"Checking common features for {task_name} {set}")

            def candidate_objects_for_matching(input: Grid, output: Grid) -> List[Object]:
                """
                Detects objects in the input grid that are candidates for matching the output grid.
                """
                output_as_object = Object((0, 0), output.data)
                if output_as_object.has_frame():
                    # If the output is a frame, detect objects in the input as frames
                    print("  Output is a frame")
                num_colors_output = len(output.get_colors())
                return input.detect_rectangular_objects(allow_multicolor=num_colors_output > 1, debug=Debug)

            def find_matching_input_object(input_objects: List[Object], output: Grid) -> Optional[int]:
                for i, io in enumerate(input_objects):
                    if io.size == output.size and io.data == output.data:
                        if Debug:
                            print(f"  Input object matching output: {io}")
                        return i
                return None

            def get_matched_objects(examples: List[Example]) -> Optional[List[ObjectMatch]]:
                matched_objects: List[ObjectMatch] = []

                for example in examples:
                    input = Grid(example['input'])
                    output = Grid(example['output'])
                    print(f"  {task_type} {input.size} -> {output.size}")

                    input_objects = candidate_objects_for_matching(
                        input, output)
                    matched_object_index = find_matching_input_object(
                        input_objects, output)

                    if matched_object_index is not None:
                        matched_objects.append(
                            (input_objects, matched_object_index))

                return matched_objects if len(matched_objects) == len(examples) else None

            matched_objects = get_matched_objects(examples)
            # Check if all examples are matched
            if matched_objects:
                if Debug:
                    print(
                        f"XXX Matched {len(matched_objects)}/{len(examples)} {task_name} {set}")
                common_decision_rule, features_used = detect_common_features(
                    matched_objects, debug=Debug)
                print(
                    f"Common decision rule ({features_used}): {common_decision_rule}")
                if not common_decision_rule:
                    # rule to choose which input object to pick was not found
                    assert False
                num_correct += 1
                continue

            print(
                f"Trying to determine dimensions via LP for {task_name} {set}")

            # Attempt to determine width and height using linear programming before giving up
            feature_vectors: List[Features] = []
            target_heights: List[int] = []
            target_widths: List[int] = []

            for example in examples:
                input_grid = Grid(example['input'])
                output_grid = Grid(example['output'])

                input_features = detect_numeric_features(input_grid)
                target_height, target_width = output_grid.size

                feature_vectors.append(input_features)
                target_heights.append(target_height)
                target_widths.append(target_width)

            predicted_height = find_weights_and_bias(
                feature_vectors, target_heights)
            predicted_width = find_weights_and_bias(
                feature_vectors, target_widths)

            if predicted_height and predicted_width:
                print(
                    f"Predictions via LP: out.height=={pretty_print_numeric_features(predicted_height)}, out.width=={pretty_print_numeric_features(predicted_width)}")
                num_correct += 1
            else:
                # If no valid dimensions could be determined, give up
                print(
                    f"Could not find correct transformation or determine dimensions via LP for {task_name} {set} examples")
                num_incorrect += 1

            # grids: List[Tuple[GridData, Optional[GridData]]] = [
            #     (Grid(example['input']).data, Grid(example['output']).data) for example in examples
            # ]
            # display_multiple(
            #     grids, title=f"Task: {task_name} {set} matched_objects:{matched_objects}/{len(examples)}")
    return num_correct, num_incorrect


def predict_sizes():
    num_correct_tr, num_incorrect_tr = process_tasks(
        training_data, "traing_data")
    do_eval = True
    num_correct_ev: Optional[int] = None
    num_incorrect_ev: Optional[int] = None
    if do_eval:
        num_correct_ev, num_incorrect_ev = process_tasks(
            evaluation_data, "evaluation_data")
    print(
        f"\nTraining data Correct:{num_correct_tr}, Incorrect:{num_incorrect_tr}, Score:{int(1000 * num_correct_tr / (num_correct_tr + num_incorrect_tr))/10}%")
    if num_correct_ev is not None and num_incorrect_ev is not None:
        print(
            f"Evaluation data Correct:{num_correct_ev}, Incorrect:{num_incorrect_ev}, Score:{int(1000 * num_correct_ev / (num_correct_ev + num_incorrect_ev))/10}%")


if __name__ == "__main__":
    predict_sizes()
