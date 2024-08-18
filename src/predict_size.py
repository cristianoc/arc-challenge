from typing import Callable, List, Optional, Tuple

from grid import Grid
from grid_data import Object, display
from load_data import Example, Tasks, iter_tasks, training_data, evaluation_data


Size = Tuple[int, int]
ExampleGrids = List[Tuple[Grid, Grid]]
SizeXform = Callable[[ExampleGrids, Grid], Size]

identity_xform: SizeXform = lambda grids, grid: grid.size
always_same_output_xform: SizeXform = lambda grids, grid: grids[0][1].size


def one_object_is_a_frame_xform_(grids: ExampleGrids, grid: Grid, allow_black: bool):
    # Check that all the output sizes are smaller than the input sizes
    for input_grid, output_grid in grids:
        if output_grid.size >= input_grid.size:
            return (0, 0)

    objects = grid.detect_objects(diagonals=False, allow_black=allow_black)

    frame_objects = [obj for obj in objects if obj.size !=
                     grid.size and obj.has_frame() or (allow_black and obj.is_block())]
    shrunk_objects: List[Object] = []
    if len(frame_objects) == 0:
        for obj in objects:
            color = obj.main_color
            threshold = 0.2

            while obj.width > 2 and obj.height > 2:
                # Check the leftmost column and remove it if the number of cells of the color is less than the threshold
                left_col = [row[0] for row in obj.data]
                size_before = obj.size
                if left_col.count(color) <= 2 or left_col.count(color) < obj.height * threshold:
                    obj = Object(
                        (obj.origin[0], obj.origin[1] + 1), [row[1:] for row in obj.data])
                    print(f"Shrinking left size: {size_before} -> {obj.size}")
                    continue

                # Check the rightmost column and remove it if the number of cells of the color is less than the threshold
                right_col = [row[-1] for row in obj.data]
                if right_col.count(color) <= 2 or right_col.count(color) < obj.height * threshold:
                    obj = Object((obj.origin[0], obj.origin[1]), [
                                 row[:-1] for row in obj.data])
                    print(f"Shrinking right size: {size_before} -> {obj.size}")
                    continue

                # Check the topmost row and remove it if the number of cells of the color is less than the threshold
                if obj.data[0].count(color) <= 2 or obj.data[0].count(color) < obj.width * threshold:
                    obj = Object(
                        (obj.origin[0] + 1, obj.origin[1]), obj.data[1:])
                    print(f"Shrinking top size: {size_before} -> {obj.size}")
                    continue

                # Check the bottommost row and remove it if the number of cells of the color is less than the threshold
                if obj.data[-1].count(color) <= 2 or obj.data[-1].count(color) < obj.width * threshold:
                    obj = Object((obj.origin[0], obj.origin[1]), obj.data[:-1])
                    print(
                        f"Shrinking bottom size: {size_before} -> {obj.size}")
                    continue
                break

            if obj.has_frame():
                shrunk_objects.append(obj)
    if len(shrunk_objects) >= 1:
        # display(grid.data, title="Grid with shrunk objects")
        frame_objects = shrunk_objects
    print(f"# of objects: {len(objects)}")
    print(f"# of frame objects: {len(frame_objects)}")

    if len(frame_objects) > 1:
        sorted_objects = []
        if allow_black:
            black_objects: List[Object] = []
            other_objects: List[Object] = []
            for obj in frame_objects:
                if obj.main_color == 0:
                    black_objects.append(obj)
                else:
                    other_objects.append(obj)
            sorted_objects = sorted(black_objects, key=lambda obj: obj.size[0] * obj.size[1], reverse=True) + sorted(
                other_objects, key=lambda obj: obj.size[0] * obj.size[1], reverse=True)
        else:
            sorted_objects = sorted(
                frame_objects, key=lambda obj: obj.size[0] * obj.size[1], reverse=True)
        # if there are multiple frame objects, keep the largest one
        print(f"Sorted objects: {sorted_objects}")
        frame = sorted_objects[0]
        frame_objects = [frame]

    # Check if there's exactly one frame
    if len(frame_objects) == 1:
        frame = frame_objects[0]
        if len(shrunk_objects) >= 1:
            print(f"Frame object: {frame}")
        h, w = frame.size
        if h > 2 and w > 2:
            # check if all the elements immediately inside the frame are of a different color
            if all(frame.data[1][j] != frame.first_color for j in range(1, w - 1)) and \
                    all(frame.data[h - 2][j] != frame.first_color for j in range(1, w - 1)) and \
                    all(frame.data[i][1] != frame.first_color for i in range(1, h - 1)) and \
                    all(frame.data[i][w - 2] != frame.first_color for i in range(1, h - 1)):
                # Reduce the frame by 1 cell on each side
                print(f"Reducing frame size to {h-2}x{w-2}")
                return (h - 2, w - 2)
            else:
                print(f"Frame size is {h}x{w}")
                return (h, w)
        elif frame.is_block():
            print(f"Frame is a block")
            return (h, w)
        else:
            # Handle case where frame is too small to reduce
            print(
                f"Frame is too small to reduce origin:{frame.origin} size:{frame.size}")
            return (0, 0)
    print("No frame object found")
    return (0, 0)


def one_object_is_a_frame_xform_noblack(grids: ExampleGrids, grid: Grid):
    return one_object_is_a_frame_xform_(grids, grid, allow_black=False)


def one_object_is_a_frame_xform_black(grids: ExampleGrids, grid: Grid):
    return one_object_is_a_frame_xform_(grids, grid, allow_black=True)


def size_is_multiple_xform(grids: ExampleGrids, grid: Grid):
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
def size_is_multiple_determined_by_colors_xform(grids: ExampleGrids, grid: Grid):
    ncolors = 0
    h = grid.height
    w = grid.width
    colors = grid.get_colors()
    # remove 0 if present
    colors = [c for c in colors if c != 0]
    ncolors = len(colors)
    return (h * ncolors, w * ncolors)


xforms = [identity_xform, always_same_output_xform,
          size_is_multiple_xform, size_is_multiple_determined_by_colors_xform, one_object_is_a_frame_xform_noblack, one_object_is_a_frame_xform_black]


def check_xform_on_examples(xform: SizeXform, examples: List[Example]):
    grids = [(Grid(example['input']), Grid(example['output']))
             for example in examples]
    for example in examples:
        input = Grid(example['input'])
        output = Grid(example['output'])
        new_output_size = xform(grids, input)
        if new_output_size != output.size:
            return False
    return True


def iter_over_tasks(tasks: Tasks):
    num_correct = 0
    num_incorrect = 0
    for task_name, task in iter_tasks(tasks):
        print(f"Task: {task_name}")
        for task_type, examples in task.items():
            if task_type not in ['train', 'test']:
                continue
            # check if at least one xform is correct
            for xform in xforms:
                if check_xform_on_examples(xform, examples):
                    if xform == one_object_is_a_frame_xform_black:
                        title = f"Size determined by frame ({task_name})"
                        print(title)
                        display(examples[0]['input'],
                                output=examples[0]['output'], title=title)
                    num_correct += 1
                    break
            else:
                num_incorrect += 1
                print(f"\n***Task: {task_name} {task_type}***")
                print(
                    f"Could not find correct xform for {task_name} {task_type} examples")
                for example in examples:
                    input = Grid(example['input'])
                    output = Grid(example['output'])
                    print(f"  {task_type} {input.size} -> {output.size}")
                    input_objects = input.detect_objects()
                    output_objects = output.detect_objects()
                    input_sizes = [obj.size for obj in input_objects]
                    output_sizes = [obj.size for obj in output_objects]
                    input_colors = input.get_colors()
                    output_colors = output.get_colors()
                    print(f"  Input sizes: {input_sizes}")
                    print(f"  Output sizes: {output_sizes}")
                    print(f"  Input colors: {input_colors}")
                    print(f"  Output colors: {output_colors}")
    return num_correct, num_incorrect


def predict_sizes():
    num_correct_tr, num_incorrect_tr = iter_over_tasks(training_data)
    do_eval = True
    num_correct_ev: Optional[int] = None
    num_incorrect_ev: Optional[int] = None
    if do_eval:
        num_correct_ev, num_incorrect_ev = iter_over_tasks(evaluation_data)
    print(
        f"Training data Correct:{num_correct_tr}, Incorrect:{num_incorrect_tr}, Score:{int(1000 * num_correct_tr / (num_correct_tr + num_incorrect_tr))/10}%")
    if num_correct_ev is not None and num_incorrect_ev is not None:
        print(
            f"Evaluation data Correct:{num_correct_ev}, Incorrect:{num_incorrect_ev}, Score:{int(1000 * num_correct_ev / (num_correct_ev + num_incorrect_ev))/10}%")


if __name__ == "__main__":
    predict_sizes()
