from typing import Callable, List, Optional, Tuple
from grid import Grid
from grid_data import display
from load_data import Example, Tasks, iter_tasks, training_data, evaluation_data


Size = Tuple[int, int]
ExampleGrids = List[Tuple[Grid, Grid]]
SizeXform = Callable[[ExampleGrids, Grid], Size]

identity_xform: SizeXform = lambda grids, grid: grid.size
always_same_output_xform: SizeXform = lambda grids, grid: grids[0][1].size


def one_object_is_a_frame_xform(grids: ExampleGrids, grid: Grid):
    # Check that all the output sizes are smaller than the input sizes
    for input_grid, output_grid in grids:
        if output_grid.size >= input_grid.size:
            return (0, 0)

    objects = grid.detect_objects()

    frame_objects = [obj for obj in objects if obj.has_frame()]
    print(f"# of objects: {len(objects)}")
    print(f"# of frame objects: {len(frame_objects)}")

    if len(frame_objects) > 1 and len(frame_objects) <= 3:
        # if there are multiple frame objects, keep the largest one
        frame = max(frame_objects, key=lambda obj: obj.size[0] * obj.size[1])
        frame_objects = [frame]

    # Check if there's exactly one frame
    if len(frame_objects) == 1:
        frame = frame_objects[0]
        h, w = frame.size
        if h > 2 and w > 2:
            # check if all the elements immediately inside the frame are of a different color
            if all(frame.data[1][j] != frame.color for j in range(1, w - 1)) and \
                    all(frame.data[h - 2][j] != frame.color for j in range(1, w - 1)) and \
                    all(frame.data[i][1] != frame.color for i in range(1, h - 1)) and \
                    all(frame.data[i][w - 2] != frame.color for i in range(1, h - 1)):
                # Reduce the frame by 1 cell on each side
                return (h - 2, w - 2)
            else:
                return (h, w)
        else:
            # Handle case where frame is too small to reduce
            return (0, 0)
    return (0, 0)


xforms = [identity_xform, always_same_output_xform,
          one_object_is_a_frame_xform]


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
        for task_type, examples in task.items():
            if task_type not in ['train', 'test']:
                continue
            # check if at least one xform is correct
            for xform in xforms:
                if check_xform_on_examples(xform, examples):
                    if xform == one_object_is_a_frame_xform:
                        display(examples[0]['input'], output=examples[0]['output'], title="Size determined by frame")
                    num_correct += 1
                    break
            else:
                num_incorrect += 1
                print(
                    f"Could not find correct xform for {task_name} {task_type} examples")
                for example in examples:
                    input = Grid(example['input'])
                    output = Grid(example['output'])
                    print(f"  {task_type} {input.size} -> {output.size}")
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
