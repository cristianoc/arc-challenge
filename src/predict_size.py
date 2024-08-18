from typing import Callable, List, Optional, Tuple
from grid import Grid
from load_data import Example, Tasks, iter_tasks, training_data, evaluation_data


Size = Tuple[int, int]
SizeExamples = List[Tuple[Size, Size]]
SizeXform = Callable[[SizeExamples, Size], Size]

identity_xform: SizeXform = lambda examples, size: size
always_same_output_xform: SizeXform = lambda examples, size: examples[0][1]

xforms = [identity_xform, always_same_output_xform]


def check_xform_on_examples(xform: SizeXform, examples: List[Example]):
    size_examples = [(Grid(example['input']).size, Grid(
        example['output']).size) for example in examples]
    for example in examples:
        input = Grid(example['input'])
        output = Grid(example['output'])
        input_size = input.size
        output_size = output.size
        new_output_size = xform(size_examples, input_size)
        if new_output_size != output_size:
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
        f"Training data Correct:{num_correct_tr}, Incorrect:{num_incorrect_tr}, Score:{int(100 * num_correct_tr / (num_correct_tr + num_incorrect_tr))}%")
    if num_correct_ev is not None and num_incorrect_ev is not None:
        print(
            f"Evaluation data Correct:{num_correct_ev}, Incorrect:{num_incorrect_ev}, Score:{int(100 * num_correct_ev / (num_correct_ev + num_incorrect_ev))}%")


if __name__ == "__main__":
    predict_sizes()
