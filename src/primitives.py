from bi_types import Match, Example, Object, Primitive, Xform
from logger import logger
from typing import List, Optional, Callable


def check_primitive_on_examples(
    prim: Callable[[Object, str, int], Object],
    examples: List[Example[Object]],
    task_name: str,
    nesting_level: int,
) -> Optional[Match[Object]]:
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
    return (state, lambda i: prim(i, task_name, nesting_level))


def primitive_to_xform(primitive: Primitive) -> Xform[Object]:
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
    obj = input.translate(dx=0, dy=1)
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
                state = f"({p1.__name__}, {p2.__name__})"
                solve = lambda input: p2(
                    p1(input, task_name, nesting_level + 1),
                    task_name,
                    nesting_level + 1,
                )
                return (state, solve)
    return None
