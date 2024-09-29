from typing import List

import config
from bi_types import XformEntry
from objects import Object

gridxforms: List[XformEntry[Object, Object]] = []


# brute force search xforms to be used when all else fails
desperatexforms: List[XformEntry[Object, Object]] = []


def init():
    global gridxforms
    global desperatexforms
    from canvas_grid_match import canvas_grid_xform, equal_modulo_rigid_transformation
    from inpainting_match import (
        inpainting_xform_no_mask,
        inpainting_xform_output_is_block,
        inpainting_xform_with_mask,
    )
    from match_colored_objects import match_colored_objects
    from match_n_objects_with_output import match_n_objects_with_output
    from match_objects_in_grid import match_rectangular_objects_in_grid
    from match_split_with_frame import match_split_with_frame
    from match_subgrids_in_lattice import match_subgrids_in_lattice
    from primitives import primitive_to_xform, translate_down_1, xform_identity

    gridxforms = [
        XformEntry(match_subgrids_in_lattice, 3),
        XformEntry(match_colored_objects, 3),
        XformEntry(xform_identity, 1),
        XformEntry(equal_modulo_rigid_transformation, 2),
        XformEntry(primitive_to_xform(translate_down_1), 2),
        XformEntry(canvas_grid_xform, 2),
        XformEntry(match_rectangular_objects_in_grid, 3),
        XformEntry(inpainting_xform_no_mask, 2),
        XformEntry(inpainting_xform_output_is_block, 2),
        XformEntry(match_n_objects_with_output, 3),
        XformEntry(match_split_with_frame, 3),
    ]
    if config.find_frame_rule:
        gridxforms.append(
            XformEntry(inpainting_xform_with_mask, 2),
        )

    from split_mirrot_match import frame_split_and_mirror_xform

    if config.find_frame_rule:
        desperatexforms.append(XformEntry(frame_split_and_mirror_xform, 100))
