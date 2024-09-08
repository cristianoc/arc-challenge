import logging
from enum import Enum
from typing import List, Tuple, NewType
from enum import Enum, auto
from typing import NamedTuple

Size = Tuple[int, int]
Cell = Tuple[int, int]
GridData = List[List[int]]

# Directions for moving in the grid: right, left, down, up
DIRECTIONS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# Direction vectors for 8 directions (N, NE, E, SE, S, SW, W, NW)
DIRECTIONS8 = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

from enum import Enum, auto
from typing import NamedTuple


class XReflection(Enum):
    NONE = False
    REFLECT = True


class ClockwiseRotation(Enum):
    R0 = 0  # 0 degrees
    R90 = 1  # 90 degrees
    R180 = 2  # 180 degrees
    R270 = 3  # 270 degrees


class RigidTransformation(NamedTuple):
    """
    A rigid transformation on the grid, represented by a rotation (R) and an optional reflection (X).

    **Laws:**

    1. **Rotation Composition**:
       R_n R_m = R_(n+m)

    2. **Reflection Composition**:
       X X = R_0

    3. **Rotation and Reflection Composition**:
       X R_n = R_(-n) X
    """

    rotation: ClockwiseRotation = ClockwiseRotation.R0
    x_reflection: XReflection = XReflection.NONE

    def __str__(self):
        return f"R{self.rotation.value}{'X' if self.x_reflection == XReflection.REFLECT else ''}"

    def inverse(self) -> "RigidTransformation":
        """
        Compute the inverse of this rigid transformation.

        **Mathematical Justification**:
        - For T = Rn * X, the inverse is itself: T_inv = T
            because Rn X Rn X = R_n R_{-n} X X = R0.
        - For T = Rn, the inverse is T_inv = R(-n).
        """
        if self.x_reflection == XReflection.REFLECT:
            # If there's a reflection, keep the same rotation
            rotation = self.rotation
        else:
            # If no reflection, invert the rotation
            rotation = ClockwiseRotation((-self.rotation.value) % 4)

        # Keep the same reflection state
        x_reflection = self.x_reflection

        return RigidTransformation(rotation, x_reflection)

    def compose_with(self, next: "RigidTransformation") -> "RigidTransformation":
        """
        Compose this transformation with another. If self is T and next is U, then the result is T U.

        **Mathematical Justification**:
        There are two cases depending on whether the first reflection is present:
        - For T = R_n X and U = R_m X_i:
          T U = R_n X R_m X_i = R_n R_{-m} X X_i = R_(n-m) X_{not i}
        - For T = R_n and U = R_m X_i:
          T U = R_n R_m X_i = R_(n+m) X_i
        """
        if self.x_reflection.value == True:
            # Case 1: R_n X R_m X_i
            combined_rotation = ClockwiseRotation(
                (self.rotation.value - next.rotation.value) % 4
            )
            combined_reflection = XReflection(not next.x_reflection.value)
        else:
            # Case 2: R_n R_m X_i
            combined_rotation = ClockwiseRotation(
                (self.rotation.value + next.rotation.value) % 4
            )
            combined_reflection = next.x_reflection

        return RigidTransformation(combined_rotation, combined_reflection)


class Rotation(str, Enum):
    CLOCKWISE = "Clockwise"
    COUNTERCLOCKWISE = "CounterClockwise"


class Axis(str, Enum):
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"

Color = NewType("Color", int)

# Define the custom color scheme as a list of colors
color_scheme = [
    "#000000",  # black
    "#0074D9",  # blue
    "#FF4136",  # red
    "#2ECC40",  # green
    "#FFDC00",  # yellow
    "#AAAAAA",  # grey
    "#F012BE",  # fuschia
    "#FF851B",  # orange
    "#7FDBFF",  # ligthblue
    "#870C25",  # brown
]

# Definitions using the indices
BLACK: Color = Color(0)  # #000000
BLUE: Color = Color(1)  # #0074D9
RED: Color = Color(2)  # #FF4136
GREEN: Color = Color(3)  # #2ECC40
YELLOW: Color = Color(4)  # #FFDC00
GREY: Color = Color(5)  # #AAAAAA
FUSCHIA: Color = Color(6)  # F012BE
ORANGE: Color = Color(7)  # #FF851B
LIGHTBLUE: Color = Color(8)  # #7FDBFF
BROWN: Color = Color(9)  # #870C25
