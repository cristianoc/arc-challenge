import logging
from enum import Enum
from typing import List, Tuple, NewType

logging.basicConfig(
    level=logging.INFO,  # change to logging.DEBUG for more verbose output
    format="%(message)s",
)
logger = logging.getLogger(__name__)


Cell = Tuple[int, int]
GridData = List[List[int]]

# Directions for moving in the grid: right, left, down, up
DIRECTIONS4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Direction vectors for 8 directions (N, NE, E, SE, S, SW, W, NW)
DIRECTIONS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


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