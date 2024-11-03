import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # early import to silence initialization log messages

logging.getLogger("pulp").setLevel(logging.ERROR)  # Suppress the verbose output from the pulp solver

logging.basicConfig(
    level=logging.ERROR,  # change to logging.DEBUG for more verbose output
    format="%(message)s",
)
logger = logging.getLogger(__name__)
