from typing import List, TypeVar
from grid_data import logger

T = TypeVar('T')


def calculate_z_array(lst: List[T]) -> List[int]:
    """
    Calculate the Z-array for a given list of elements.

    Args:
    lst (list): The input list of elements.

    Returns:
    list: The Z-array where each element at index i represents the length
          of the longest sub-list starting from i which is also a prefix
          of the list lst.
    """
    z = [0] * len(lst)
    z[0] = len(lst)

    # Initialize the right and left pointers of the Z-box
    right, left = 0, 0

    # Loop through each element in the list, starting from the second one
    for i in range(1, len(lst)):
        if i > right:
            # If i is outside the current Z-box, calculate the Z value naively
            left, right = i, i
            while right < len(lst) and lst[right] == lst[right - left]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            # If i is within the Z-box, use previously computed values
            k = i - left
            if z[k] < right - i + 1:
                z[i] = z[k]
            else:
                left = i
                while right < len(lst) and lst[right] == lst[right - left]:
                    right += 1
                z[i] = right - left
                right -= 1
    return z


def find_shortest_period(lst: List[T]) -> int:
    """
    Find the shortest period of a list using the Z-algorithm.

    Args:
    lst (list): The input list of elements.

    Returns:
    int: The length of the shortest period of the list.
    """
    z_array = calculate_z_array(lst)
    n = len(lst)

    for i in range(1, n):
        if z_array[i] == n - i:
            return i
    return n


def test():
    examples = [
        [1, 2, 1, 3, 1, 2],
        [1, 2, 1, 2, 1, 2],
        [1, 2, 1, 1, 2, 1],

    ]
    for ex in examples:
        period = find_shortest_period(ex)
        sublist = ex[:period]
        logger.info(f"The shortest period of the list {ex} is: {sublist}")
