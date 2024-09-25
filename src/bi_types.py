from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Union

from objects import Object

GridAndObjects = Tuple[Object, List[Object]]


T1 = TypeVar("T1", bound=Union[Object, GridAndObjects, List[Object], int, Tuple[Object, Object], Tuple[Object, ...]])
T2 = TypeVar("T2", bound=Union[Object, GridAndObjects, List[Object], int, Tuple[Object, Object], Tuple[Object, ...]])

State = str

Example = Tuple[T1, T2]  # (input, output)
Examples = List[Example[T1, T2]]
Primitive = Callable[[Object, str, int], Object]
Match = Tuple[State, Callable[[T1], Optional[T2]]]
Xform = Callable[[Examples[T1, T2], str, int], Optional[Match[T1, T2]]]


@dataclass
class XformEntry(Generic[T1, T2]):
    xform: Xform[T1, T2]
    difficulty: int
