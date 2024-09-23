from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Union

from load_data import Example
from objects import Object

GridAndObjects = Tuple[Object, List[Object]]


T1 = TypeVar("T1", bound=Union[Object, GridAndObjects])
T2 = TypeVar("T2", bound=Union[Object, GridAndObjects])

State = str

Primitive = Callable[[Object, str, int], Object]
Match = Tuple[State, Callable[[T1], Optional[T2]]]
Xform = Callable[[List[Example[T1]], str, int], Optional[Match[T2, T2]]]


@dataclass
class XformEntry(Generic[T1, T2]):
    xform: Xform[T1, T2]
    difficulty: int
