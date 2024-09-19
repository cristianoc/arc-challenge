from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Generic,
    Union,
)
from dataclasses import dataclass
from objects import Object, display, display_multiple
from load_data import Example


GridAndObjects = Tuple[Object, List[Object]]

T = TypeVar("T", bound=Union[Object, GridAndObjects])
State = str

Primitive = Callable[[Object, str, int], Object]
Match = Tuple[State, Callable[[T], Optional[T]]]
Xform = Callable[[List[Example[T]], str, int], Optional[Match[T]]]


@dataclass
class XformEntry(Generic[T]):
    xform: Xform[T]
    difficulty: int
