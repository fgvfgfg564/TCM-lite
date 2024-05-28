from typing import Any, TypedDict, Dict, Literal, Union, Tuple, NamedTuple
from typing_extensions import TypeAlias, TypeVar, Self
import numpy as np
from numpy.typing import NDArray
import dataclasses

from .math_utils import Fitter


class CTUCurves(TypedDict):
    b_e: Fitter
    b_t: np.ndarray


MethodIdx: TypeAlias = int
CTUIdx: TypeAlias = int
ImgCurves: TypeAlias = Dict[MethodIdx, Dict[CTUIdx, CTUCurves]]

WorkerConfig: TypeAlias = Union[int, Literal["AUTO"]]


class LossType(NamedTuple):
    r: float
    t: float
    d: float

    def __repr__(self) -> str:
        return f"<RDT Loss: R={self.r:.6f}; T={self.t:.6f}; D={self.d:.6f}>"

    def __add__(self, other):
        return tuple([(x + y) for x, y in zip(self, other)])

    def __sub__(self, other: Self):
        return tuple([(x - y) for x, y in zip(self, other)])


class SAConfig:
    def __init__(self, level: int) -> None:
        self.ada_init = False
        self.sa_init = False
        self.sa_body = False
        self.inertia = False
        self.swap_op = False
        self.distinct = False

        if level >= 1:
            self.ada_init = True
        if level >= 2:
            self.sa_init = True
        if level >= 3:
            self.sa_body = True
        if level >= 4:
            self.inertia = True
        if level >= 5:
            self.swap_op = True
        if level >= 6:
            self.distinct = True

    def __repr__(self) -> str:
        return f"<SAConfig {self.__dict__}>"
