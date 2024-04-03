from typing import TypedDict, Dict, Literal, Union, Tuple
from typing_extensions import TypeAlias
import numpy as np
import dataclasses

from .math_utils import Fitter


class CTUCurves(TypedDict):
    b_e: Fitter
    b_t: np.ndarray


MethodIdx: TypeAlias = int
CTUIdx: TypeAlias = int
ImgCurves: TypeAlias = Dict[MethodIdx, Dict[CTUIdx, CTUCurves]]

WorkerConfig: TypeAlias = Union[int, Literal["AUTO"]]


class ArithTuple(tuple):

    def __add__(self, other):
        return tuple([(x + y) for x, y in zip(self, other)])

    def __sub__(self, other):
        return tuple([(x - y) for x, y in zip(self, other)])

    def __repr__(self) -> str:
        return f"<RDT Loss: R={self[0]:.6f}; T={self[1]:.6f}; D={self[2]:.6f}>"


LossType: TypeAlias = ArithTuple[float, float, float]
