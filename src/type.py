from typing import TypedDict, Dict, Literal, Union
from typing_extensions import TypeAlias
import numpy as np

from .math_utils import WarppedInterpolator


class CTUCurves(TypedDict):
    b_e: WarppedInterpolator
    b_q: WarppedInterpolator
    b_t: np.array


MethodIdx: TypeAlias = int
CTUIdx: TypeAlias = int
ImgCurves: TypeAlias = Dict[MethodIdx, Dict[CTUIdx, CTUCurves]]

WorkerConfig: TypeAlias = Union[int, Literal["AUTO"]]
