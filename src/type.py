from typing import TypedDict, Dict, Literal, Union
from typing_extensions import TypeAlias
import numpy as np

from .math_utils import Warpped4DFitter


class CTUCurves(TypedDict):
    b_e: Warpped4DFitter
    b_q: Warpped4DFitter
    b_t: np.ndarray


MethodIdx: TypeAlias = int
CTUIdx: TypeAlias = int
ImgCurves: TypeAlias = Dict[MethodIdx, Dict[CTUIdx, CTUCurves]]

WorkerConfig: TypeAlias = Union[int, Literal["AUTO"]]
