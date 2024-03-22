from typing import TypedDict, Dict
from typing_extensions import TypeAlias

from .math import *
import numpy as np


class CTUCurves(TypedDict):
    b_e: WarppedPchipInterpolator
    b_q: WarppedPchipInterpolator
    b_t: np.array


ImgCurves: TypeAlias = Dict[int, Dict[int, CTUCurves]]
