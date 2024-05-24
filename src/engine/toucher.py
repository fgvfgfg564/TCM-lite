import numpy as np

from ..math_utils import binary_search
from coding_tools.traditional_tools import TraditionalCodingToolBase


class Toucher:
    def __init__(self, tool: TraditionalCodingToolBase) -> None:
        self.tool = tool

    def touch_complexity(self, img_block: np.ndarray, target_bpp: float = 1.0) -> float:
        h, w, c = img_block.shape
        target_bytes = target_bpp * h * w / 8

        def _f(q_scale):
            return -len(self.tool.compress_block(img_block, q_scale))

        # q_scale越高则码率越低，质量越差；需要求个负数。
        resulting_q_scale = binary_search(_f, -target_bytes, 0.0, 1.0, 1e-3)
        recon: np.ndarray = self.tool.decompress_block(
            self.tool.compress_block(img_block, resulting_q_scale),
            None,
            None,
        )
        recon = recon.astype(np.float32)
        img_block = img_block.astype(np.float32)
        mse = ((recon - img_block) ** 2).mean()
        return mse
