from .register import register_tool
from .coding_tool import LICToolBase

from lvae import get_model
from lvae.models.qarv.model import VariableRateLossyVAE

import numpy as np
import torch


@register_tool("QARV")
class QARV(LICToolBase):

    MODELS = {"QARV": None}
    MIN_CTU_SIZE = 256

    def __init__(self, model_name, dtype, ctu_size) -> None:
        super().__init__(model_name, dtype, ctu_size)
        self.i_frame_net: VariableRateLossyVAE = get_model("qarv_base", pretrained=True)
        self.i_frame_net.eval()
        self.i_frame_net.compress_mode(True)
        self.i_frame_net.to(dtype)

        self.lmb_min, self.lmb_max = self.i_frame_net.lmb_range

    def _q_scale_mapping(self, q_scale_0_1):
        # Using exponential function; map q_scale to [lmb_min, lmb_max]
        # 0 -> self.lmb_max
        # 1 -> self.lmb_min

        # Scale to a safer range

        lg_lmb_min = np.log(self.lmb_min * 1.25)
        lg_lmb_max = np.log(self.lmb_max * 0.75)

        lg_lmb = (1 - q_scale_0_1) * (lg_lmb_max - lg_lmb_min) + lg_lmb_min

        lmb = np.exp([lg_lmb])
        lmb = torch.tensor(lmb)

        return lmb

    def lic_compress(self, padded_block, q_scale):
        lmb = self._q_scale_mapping(q_scale)
        return self.i_frame_net.compress(padded_block, lmb)

    def lic_decompress(self, bit_stream, padded_h, padded_w, q_scale):
        return self.i_frame_net.decompress(bit_stream)
