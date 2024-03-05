import os
import argparse
import struct
import time
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image

from .utils.timer import Timer

class CodingToolBase(nn.Module):
    MODELS = None

    def __init__(self, model_name, dtype, ctu_size) -> None:
        super().__init__()
        self.dtype = dtype
        self.model_name = model_name
        self.ctu_size = ctu_size

    def _q_scale_mapping(self, q_scale_0_1):
        # 0 -> self.q_scale_min
        # 1 -> self.q_scale_max

        qs = self.q_scale_min + q_scale_0_1 * (self.q_scale_max - self.q_scale_min)
        return qs

    def compress_block(self, img_block, q_scale):
        raise NotImplemented

    def decompress_block(self, bit_stream, h, w, q_scale):
        raise NotImplemented

    @classmethod
    def _load_from_weight(cls, model_name, dtype, ctu_size):
        decoder_app = cls(model_name, dtype, ctu_size)
        torch.cuda.synchronize()
        dummy_input = torch.zeros(
            [1, 3, ctu_size, ctu_size], device="cuda", dtype=dtype
        )
        decoder_app.i_frame_net(dummy_input, 0.5)
        return decoder_app

    def preheat(self):
        dummy_input = torch.zeros(
            [1, 3, self.ctu_size, self.ctu_size], device="cuda", dtype=self.dtype
        )

        for q_scale in np.linspace(0, 1, 20):
            bitstream = self.compress_block(dummy_input, 0.5)
            _ = self.decompress_block(bitstream, self.ctu_size, self.ctu_size, q_scale)

    @classmethod
    def from_model_name(cls, model_name, dtype, ctu_size):
        with Timer("Loading"):
            decoder_app = cls._load_from_weight(
                model_name, dtype, ctu_size)
            decoder_app.cuda()
            decoder_app.preheat()
        return decoder_app

    @property
    def PLATFORM(self):
        raise ValueError("PLATFORM not defined. Should be one of ['numpy', 'torch'].")