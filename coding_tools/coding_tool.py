import os
import argparse
import struct
import time
import tempfile
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image

from .utils.timer import Timer
from .utils.pad import pad_torch_img, get_padded_hw

import abc
import struct


class CodingToolBase(abc.ABC):
    @abc.abstractproperty
    def MODELS(self):
        pass

    def _q_scale_mapping(self, q_scale_0_1):
        # 0 -> self.q_scale_min
        # 1 -> self.q_scale_max

        qs = self.q_scale_min + q_scale_0_1 * (self.q_scale_max - self.q_scale_min)
        return qs

    @abc.abstractmethod
    def compress_block(self, img_block, q_scale) -> bytes:
        """
        Encode a image block with given q_scale. The block is not padded.
        """

    @abc.abstractmethod
    def decompress_block(self, bit_stream, h, w):
        """
        Decode a image block. The block is not padded. H and W are passed from image header.
        """

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
            _ = self.decompress_block(bitstream, self.ctu_size, self.ctu_size)

    @classmethod
    def from_model_name(cls, model_name, dtype, ctu_size):
        with Timer("Loading"):
            decoder_app = cls._load_from_weight(model_name, dtype, ctu_size)
            decoder_app.cuda()
            decoder_app.preheat()
        return decoder_app

    @abc.abstractproperty
    def PLATFORM(self):
        """
        Defines the platform of this coding tool. Must be one of 'numpy', 'torch'.
        """


class TraditionalCodingToolBase(CodingToolBase):
    PLATFORM = "numpy"
    MODELS = None


class LICToolBase(CodingToolBase, nn.Module):
    PLATFORM = "torch"

    def __init__(self, model_name, dtype, ctu_size) -> None:
        super().__init__()
        self.dtype = dtype
        self.model_name = model_name
        self.ctu_size = ctu_size

    @abc.abstractproperty
    def MIN_CTU_SIZE(self) -> int:
        """
        Minimal CTU size for this LIC. Usually 128 or 256.
        """

    @staticmethod
    def _pack_qscale(q_scale):
        return struct.pack("f", q_scale)

    @staticmethod
    def _unpack_qscale(buffer):
        return struct.unpack("f", buffer)[0]

    @property
    def _ctu_header_size(self):
        return struct.calcsize("f")

    @abc.abstractmethod
    def lic_compress(self, padded_block, q_scale):
        """
        compress with LIC network. The image block is padded.
        """

    @abc.abstractmethod
    def lic_decompress(self, bit_stream, padded_h, padded_w, q_scale):
        """
        Decompress with LIC network.
        """

    def compress_block(self, img_block, q_scale):
        """
        Pad and call lic_compress
        """
        padded_block, _ = pad_torch_img(img_block, self.MIN_CTU_SIZE)
        return self._pack_qscale(q_scale) + self.lic_compress(padded_block, q_scale)

    def decompress_block(self, bit_stream, h, w):
        padded_h, padded_w = get_padded_hw(h, w, self.MIN_CTU_SIZE)
        q_scale = self._unpack_qscale(bit_stream[: self._ctu_header_size])
        bit_stream = bit_stream[self._ctu_header_size :]
        padded_block = self.lic_decompress(bit_stream, padded_h, padded_w, q_scale)
        return padded_block[:, :, :h, :w]
