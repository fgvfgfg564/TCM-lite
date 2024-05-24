import os
import argparse
import struct
import time
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from PIL import Image

from ..models.tcm import TCM_vbr, TCM_vbr2

from .utils import *

from ...coding_tool import LICToolBase
from ...register import register_tool


def get_state_dict(model_path, device):
    dictory = {}
    print("Loading", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    for k, v in checkpoint["state_dict"].items():
        dictory[k.replace("module.", "")] = v
    return dictory


@register_tool("TCM")
class ModelEngine(LICToolBase):
    MODELS = {f"TCM_VBR2_{i}": f"vcip_vbr2_{i}_best.pth.tar" for i in [2]}
    MODELS.update({"TCM_VBR2_ALL": "vbr2_all_best.pth.tar"})
    MIN_CTU_SIZE = 256

    def __init__(self, model_name, dtype, ctu_size) -> None:
        super().__init__(model_name, dtype, ctu_size)

        # load model
        model_path = self.get_model_path(model_name)

        i_state_dict = get_state_dict(model_path, device="cuda")
        i_frame_net = TCM_vbr2()
        i_frame_net.load_state_dict(i_state_dict, verbose=False)
        i_frame_net = i_frame_net.cuda()
        i_frame_net.eval()

        i_frame_net.update(force=True)
        self.i_frame_net = i_frame_net.to(dtype)

        self.q_scales = i_state_dict["q_scale"].to(dtype)
        self.q_scale_min = self.q_scales[-1]
        self.q_scale_max = self.q_scales[0]
        self.cuda()

    def lic_compress(self, img_block, q_scale):
        q_scale = self._q_scale_mapping(q_scale).to(img_block.device)
        bit_stream_y, bit_stream_z = self.i_frame_net.compress(
            img_block, q_scale.to(self.dtype)
        )["strings"]
        bit_stream = combine_bytes(bit_stream_y[0], bit_stream_z[0])
        return bit_stream

    def lic_decompress(self, bit_stream, h, w, q_scale):
        q_scale = self._q_scale_mapping(q_scale).cuda()
        bit_stream_y, bit_stream_z = separate_bytes(bit_stream)
        bit_stream = ([bit_stream_y], [bit_stream_z])
        recon_img = self.i_frame_net.decompress(
            bit_stream, [h // 64, w // 64], q_scale
        )["x_hat"]
        return recon_img

    @classmethod
    def get_model_path(cls, model_name):
        model_path = cls.MODELS[model_name]
        file_folder = os.path.split(__file__)[0]
        model_path = os.path.join(file_folder, "../checkpoints", model_path)
        return model_path
