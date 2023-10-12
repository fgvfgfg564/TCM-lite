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

from ..models.tcm import TCM_vbr

from .utils import *
from ...utils.tensorrt_support import *

BLOCK_SIZE = 512

def get_state_dict(model_path, device):
    dictory = {}
    print("Loading", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    for k, v in checkpoint["state_dict"].items():
        dictory[k.replace("module.", "")] = v

class ModelEngine(nn.Module):
    MODELS = {f"TCM_VBR_{i}": f"vcip_vbr{i}_best.pth.tar" for i in range(3)}

    def __init__(self, model_name) -> None:
        super().__init__()

        # load model
        model_path, compiled_path = self.get_model_path(model_name)

        i_state_dict = get_state_dict(model_path, device='cuda')
        i_frame_net = TCM_vbr(model_name[:6], ec_thread=True)
        i_frame_net.load_state_dict(i_state_dict, verbose=False)
        i_frame_net = i_frame_net.cuda()
        i_frame_net.eval()

        i_frame_net.update(force=True)
        self.model_name = model_name
        self.i_frame_net = i_frame_net.half()
        self.lmbda_min = torch.tensor(0.0025*np.exp(int(model_name[8])), dtype=torch.half, device='cuda')
        self.lmbda_max = torch.tensor(0.0025*np.exp(int(model_name[8]) + 1), dtype=torch.half, device='cuda')
        self.cuda()
    
    def _q_scale_mapping(self, q_scale_0_1):
        # 0 -> self.lmbda_max
        # 1 -> self.lmbda_min

        qs = self.lmbda_max + q_scale_0_1 * (self.lmbda_min - self.lmbda_max)
        return qs
    
    def compile(self, output_dir):
        compile(self.i_frame_net, output_dir)
    
    def compress_block(self, img_block, q_scale):
        lmbda = self._q_scale_mapping(q_scale).to(img_block.device)
        bit_stream_y, bit_stream_z = self.i_frame_net.compress(img_block, lmbda)['strings']
        bit_stream = combine_bytes(bit_stream_y, bit_stream_z)
        return bit_stream

    def decompress_block(self, bit_stream, h, w, _):
        bit_stream_y, bit_stream_z = separate_bytes(bit_stream)
        bit_stream = (bit_stream_y, bit_stream_z)
        recon_img = self.i_frame_net.decompress(bit_stream, [h // 64, w // 64])['x_hat']
        return recon_img
    
    @classmethod
    def get_model_path(cls, model_name):
        model_path = cls.MODELS[model_name]
        file_folder = os.path.split(__file__)[0]
        model_path = os.path.join(file_folder, "../checkpoints", model_path)
        compiled_path = model_path + ".trt"
        return model_path, compiled_path
    
    @classmethod
    def _load_from_weight(cls, model_name, compiled_path, compile=True):
        decoder_app = cls(model_name)
        torch.cuda.synchronize()
        dummy_input = torch.zeros([1, 3, BLOCK_SIZE, BLOCK_SIZE], device='cuda', dtype=torch.half)
        decoder_app.i_frame_net(dummy_input, 0.5)
        if compile:
            decoder_app.compile(compiled_path)
        return decoder_app

    def preheat(self):
        dummy_input = torch.zeros([1, 3, BLOCK_SIZE, BLOCK_SIZE], device='cuda', dtype=torch.half)

        for q_scale in np.linspace(0, 1, 20):
            bitstream = self.compress_block(dummy_input, 0.5)
            _ = self.decompress_block(bitstream, BLOCK_SIZE, BLOCK_SIZE, q_scale)
    
    @classmethod
    def _load_from_compiled(cls, model_name, compiled_path):
        print("Load from compiled model")
        with InitTRTModelWithPlaceholder():
            decoder_app = cls(model_name)
        load_weights(decoder_app.i_frame_net, compiled_path)

        dummy_input = torch.zeros([1, 3, BLOCK_SIZE, BLOCK_SIZE], device='cuda', dtype=torch.half)
        decoder_app.i_frame_net(dummy_input, 0.5)
        return decoder_app

    @classmethod
    def from_model_name(cls, model_name, ignore_tensorrt=False):
        model_path, compiled_path = cls.get_model_path(model_name)
        if not ignore_tensorrt and os.path.isdir(compiled_path):
            try:
                decoder_app = cls._load_from_compiled(model_name, compiled_path)
            except FileNotFoundError:
                decoder_app = cls._load_from_weight(model_name, compiled_path, not ignore_tensorrt)
        else:
            decoder_app = cls._load_from_weight(model_name, compiled_path, not ignore_tensorrt)
        decoder_app.preheat()
        return decoder_app