import os
import argparse
import struct
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image

from ..src.models import build_model, image_model
from ..src.utils.common import str2bool, interpolate_log, create_folder, dump_json
from ..src.utils.stream_helper import get_padding_size, get_state_dict
from ..src.utils.png_reader import PNGReader
from ..src.utils.stream_helper import get_padding_size, get_state_dict

from ..src.models.MLCodec_rans import RansEncoder, RansDecoder
from ..src.utils.timer import Timer
from ..src.tensorrt_support import *

MODELS = {
    "EVC_LL": 'EVC_LL.pth.tar',
    "EVC_ML": 'EVC_ML_MD.pth.tar',
    "EVC_SL": 'EVC_SL_MD.pth.tar',
    "EVC_LM": 'EVC_LM_MD.pth.tar',
    "EVC_LS": 'EVC_LS_MD.pth.tar',
    "EVC_MM": 'EVC_MM_MD.pth.tar',
    "EVC_SS": 'EVC_SS_MD.pth.tar',
    # "Scale_EVC_SL": 'Scale_EVC_SL_MDRRL.pth.tar',
    # "Scale_EVC_SS": 'Scale_EVC_SS_MDRRL.pth.tar',
}

BLOCK_SIZE = 512

class ModelEngine(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()

        # load model
        model_path, compiled_path = self.get_model_path(model_name)

        i_state_dict = get_state_dict(model_path)
        i_frame_net = build_model(model_name, ec_thread=True)
        i_frame_net.load_state_dict(i_state_dict, verbose=False)
        i_frame_net = i_frame_net.cuda()
        i_frame_net.eval()

        i_frame_net.update(force=True)
        self.model_name = model_name
        self.i_frame_net: image_model.EVC = i_frame_net.half()
        self.cuda()
    
    def compile(self, output_dir):
        compile(self.i_frame_net, output_dir)
    
    def compress_block(self, img_block, q_scale):
        q_scale *= 4
        bit_stream = self.i_frame_net.compress(img_block, q_scale)['bit_stream']
        return bit_stream

    def decompress_block(self, bit_stream, h, w, q_scale):
        q_scale *= 4
        recon_img = self.i_frame_net.decompress(bit_stream, h, w, q_scale)['x_hat']
        return recon_img
    
    @classmethod
    def get_model_path(cls, model_name):
        model_path = MODELS[model_name]
        file_folder = os.path.split(__file__)[0]
        model_path = os.path.join(file_folder, "../checkpoints", model_path)
        compiled_path = model_path + ".trt"
        return model_path, compiled_path
    
    @classmethod
    def _load_from_weight(cls, model_name, compiled_path):
        decoder_app = cls(model_name)
        torch.cuda.synchronize()
        dummy_input = torch.zeros([1, 3, BLOCK_SIZE, BLOCK_SIZE], device='cuda', dtype=torch.half)
        decoder_app.i_frame_net(dummy_input, 0.5)
        decoder_app.compile(compiled_path)
        return decoder_app
    
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
    def from_model_name(cls, model_name):
        with Timer("Loading"):
            model_path, compiled_path = cls.get_model_path(model_name)
            if os.path.isdir(compiled_path):
                try:
                    decoder_app = cls._load_from_compiled(model_name, compiled_path)
                except FileNotFoundError:
                    decoder_app = cls._load_from_weight(model_name, compiled_path)
            else:
                decoder_app = cls._load_from_weight(model_name, compiled_path)
        return decoder_app