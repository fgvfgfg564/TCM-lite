import os
import argparse
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image

from src.models import build_model, image_model
from src.utils.common import str2bool, interpolate_log, create_folder, dump_json
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.png_reader import PNGReader
from src.utils.stream_helper import get_padding_size, get_state_dict

from src.models.MLCodec_rans import RansEncoder, RansDecoder
from src.utils.timer import Timer
from src.tensorrt_support import *

MODELS = {
    "EVC_LL": 'EVC_LL.pth.tar',
    "EVC_ML": 'EVC_ML_MD.pth.tar',
    "EVC_SL": 'EVC_SL_MD.pth.tar',
    "EVC_LM": 'EVC_LM_MD.pth.tar',
    "EVC_LS": 'EVC_LS_MD.pth.tar',
    "EVC_MM": 'EVC_MM_MD.pth.tar',
    "EVC_SS": 'EVC_SS_MD.pth.tar',
    "Scale_EVC_SL": 'Scale_EVC_SL_MDRRL.pth.tar',
    "Scale_EVC_SS": 'Scale_EVC_SS_MDRRL.pth.tar',
}

def get_model_id(model_name):
    for i, name in enumerate(MODELS.keys()):
        if model_name == name:
            return i
    raise ValueError(f"{model_name} is not a valid model name.")

def get_model_name(model_id):
    for i, name in enumerate(MODELS.keys()):
        if i == model_id:
            return name
    raise ValueError(f"{i}: model_id is invalid.")

def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)

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
        self.model_id = get_model_id(model_name)
        self.i_frame_net: image_model.EVC = i_frame_net.half()
        self.cuda()
    
    def compile(self, output_dir):
        compile(self.i_frame_net, output_dir)

    @staticmethod
    def read_img(img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        rgb = Image.open(img_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        rgb = torch.from_numpy(rgb).type(torch.half)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.cuda()
        return rgb
    
    @staticmethod
    def pad_img(x):
        pic_height = x.shape[2]
        pic_width = x.shape[3]
        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
        x_padded = F.pad(
            x,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )
        return pic_height, pic_width, x_padded
    
    def compress(self, input_pth, target_bpp, output_pth, id_bias):
        x = self.read_img(input_pth)
        h, w, x_padded = self.pad_img(x)

        # ignore target_bpp and patchification first
        q_scale = 0.5
        compress_results = self.i_frame_net.compress(x_padded, q_scale)
        bit_stream = compress_results['bit_stream']

        # Create model header
        model_id = self.model_id + id_bias
        header = struct.pack("B2Hf", model_id, h, w, float(q_scale))

        file_obj = open(output_pth, "wb")
        file_obj.write(header)
        file_obj.write(bit_stream)
        file_obj.close()
    
    def decompress(self, input_pth, output_pth, id_bias):
        with Timer("Decompress."):
            file_obj = open(input_pth, "rb")
            headersize = struct.calcsize('B2Hf')
            header_bits = file_obj.read(headersize)
            model_id, h, w, q_scale = struct.unpack('B2Hf', header_bits)
            model_id -= id_bias
            model_name = get_model_name(model_id)
            assert(model_id == self.model_id)

            main_bits = file_obj.read()
            file_obj.close()

            padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w)
            padded_h = h + padding_t + padding_b
            padded_w = w + padding_l + padding_r

            with Timer("Decompress network."):
                recon_img = self.i_frame_net.decompress(main_bits, padded_h, padded_w, q_scale)['x_hat']
            recon_img = F.pad(recon_img, (-padding_l, -padding_r, -padding_t, -padding_b))
            save_torch_image(recon_img, output_pth)
    
    @classmethod
    def get_model_id(cls, input_pth, id_bias):
        file_obj = open(input_pth, "rb")
        headersize = struct.calcsize('B2Hf')
        header_bits = file_obj.read(headersize)
        model_id, h, w, q_scale = struct.unpack('B2Hf', header_bits)
        model_id -= id_bias
        file_obj.close()
        return model_id
    
    @classmethod
    def get_model_path(cls, model_name):
        model_path = MODELS[model_name]
        model_path = os.path.join("checkpoints", model_path)
        compiled_path = model_path + ".trt"
        return model_path, compiled_path
    
    @classmethod
    def from_model_name(cls, model_name):
        with Timer("Loading"):
            model_path, compiled_path = cls.get_model_path(model_name)
            if os.path.isdir(compiled_path):
                print("Load from compiled model")
                with InitTRTModelWithPlaceholder():
                    decoder_app = cls(model_name)
                load_weights(decoder_app.i_frame_net, compiled_path)

                dummy_input = torch.zeros([1, 3, BLOCK_SIZE, BLOCK_SIZE], device='cuda', dtype=torch.half)
                decoder_app.i_frame_net(dummy_input, 0.5)
            else:
                decoder_app = cls(model_name)
                torch.cuda.synchronize()
                dummy_input = torch.zeros([1, 3, BLOCK_SIZE, BLOCK_SIZE], device='cuda', dtype=torch.half)
                decoder_app.i_frame_net(dummy_input, 0.5)
                decoder_app.compile(compiled_path)
        return decoder_app
    
    @classmethod
    def from_model_id(cls, model_id):
        model_name = get_model_name(model_id)
        return cls.from_model_name(model_name)
