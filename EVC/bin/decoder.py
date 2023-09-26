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

torch.jit.optimized_execution(False)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--id_bias", type=int, default=0, help="bias of stored model id in bitstream")

    args = parser.parse_args()
    return args

class DecoderApp(nn.Module):
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
        # Save model to disk
        self.i_frame_net.entropy_coder.encoder = None
        self.i_frame_net.entropy_coder.decoder = None
        torch.save(self, compiled_path)
        self.i_frame_net.entropy_coder.encoder = RansEncoder(True, 2)
        self.i_frame_net.entropy_coder.decoder = RansDecoder(2)
        for name, child in self.i_frame_net.named_children():
            print(name)
            print(child)
    
    @classmethod
    def get_model_path(cls, model_name):
        model_path = MODELS[model_name]
        model_path = os.path.join("checkpoints", model_path)
        compiled_path = model_path + ".compiled"
        return model_path, compiled_path
    
    @classmethod
    def from_model_name(cls, model_name):
        with Timer("Loading"):
            model_path, compiled_path = cls.get_model_path(model_name)
            if os.path.isfile(compiled_path):
                with Timer("loading from compiled model."):
                    decoder_app = torch.load(compiled_path)
                    decoder_app.i_frame_net.entropy_coder.encoder = RansEncoder(True, 2)
                    decoder_app.i_frame_net.entropy_coder.decoder = RansDecoder(2)
                    decoder_app = decoder_app.cuda()
                    torch.cuda.synchronize()
            else:
                with Timer("loading weights from file."):
                    decoder_app = cls(model_name)
                    torch.cuda.synchronize()
            dummy_input = torch.zeros([1, 3, 512, 512], device='cuda', dtype=torch.half)
            decoder_app.i_frame_net(dummy_input, 0.5)
            return decoder_app
    
    def decompress_bits(self, bits, height, width, q_scale):
        with Timer("Decompress network."):
            recon_img = self.i_frame_net.decompress(bits, height, width, q_scale)['x_hat']
        return recon_img
    
    @classmethod
    def decompress(cls, input_pth, output_pth, id_bias):
        with Timer("Decompress."):
            file_obj = open(input_pth, "rb")
            headersize = struct.calcsize('B2Hf')
            header_bits = file_obj.read(headersize)
            model_id, h, w, q_scale = struct.unpack('B2Hf', header_bits)
            model_id -= id_bias
            model_name = get_model_name(model_id)

            main_bits = file_obj.read()
            file_obj.close()

            padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w)
            padded_h = h + padding_t + padding_b
            padded_w = w + padding_l + padding_r

            decoder_app = cls.from_model_name(model_name)
            recon_img = decoder_app.decompress_bits(main_bits, padded_h, padded_w, q_scale)
            recon_img = F.pad(recon_img, (-padding_l, -padding_r, -padding_t, -padding_b))
            save_torch_image(recon_img, output_pth)


def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    DecoderApp.decompress(args.input, args.output, args.id_bias)

if __name__ == "__main__":
    with torch.no_grad():
        main()