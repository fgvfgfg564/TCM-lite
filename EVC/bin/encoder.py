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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("--target-bpp", type=float, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys())
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--id_bias", type=int, default=0, help="bias of stored model id in bitstream")

    args = parser.parse_args()
    return args

class EncoderApp(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()

        # load model
        model_path = MODELS[model_name]
        i_state_dict = get_state_dict(os.path.join("checkpoints", model_path))
        i_frame_net = build_model(model_name, ec_thread=True)
        i_frame_net.load_state_dict(i_state_dict, verbose=False)
        i_frame_net = i_frame_net.cuda()
        i_frame_net.eval()

        i_frame_net.update(force=True)
        self.model_name = model_name
        self.model_id = get_model_id(model_name)
        self.i_frame_net: image_model.EVC = i_frame_net

    @staticmethod
    def read_img(img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        rgb = Image.open(img_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        rgb = torch.from_numpy(rgb).type(torch.FloatTensor)
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

def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    engine = EncoderApp(args.model)
    engine.compress(args.input, args.target_bpp, args.output, args.id_bias)

if __name__ == "__main__":
    main()