import sys
import os
import argparse
import importlib
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from encode_decode import compute_psnr, convert, crop, merge

CODECS = ["TCM", "VTM", "BPG"]
TRADITIONAL_CODECS = ["VTM", "BPG"]

def parse_args(argv):
    parser = argparse.ArgumentParser(description="VCIP2023_challenge testing script.")
    parser.add_argument(dest="codec", type=str, default="bpg", help="Select codec")
    args = parser.parse_args(argv)
    return args

def setup_lc_args(argv):
    parser = argparse.ArgumentParser(description="Learning-based codec setup")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    # tcm
    parser.add_argument("-c", "--config", default=[2, 2, 2, 2, 2, 2], type=list, help="TCM config")
    parser.add_argument("-h", "--head_dim", default=[8, 16, 32, 32, 16, 8], type=list, help="TCM head_dim")
    parser.add_argument("-d", "--drop_path_rate", default=0., type=float, help="TCM drop_path_rate")
    parser.add_argument("-N", default=128, type=int, help="TCM channel number N")
    parser.add_argument("-M", default=320, type=int, help="TCM channel number M")
    parser.add_argument("-n", "--num_slices", default=5, type=int, help="TCM num_slices")
    parser.add_argument("-m", "--max_support_slices", default=5, type=int, help="TCM max_support_slices")

    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--code", type=str, help="Path to store bitStream")
    args = parser.parse_args(argv)
    return args

def setup_tc_args(argv):
    parser = argparse.ArgumentParser(description="Traditional codec setup")
    parser.add_argument("dataset", type=str)
    # bpg
    parser.add_argument("-m", choices=["420", "444"], default="8")
    parser.add_argument("-b", choices=["8", "10"], default="8")
    parser.add_argument("-c", choices=["rgb", "ycbcr"], default="ycbcr")
    parser.add_argument("-e", choices=["jctvc", "x265"], default="x265")
    parser.add_argument("--encoder-path", default="bpgenc", help="BPG encoder path")
    parser.add_argument("--decoder-path", default="bpgdec", help="BPG decoder path")
    # vtm
    parser.add_argument("--build-dir", type=str, help="VTM build dir")
    parser.add_argument("--config", type=str, help="VTM config file")
    parser.add_argument("--rgb", action="store_true", help="Use RGB color space (over YCbCr)")

    parser.add_argument("--code", type=str, help="Path to store bitStream")
    parser.add_argument("-j", "--num-jobs", type=int, default=1, help="number of parallel jobs")
    parser.add_argument("-q", "--qps", dest="qps", metavar="Q", default=[28], nargs="+", type=int, help="list of quality/quantization parameter")
    parser.add_argument("--metrics", dest="metrics", default=["psnr"], nargs="+")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv[:1])
    PSNR = 0.
    BPP = 0.
    dec_time = 0.
    if args.codec not in TRADITIONAL_CODECS:
        modelImport = importlib.import_module("models.tcm")
        modelClass = getattr(modelImport, args.codec)
        lc_args = setup_lc_args(argv[1:])
        if lc_args.cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'
        model = modelClass(config=lc_args.c, head_dim=lc_args.h, drop_path_rate=lc_args.d, N=lc_args.N, M=lc_args.M,
                           num_slices=lc_args.n, max_support_slices=lc_args.m).to(device)
        dictory = {}
        if lc_args.checkpoint:
            checkpoint = torch.load(lc_args.checkpoint, map_location=device)
            for k, v in checkpoint["state_dict"].items():
                dictory[k.replace("module.", "")] = v
            model.load_state_dict(dictory)
        model.update()
        subdirs = [x[0] for x in os.walk(lc_args.code)]
        count = 0
        with torch.no_grad():
            if lc_args.cuda:
                torch.cuda.synchronize()
            for dir in subdirs[1:]:
                _, directory = dir.rsplit('/', 1)
                in_dir = lc_args.dataset + '/' + directory
                for file in os.listdir(dir):
                    out_filepath = dir + '/' + file
                    in_filepath = in_dir + '/' + file[:-4] + '.png'
                    img = Image.open(in_filepath).convert('RGB')
                    x = transforms.ToTensor()(img).unsqueeze(0)
                    start_time = time.time()
                    with open(out_filepath, 'rb') as in_file:
                        pl = np.frombuffer(in_file.read(8), dtype=np.uint16)
                        size, hn, wn = np.frombuffer(in_file.read(6), dtype=np.uint16)
                        out_list = []
                        for idx in range(hn * wn):
                            length_y, length_z = np.frombuffer(in_file.read(8), dtype=np.uint32)
                            string_y = in_file.read(length_y)
                            string_z = in_file.read(length_z)
                            out_dec = model.decompress([[string_y], [string_z]], [size // 64, size // 64])
                            out_list.append(out_dec["x_hat"].cpu())
                        out_blocked = torch.cat(out_list, dim=0)
                        x_hat = merge(out_blocked, hn, wn)
                    x_hat = crop(x_hat, pl)
                    rec = transforms.ToPILImage()(x_hat.squeeze(0))
                    dec_time += time.time() - start_time
                    count += 1
                    PSNR += compute_psnr(convert(img), convert(rec))
                    BPP += os.path.getsize(out_filepath) * 8 / (x.size(2) * x.size(3))
        PSNR /= count
        BPP /= count
        dec_time /= count
    else:
        codecImport = importlib.import_module("models.codecs")
        codecClass = getattr(codecImport, args.codec)
        tc_args = setup_tc_args(argv[1:])
        codec = codecClass(tc_args)
        subdirs = [x[0] for x in os.walk(tc_args.code)]
        count = 0
        for dir in subdirs:
            _, directory = dir.rsplit('/', 1)
            in_dir = tc_args.dataset + '/' + directory
            for file in os.listdir(dir):
                out_filepath = dir + '/' + file
                in_filepath = in_dir + '/' + file[:-4] + '.png'
                img = Image.open(in_filepath).convert('RGB')
                x = transforms.ToTensor()(img).unsqueeze(0)
                start_time = time.time()
                rec = codec.decode(out_filepath, in_filepath)
                dec_time += time.time() - start_time
                count += 1
                PSNR += compute_psnr(convert(img), convert(rec))
                BPP += os.path.getsize(out_filepath) * 8 / (x.size(2) * x.size(3))
        PSNR /= count
        BPP /= count
        dec_time /= count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_Bit-rate: {BPP:.3f} bpp')
    print(f'average_time: {dec_time:.3f} s')

if __name__ == "__main__":
    main(sys.argv[1:])