import sys

sys.path.append("./TCM")
import os
import argparse
import importlib
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from encode import compute_psnr, convert, subprocess_popen

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
    parser.add_argument("-o", type=str, help="Path to store reconstructed image")
    parser.add_argument("--code", type=str, help="Path to store bitStream")
    args = parser.parse_args(argv)
    return args


def setup_tc_args(argv):
    parser = argparse.ArgumentParser(description="Traditional codec setup")
    parser.add_argument("dataset", type=str)
    # bpg
    parser.add_argument("-m", choices=["420", "444"], default="444")
    parser.add_argument("-b", choices=["8", "10"], default="8")
    parser.add_argument("-c", choices=["rgb", "ycbcr"], default="ycbcr")
    parser.add_argument("-e", choices=["jctvc", "x265"], default="x265")
    parser.add_argument("--encoder-path", default="bpgenc", help="BPG encoder path")
    parser.add_argument("--decoder-path", default="bpgdec", help="BPG decoder path")
    # vtm
    parser.add_argument("--build-dir", type=str, help="VTM build dir")
    parser.add_argument("--config", type=str, help="VTM config file")
    parser.add_argument(
        "--rgb", action="store_true", help="Use RGB color space (over YCbCr)"
    )

    parser.add_argument("--code", type=str, help="Path to store bitStream")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv[:1])
    PSNR = 0.0
    BPP = 0.0
    if args.codec not in TRADITIONAL_CODECS:
        lc_args = setup_lc_args(argv[1:])
        subdirs = [x[0] for x in os.walk(lc_args.code)]
        count = 0
        for dir in subdirs:
            if dir == lc_args.code:
                in_dir = lc_args.dataset
                out_dir = lc_args.o
            else:
                _, directory = dir.rsplit("/", 1)
                in_dir = lc_args.dataset + "/" + directory
                out_dir = lc_args.o + "/" + directory
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            for file in os.listdir(dir):
                if file[-3:] != "bin":
                    continue
                out_filepath = dir + "/" + file
                in_filepath = in_dir + "/" + file[:-4] + ".png"
                rec_filepath = out_dir + "/" + file[:-4] + ".png"
                img = Image.open(in_filepath).convert("RGB")
                x = transforms.ToTensor()(img)
                statement = (
                    "python TCM/bin/decoder.py " + out_filepath + " -o " + rec_filepath
                )
                result = subprocess_popen(statement)
                if result:
                    count += 1
                    rec_img = Image.open(rec_filepath).convert("RGB")
                    PSNR += compute_psnr(convert(img), convert(rec_img))
                    BPP += os.path.getsize(out_filepath) * 8 / (x.size(1) * x.size(2))
        PSNR /= count
        BPP /= count
    else:
        codecImport = importlib.import_module("models.codecs")
        codecClass = getattr(codecImport, args.codec)
        tc_args = setup_tc_args(argv[1:])
        codec = codecClass(tc_args)
        subdirs = [x[0] for x in os.walk(tc_args.code)]
        count = 0
        for dir in subdirs:
            if dir == tc_args.code:
                in_dir = tc_args.dataset
            else:
                _, directory = dir.rsplit("/", 1)
                in_dir = tc_args.dataset + "/" + directory
            for file in os.listdir(dir):
                if file[-3:] != "bin":
                    continue
                out_filepath = dir + "/" + file
                in_filepath = in_dir + "/" + file[:-4] + ".png"
                img = Image.open(in_filepath).convert("RGB")
                x = transforms.ToTensor()(img).unsqueeze(0)
                rec = codec.decode(out_filepath, in_filepath)
                count += 1
                PSNR += compute_psnr(convert(img), convert(rec))
                BPP += os.path.getsize(out_filepath) * 8 / (x.size(2) * x.size(3))
        PSNR /= count
        BPP /= count
    print(f"average_PSNR: {PSNR:.2f}dB")
    print(f"average_Bit-rate: {BPP:.3f} bpp")


if __name__ == "__main__":
    main(sys.argv[1:])
