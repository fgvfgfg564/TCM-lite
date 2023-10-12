import sys

sys.path.append("./TCM")
import os
import argparse
import importlib
import time
from itertools import starmap
from collections import defaultdict
import subprocess

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import multiprocessing as mp
from einops import rearrange

CODECS = ["TCM", "VTM", "BPG"]
TRADITIONAL_CODECS = ["VTM", "BPG"]
block_size = 1024


def compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def parse_args(argv):
    parser = argparse.ArgumentParser(description="VCIP2023_challenge testing script.")
    parser.add_argument(dest="codec", type=str, default="bpg", help="Select codec")
    args = parser.parse_args(argv)
    return args


def setup_lc_args(argv):
    parser = argparse.ArgumentParser(description="Learning-based codec setup")
    parser.add_argument("dataset", type=str)
    parser.add_argument(
        "--bpg_path",
        type=str,
        help="Corresponding BPG filepath to calculate target bpp",
    )
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
    parser.add_argument(
        "-j", "--num-jobs", type=int, default=1, help="number of parallel jobs"
    )
    parser.add_argument(
        "-q",
        "--qps",
        dest="qps",
        metavar="Q",
        default=[28],
        nargs="+",
        type=int,
        help="list of quality/quantization parameter",
    )
    args = parser.parse_args(argv)
    return args


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="replicate",
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-int(padding[0]), -int(padding[1]), -int(padding[2]), -int(padding[3])),
    )


def block(x, size):
    x_blocked = rearrange(x, "b c (h p1) (w p2) -> b c h w p1 p2", p1=size, p2=size)
    h = x_blocked.size(2)
    w = x_blocked.size(3)
    x_blocked = rearrange(x_blocked, "b c h w p1 p2 -> (b h w) c p1 p2")
    return x_blocked, h, w


def merge(x, h, w):
    x_merged = rearrange(x, "(b h w) c p1 p2 -> b c h w p1 p2", h=h, w=w)
    x_merged = rearrange(x_merged, "b c h w p1 p2 -> b c (h p1) (w p2)")
    return x_merged


def func(codec, i, *args):
    rv = codec.encode(*args)
    return i, rv


def convert(x):
    if isinstance(x, Image.Image):
        x = np.asarray(x)
    x = torch.from_numpy(x.copy()).float().unsqueeze(0)
    if x.size(3) == 3:
        x = x.permute(0, 3, 1, 2)
    return x


def subprocess_popen(statement):
    p = subprocess.Popen(statement, stdout=subprocess.PIPE)
    while p.poll() is None:
        if p.wait() != 0:
            return False
        else:
            re = p.stdout.readlines()
            result = []
            for i in range(len(re)):
                res = re[i].decode("utf-8").strip("\r\n")
                result.append(res)
            return result


def main(argv):
    args = parse_args(argv[:1])
    BPP = 0.0
    if args.codec not in TRADITIONAL_CODECS:
        lc_args = setup_lc_args(argv[1:])
        subdirs = [x[0] for x in os.walk(lc_args.dataset)]
        count = 0
        for dir in subdirs:
            if dir == lc_args.dataset:
                out_dir = lc_args.code
            else:
                _, directory = dir.rsplit("/", 1)
                out_dir = lc_args.code + "/" + directory
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            for file in os.listdir(dir):
                if file[-3:] != "png":
                    continue
                input_image = dir + "/" + file
                out_filepath = out_dir + "/" + file[:-4] + ".bin"
                bpg_filepath = lc_args.bpg_path + "/" + file[:-4] + ".bin"
                img = Image.open(input_image).convert("RGB")
                x = transforms.ToTensor()(img)
                target_bpp = os.path.getsize(bpg_filepath) * 8 / (x.size(1) * x.size(2))
                statement = (
                    "python TCM/bin/encoder.py "
                    + input_image
                    + " --target_bpp "
                    + target_bpp
                    + " -o "
                    + out_filepath
                )
                result = subprocess_popen(statement)
                if result:
                    count += 1
                    BPP += os.path.getsize(out_filepath) * 8 / (x.size(1) * x.size(2))
        BPP /= count
    else:
        codecImport = importlib.import_module("models.codecs")
        codecClass = getattr(codecImport, args.codec)
        tc_args = setup_tc_args(argv[1:])
        codec = codecClass(tc_args)
        subdirs = [x[0] for x in os.walk(tc_args.dataset)]
        file_num = []
        out_list = []
        for dir in subdirs:
            img_list = []
            for file in os.listdir(dir):
                if file[-3:] == "png":
                    img_list.append(dir + "/" + file)
            if len(img_list) == 0:
                continue
            file_num.append(len(img_list))
            if dir == tc_args.dataset:
                out_dir = tc_args.code
            else:
                _, directory = dir.rsplit("/", 1)
                out_dir = tc_args.code + "/" + directory
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            nargs = [
                (codec, i, f, out_dir + "/" + f.rsplit("/", 1)[1][:-4] + ".bin", q)
                for i, q in enumerate(tc_args.qps)
                for f in img_list
            ]
            num_jobs = tc_args.num_jobs
            pool = mp.Pool(num_jobs) if num_jobs > 1 else None
            if pool:
                rv = pool.starmap(func, nargs)
            else:
                rv = list(starmap(func, nargs))
            results = [defaultdict(float) for _ in range(len(tc_args.qps))]
            for i, metrics in rv:
                for k, v in metrics.items():
                    results[i][k] += v
            for i, _ in enumerate(results):
                for k, v in results[i].items():
                    results[i][k] = v / len(img_list)
            out = defaultdict(list)
            for r in results:
                for k, v in r.items():
                    out[k].append(v)
            out_list.append(out)
        for i in range(len(file_num)):
            BPP += out_list[i]["bpp"][0] * file_num[i]
        BPP /= sum(file_num)
    print(f"average_Bit-rate: {BPP:.3f} bpp")


if __name__ == "__main__":
    main(sys.argv[1:])
