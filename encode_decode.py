import sys
import os
import argparse
import importlib
import time
from collections import defaultdict

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
    # parser.add_argument("-p", "--params", dest="params", type=tuple)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    # tcm
    parser.add_argument("-c", default=[2, 2, 2, 2, 2, 2], type=list, help="TCM config")
    # parser.add_argument("-h", default=[8, 16, 32, 32, 16, 8], type=list, help="TCM head_dim")
    parser.add_argument("-d", default=0., type=float, help="TCM drop_path_rate")
    parser.add_argument("-N", default=128, type=int, help="TCM channel number N")
    parser.add_argument("-M", default=320, type=int, help="TCM channel number M")
    parser.add_argument("-n", default=5, type=int, help="TCM num_slices")
    parser.add_argument("-m", default=5, type=int, help="TCM max_support_slices")

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
    x_blocked = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=size, p2=size)
    h = x_blocked.size(2)
    w = x_blocked.size(3)
    x_blocked = rearrange(x_blocked, 'b c h w p1 p2 -> (b h w) c p1 p2')
    return x_blocked, h, w

def merge(x, h, w):
    x_merged = rearrange(x, '(b h w) c p1 p2 -> b c h w p1 p2', h=h, w=w)
    x_merged = rearrange(x_merged, 'b c h w p1 p2 -> b c (h p1) (w p2)')
    return x_merged

def func(codec, i, *args):
    rv = codec.run(*args)
    return i, rv

def convert(x):
    if isinstance(x, Image.Image):
        x = np.asarray(x)
    x = torch.from_numpy(x.copy()).float().unsqueeze(0)
    if x.size(3) == 3:
        x = x.permute(0, 3, 1, 2)
    return x

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
        model = modelClass(config=lc_args.c, head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=lc_args.d, N=lc_args.N, M=lc_args.M, num_slices=lc_args.n, max_support_slices=lc_args.m).to(device)
        dictory = {}
        if lc_args.checkpoint:
            checkpoint = torch.load(lc_args.checkpoint, map_location=device)
            for k, v in checkpoint["state_dict"].items():
                dictory[k.replace("module.", "")] = v
            model.load_state_dict(dictory)
        model.update()
        subdirs = [x[0] for x in os.walk(lc_args.dataset)]
        count = 0
        with torch.no_grad():
            if lc_args.cuda:
                torch.cuda.synchronize()
            for dir in subdirs[1:]:
                _, directory = dir.rsplit('/', 1)
                out_dir = lc_args.code + '/' + directory
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                for file in os.listdir(dir):
                    img = Image.open(dir + '/' + file).convert('RGB')
                    x = transforms.ToTensor()(img)
                    x = x.unsqueeze(0)
                    x_padded, padding = pad(x, block_size)
                    x_blocked, h, w = block(x_padded, block_size)
                    out_filepath = out_dir + '/' + file[:-4] + '.bin'
                    with open(out_filepath, "wb") as out_file:
                        out_file.write(np.array(padding, dtype=np.uint16).tobytes())
                        out_file.write(np.array([block_size, h, w], dtype=np.uint16).tobytes())
                        for idx in range(x_blocked.size(0)):
                            print(idx)
                            input = x_blocked[idx, :, :, :].unsqueeze(0).to(device)
                            out_enc = model.compress(input)
                            out_file.write(np.array([len(out_enc["strings"][0][0]), len(out_enc["strings"][1][0])],
                                                    dtype=np.uint32).tobytes())
                            out_file.write(out_enc["strings"][0][0])
                            out_file.write(out_enc["strings"][1][0])
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
                    print(x_hat.shape)
                    x_hat = crop(x_hat, pl)
                    print(x_hat.shape)
                    rec = transforms.ToPILImage()(x_hat.squeeze(0))
                    dec_time += time.time() - start_time
                    count += 1
                    ps = compute_psnr(convert(img), convert(rec))
                    print(ps)
                    PSNR += ps
                    BPP += os.path.getsize(out_filepath) * 8 / (x.size(2) * x.size(3))
        PSNR /= count
        BPP /= count
        dec_time /= count
    else:
        codecImport = importlib.import_module("models.codecs")
        codecClass = getattr(codecImport, args.codec)
        tc_args = setup_tc_args(argv[1:])
        codec = codecClass(tc_args)
        subdirs = [x[0] for x in os.walk(tc_args.dataset)]
        file_num = []
        out_list = []
        for dir in subdirs[1:]:
            img_list = []
            for file in os.listdir(dir):
                img_list.append(dir + '/' + file)
            file_num.append(len(img_list))
            _, directory = dir.rsplit('/', 1)
            out_dir = tc_args.code + '/' + directory
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            nargs = [(codec, i, f, out_dir + '/' + f.rsplit('/', 1)[1][:-4] + '.bin', q, tc_args.metrics) for i, q in enumerate(tc_args.qps) for f in img_list]
            num_jobs = tc_args.num_jobs
            pool = mp.Pool(num_jobs) if num_jobs > 1 else None
            rv = pool.starmap(func, nargs)
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
            PSNR += out_list[i]["psnr"] * file_num[i]
            BPP += out_list[i]["bpp"] * file_num[i]
            dec_time += out_list[i]["decoding_time"] * file_num[i]
        PSNR /= sum(file_num)
        BPP /= sum(file_num)
        dec_time /= sum(file_num)
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_Bit-rate: {BPP:.3f} bpp')
    print(f'average_time: {dec_time:.3f} s')

if __name__ == "__main__":
    main(sys.argv[1:])