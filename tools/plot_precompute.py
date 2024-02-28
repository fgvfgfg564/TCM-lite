import argparse
import tempfile
import os
import json
import pathlib
import glob

import torch
import time

import matplotlib.pyplot as plt
import numpy as np

from bin.engine import GAEngine1
from bin.utils import *
import einops


def parse_args():
    parser = argparse.ArgumentParser()

    # tester args
    parser.add_argument("-o", type=str)
    parser.add_argument("-i", "--input", type=str, required=True, help="input glob")

    # Engine args
    parser.add_argument(
        "--tools", nargs="+", type=str, default=GAEngine1.TOOL_GROUPS.keys()
    )
    parser.add_argument("--tool_filter", nargs="+", type=str, default=None)
    parser.add_argument("--ctu_size", type=int, default=512)

    args = parser.parse_args()
    return args

METHOD_COLORS = ["#418DED", "#D8A31A", "#002875", "E6F4F1"]

if __name__ == "__main__":
    # Tester for CVPR 2023 paper
    torch.backends.cudnn.enabled = True

    args = parse_args()

    engine = GAEngine1(
        ctu_size=args.ctu_size,
        tool_groups=args.tools,
        tool_filter=args.tool_filter,
        ignore_tensorrt=True,
        dtype=torch.float32,
    )
    
    input_img = engine.read_img(args.input)
    h, w, padded_img = engine.pad_img(input_img)

    img_blocks = einops.rearrange(
        padded_img,
        "b c (n_ctu_h ctu_size_h) (n_ctu_w ctu_size_w) -> n_ctu_h n_ctu_w b c ctu_size_h ctu_size_w",
        ctu_size_h=engine.ctu_size,
        ctu_size_w=engine.ctu_size,
    )

    n_ctu_h = img_blocks.shape[0]
    n_ctu_w = img_blocks.shape[1]

    engine._precompute_score(img_blocks, (h, w))

    fig, axes = plt.subplots(n_ctu_h, n_ctu_w, figsize=(n_ctu_w*2, n_ctu_h*2))

    for i in range(n_ctu_h):
        for j in range(n_ctu_w):
            ax_e = axes[i, j]
            ax_t = ax_e.twinx()

            for idx in range(len(engine.methods)):
                min_b = engine._precomputed_curve[idx][i][j]["min_t"]
                max_b = engine._precomputed_curve[idx][i][j]["max_t"]
                b = np.linspace(min_b, max_b, 100)
                e = engine._precomputed_curve[idx][i][j]["b_e"](b)
                t = np.polyval(engine._precomputed_curve[idx][i][j]["b_t"], b)

                ax_e.plot(b, e, marker=' ', color=METHOD_COLORS[idx], linestyle='-')
                ax_t.plot(b, t, marker=' ', color=METHOD_COLORS[idx], linestyle='--')
            
            # ax_e.set_xlabel("num_bytes")
            # ax_e.set_ylabel("squared error")
            # ax_t.set_ylabel("dec. time")
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            ax_t.set_yticks([])
    
    plt.tight_layout()

    plt.savefig(args.o+".png", dpi=300)
    plt.savefig(args.o+".pdf")