import argparse
import tempfile
import os
import json
import pathlib
import glob

import torch
import time
import cv2

from bin.engine import *
from bin.utils import *

import matplotlib.pyplot as plt
from matplotlib import colors

CMAP = plt.get_cmap("bwr")


def get_color(position):
    return list([int(x * 255) for x in CMAP(position)[:3]])[::-1]


def parse_args():
    parser = argparse.ArgumentParser()

    # IO args
    parser.add_argument("-o", type=str, help="output image filename")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="input bytestream"
    )
    parser.add_argument("--image", type=str, help="reconstructed image")

    # illustrator args
    parser.add_argument("--border", action="store_true")
    parser.add_argument(
        "--type", type=str, choices=["num_bytes", "method"], required=True
    )
    parser.add_argument("--alpha", type=float, default=0.5)

    # Engine args
    parser.add_argument("--ctu_size", type=int, required=True)
    parser.add_argument("--mosaic", action="store_true")

    args = parser.parse_args()
    return args


METHOD_COLORS = ["#418DED", "#D8A31A", "#002875", "E6F4F1"]


def hex_to_bgr(hex_color):
    # Remove the '#' character if present
    hex_color = hex_color.lstrip("#")

    # Convert the hexadecimal color to integers
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Create a BGR color in OpenCV format
    bgr_color = (b, g, r)

    return bgr_color


if __name__ == "__main__":
    """
    Bytestream illustrator for CVPR 2023 paper
    """
    torch.backends.cudnn.enabled = True

    args = parse_args()

    print(args.image, flush=True)
    fileio = FileIO.load(args.input, args.mosaic, args.ctu_size)
    recon_img = cv2.imread(args.image)
    print("Image shape:", recon_img.shape)

    # Draw mask
    mask = np.zeros_like(recon_img, dtype=np.uint8)
    ctu_size = args.ctu_size

    if args.type == "method":
        # Method ID
        method_ids = fileio.method_id
        for i in range(fileio.n_ctu):
            method_id = method_ids[i]
            bbox = fileio.block_indexes[i]
            print(fileio.n_ctu, i, method_id, bbox)
            color = hex_to_bgr(METHOD_COLORS[method_id])

            topleft = (bbox[1], bbox[0])
            bottomright = (bbox[3], bbox[2])
            mask = cv2.rectangle(mask, topleft, bottomright, color, -1)
    else:
        # num_bytes
        bpps = []
        for i in range(fileio.n_ctu):
            num_byte = fileio.num_bytes[i]
            top, left, bottom, right = fileio.block_indexes[i]
            bottom = min(bottom, fileio.h)
            right = min(right, fileio.w)
            num_pixels = (bottom - top) * (right - left)
            bpps.append(num_byte / num_pixels)

        mx = max(bpps)
        mi = min(bpps)

        BOUND = 0

        for i in range(fileio.n_ctu):
            num_byte = bpps[i]
            w = (num_byte - mi) / (mx - mi)
            color = get_color(w)

            bbox = fileio.block_indexes[i]
            topleft = (bbox[1], bbox[0])
            bottomright = (bbox[3], bbox[2])
            mask = cv2.rectangle(mask, topleft, bottomright, color, -1)

    alpha = args.alpha
    output_img = cv2.addWeighted(recon_img, 1 - alpha, mask, alpha, 0)
    # output_img = np.pad(output_img, ((0, 1), (0, 1), (0, 0)), mode='constant')

    # Draw grid
    for i in range(fileio.n_ctu):
        bbox = fileio.block_indexes[i]
        color = (0, 0, 0)

        bbox = fileio.block_indexes[i]
        topleft = (bbox[1], bbox[0])
        bottomright = (bbox[3], bbox[2])
        output_img = cv2.rectangle(output_img, topleft, bottomright, color, 4)

    cv2.imwrite(args.o, output_img)
