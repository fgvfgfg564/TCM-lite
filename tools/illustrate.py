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


def parse_args():
    parser = argparse.ArgumentParser()

    # IO args
    parser.add_argument("-o", type=str, help="output image filename")
    parser.add_argument("-i", "--input", type=str, required=True, help="input bytestream")
    parser.add_argument("--image", type=str, help="reconstructed image")

    # illustrator args
    parser.add_argument("--border", action="store_true")
    parser.add_argument("--type", type=str, choices=["num_bytes", "method"], required=True)
    parser.add_argument("--alpha", type=float, default=0.5)

    # Engine args
    parser.add_argument("--ctu_size", type=int)

    args = parser.parse_args()
    return args

METHOD_COLORS = ["#418DED", "#002875", "#D8A31A", "E6F4F1"]

def hex_to_bgr(hex_color):
    # Remove the '#' character if present
    hex_color = hex_color.lstrip('#')

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
    fileio = FileIO.load(args.input, args.ctu_size)
    recon_img = cv2.imread(args.image)
    print("Image shape:", recon_img.shape)

    # Draw mask
    mask = np.zeros_like(recon_img, dtype=np.uint8)
    ctu_size = args.ctu_size

    if args.type == 'method':
        # Method ID
        method_ids = fileio.method_id
        for i in range(fileio.ctu_h):
            for j in range(fileio.ctu_w):
                method_id = method_ids[i, j]
                color = hex_to_bgr(METHOD_COLORS[method_id])
                
                topleft = (i * ctu_size, j * ctu_size)
                bottomright = ((i+1) * ctu_size, (j+1) * ctu_size)
                mask = cv2.rectangle(mask, topleft, bottomright, color, -1)
    else:
        # num_bytes
        num_bytes = fileio.num_bytes

        mx = num_bytes.max()
        mi = num_bytes.min()

        for i in range(fileio.ctu_h):
            for j in range(fileio.ctu_w):
                num_byte = num_bytes[i, j]
                w = 255 - int(255 * (num_bytes - mi) / (mx-mi))
                color = (w, w, w)
                
                topleft = (i * ctu_size, j * ctu_size)
                bottomright = ((i+1) * ctu_size, (j+1) * ctu_size)
                mask = cv2.rectangle(mask, topleft, bottomright, color, 0)

    alpha = args.alpha
    output_img = cv2.addWeighted(recon_img, 1 - alpha, mask, alpha, 0)
    cv2.imwrite(args.o, output_img)