import torch
import argparse
import time
import pathlib
from PIL import Image
import numpy as np

from src.engine import GAEngine1
from src.fileio import FileIO
from src.utils import dump_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-r", "--original", type=str, required=True)
    parser.add_argument("--logfile", type=str, required=True)
    parser.add_argument(
        "--tools", nargs="+", type=str, default=GAEngine1.TOOL_GROUPS.keys()
    )
    parser.add_argument("--tool_filter", nargs="+", type=str, default=None)

    args = parser.parse_args()
    return args


def psnr(img1, img2):
    img1 = np.array(Image.open(img1)).astype(np.float32)
    img2 = np.array(Image.open(img2)).astype(np.float32)

    print(img1.shape)

    mse = np.mean((img1 - img2) ** 2)

    return -10 * np.log10(mse / (255**2))


def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    engine = GAEngine1(tool_groups=args.tools, tool_filter=args.tool_filter)

    # Load bitstream
    fd = open(args.input, "rb")
    bitstream = fd.read()
    fd.close()

    # Decoding process; generate recon image
    time_start = time.time()
    file_io: FileIO = FileIO.load(bitstream, engine.ctu_size)
    out_img = engine.decode(file_io)  # Decoded image; shape=[3, H, W]
    torch.cuda.synchronize()
    time_end = time.time()
    t_dec = time_end - time_start

    # Save image
    out_img = dump_image(out_img)
    Image.fromarray(out_img).save(args.output)

    PSNR = psnr(args.output, args.original)
    with open(args.logfile, "a") as f:
        print(
            pathlib.Path(args.input).stem, PSNR, t_dec, PSNR - t_dec, sep="\t", file=f
        )


if __name__ == "__main__":
    main()
