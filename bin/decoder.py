import torch
import argparse
import time
from PIL import Image
import numpy as np

from bin.engine import Engine
from bin.fileio import FileIO
from bin.utils import dump_torch_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument(
        "--tools", nargs="+", type=str, default=Engine.TOOL_GROUPS.keys()
    )
    parser.add_argument(
        "--tool_filter", nargs="+", type=str, default=None
    )

    args = parser.parse_args()
    return args


def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    engine = Engine(tool_groups=args.tools, tool_filter=args.tool_filter)

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
    print(f"Decode time: {time_end - time_start:.4f}s")

    # Save image
    out_img = dump_torch_image(out_img)
    Image.fromarray(out_img).save(args.output)


if __name__ == "__main__":
    main()
