import torch
import argparse
import time
from PIL import Image
import numpy as np

from bin.engine import Engine
from bin.fileio import FileIO


def dump_torch_image(img):
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    return img


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
    parser.add_argument(
        "--save_statistic", action='store_true'
    )

    args = parser.parse_args()
    return args


def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    engine = Engine(tool_groups=args.tools, tool_filter=args.tool_filter, save_statistic=args.save_statistic)

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
