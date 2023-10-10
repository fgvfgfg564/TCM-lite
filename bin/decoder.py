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

    args = parser.parse_args()
    return args

def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    engine = Engine()

    # Load bitstream
    file_io: FileIO = FileIO.load(args.input, engine.ctu_size)

    # Decoding process; generate recon image
    time_start = time.time()
    out_img = engine.decode(file_io) # Decoded image; shape=[3, H, W]
    torch.cuda.synchronize()
    time_end = time.time()
    print(f"Decode time: {time_end - time_start:.4f}s")

    # Save image
    out_img = dump_torch_image(out_img)
    Image.fromarray(out_img).save(args.output)

if __name__ == "__main__":
    main()