import torch
import argparse

from bin.engine import ModelEngine, MODELS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("--target-bpp", type=float, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys())
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--id_bias", type=int, default=0, help="bias of stored model id in bitstream")

    args = parser.parse_args()
    return args

def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    engine = ModelEngine(args.model)
    engine.compress(args.input, args.target_bpp, args.output, args.id_bias)

if __name__ == "__main__":
    main()