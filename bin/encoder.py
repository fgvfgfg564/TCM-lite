import torch
import argparse

from bin.engine import Engine

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
    engine.encode(args.input, args.output)

if __name__ == "__main__":
    main()