import torch
import argparse

from bin.engine import Engine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--num-gen", type=int, default=100)
    parser.add_argument("--tools", nargs='+', type=str, default=Engine.TOOL_GROUPS.keys())

    args = parser.parse_args()
    return args

def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    engine = Engine(tool_groups=args.tools)
    engine.encode(args.input, args.output, args.num_gen)

if __name__ == "__main__":
    main()