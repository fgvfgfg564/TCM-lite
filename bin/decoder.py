import torch
import argparse
import time

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
    time_start = time.time()
    engine.decode(args.input, args.output)
    torch.cuda.synchronize()
    time_end = time.time()
    print(f"Decode time: {time_end - time_start:.4f}s")

if __name__ == "__main__":
    main()