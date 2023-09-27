import torch
import argparse

from bin.engine import ModelEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--id_bias", type=int, default=0, help="bias of stored model id in bitstream")

    args = parser.parse_args()
    return args



def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    model_id = ModelEngine.get_model_id(args.input, args.id_bias)
    engine = ModelEngine.from_model_id(model_id)
    engine.decompress(args.input, args.output, args.id_bias)

if __name__ == "__main__":
    with torch.no_grad():
        main()