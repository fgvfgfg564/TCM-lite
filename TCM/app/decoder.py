from app.engine import TCMModelEngine
from utils import Timer
from tensorrt_support import InitTRTModelWithPlaceholder
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--model_weight", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    with Timer("Initialize time"):
        args = parse_args()

        with InitTRTModelWithPlaceholder():
            engine = TCMModelEngine("decode")

    print(engine.g_s)
    engine.load(args.model_weight)
    print(engine.g_s)