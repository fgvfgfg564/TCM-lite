import torch
import torch_tensorrt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
import numpy as np
warnings.filterwarnings("ignore")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def benchmark(model, input_shape, dtype='fp32', nwarmup=50, nruns=100):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print('Average throughput: %.2f images/second'%(input_shape[0]/np.mean(timings)))

def main(argv):
    args = parse_args(argv)
    device = 'cuda:0'
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    net = net.to(device)
    net.eval()
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    
    model = net.g_s
    print(model)
    input_shape = [1, 320, 32, 48]
    benchmark(model, input_shape)

    scripted_model = torch.jit.script(model, example_inputs=[torch.randn(input_shape)])
    print(scripted_model.code, flush=True)

    with torch_tensorrt.logging.debug():
        trt_model = torch_tensorrt.ts.compile(scripted_model, 
            inputs= [torch_tensorrt.Input(input_shape, dtype=torch.half)],
            enabled_precisions= {torch.float, torch.half},
            debug=True,
            require_full_compilation=True,
        )

    benchmark(trt_model, input_shape, dtype='fp16')
    torch.jit.save(trt_model, f"g_s_6.ts")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    