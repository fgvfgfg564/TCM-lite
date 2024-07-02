import tempfile
import os
import json
import pathlib
import glob
from collections.abc import Iterable

import torch
import time

from src.engine import *
from src.utils import *

import random
import numpy as np


def reset_random_seeds(seed=42):
    random.seed(seed)  # Reset random seed for Python's random module
    np.random.seed(seed)  # Reset random seed for NumPy
    torch.manual_seed(seed)  # Reset random seed for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Reset random seed for CUDA operations in PyTorch
        torch.cuda.manual_seed_all(seed)  # Reset random seed for all GPUs in PyTorch


def test_single_image(
    engine: EngineBase,
    entrance: str,
    input_filename,
    output_dir,
    save_image,
    **kwargs,
):
    print(f"Testing: input={input_filename}; output_dir={output_dir}; kwargs={kwargs}")
    os.makedirs(output_dir, exist_ok=True)
    realname = pathlib.Path(input_filename).stem
    obin = os.path.join(output_dir, realname + ".bin")
    orec = os.path.join(output_dir, realname + "_rec.bmp")
    osta = os.path.join(output_dir, realname + "_statistics.json")

    reset_random_seeds()

    ## Encode
    if not os.path.isfile(obin):
        entrance_func = engine.__getattribute__(entrance)
    else:
        print(f"{obin} already exists. Using fix mode.")
        entrance_func = engine.accelerate_fix
    time0 = time.time()
    encoder_returns = entrance_func(
        input_filename,
        obin,
        **kwargs,
    )
    torch.cuda.synchronize()

    if encoder_returns is not None:
        with open(osta, "w") as f:
            json.dump(encoder_returns, f, indent=4, sort_keys=True)

    time_enc = time.time() - time0

    # Decoding process; generate recon image
    engine.decode(obin, orec)  # Preheat
    time_dec_meter = AverageMeter()
    for i in range(3):
        time_dec = engine.decode(obin, orec)  # Decoded image; shape=[3, H, W]
        print(f"Decode time={time_dec:.5f}s")
        time_dec_meter.update(time_dec)

    n_bytes = os.path.getsize(obin)
    w, h = get_image_dimensions(input_filename)
    bpp = n_bytes * 8 / (w * h)
    psnr = psnr_with_file(input_filename, orec)
    ms_ssim = msssim_with_file(input_filename, orec)

    results = {
        "bpp": bpp,
        "PSNR": psnr,
        "MS-SSIM": ms_ssim,
        "t_dec": time_dec_meter.avg,
    }
    print(results)

    if not save_image:
        os.remove(orec)

    return results


def test_glob(
    engine,
    entrance,
    input_pattern,
    output_dir,
    save_image,
    **kwargs,
):
    input_glob = glob.glob(input_pattern)
    random.shuffle(input_glob)

    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()
    avg_t_dec = AverageMeter()
    avg_ms_ssim = AverageMeter()

    results = {}

    for filename in input_glob:
        img_result = test_single_image(
            engine,
            entrance,
            filename,
            output_dir,
            save_image=save_image,
            **kwargs,
        )
        avg_bpp.update(img_result["bpp"])
        avg_psnr.update(img_result["PSNR"])
        avg_t_dec.update(img_result["t_dec"])
        avg_ms_ssim.update(img_result["MS-SSIM"])

        results[pathlib.Path(filename).stem] = img_result

    results["avg_bpp"] = avg_bpp.avg
    results["avg_psnr"] = avg_psnr.avg
    results["avg_t_dec"] = avg_t_dec.avg
    results["avg_ms_ssim"] = avg_ms_ssim.avg

    return results


def config_mapper(config_list, f):
    if len(config_list) == 0:
        return f()

    config_name, configs = config_list[0]
    config_list = config_list[1:]
    result = {}

    # Make sure config is iterable
    if not isinstance(configs, Iterable):
        raise TypeError(
            f"Config '{config_name}' should be an iterable. Found '{configs}'"
        )

    for config in configs:

        def fnew(**kwargs):
            kwargs[config_name] = config
            return f(**kwargs)

        result_single = config_mapper(config_list, fnew)
        if len(configs) == 1:
            result = result_single
        else:
            result[f"{config_name}={config}"] = result_single
    return result
