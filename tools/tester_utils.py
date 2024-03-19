import tempfile
import os
import json
import pathlib
import glob
from collections.abc import Iterable

import torch
import time

from bin.engine import *
from bin.utils import *


def test_single_image(
    engine: EngineBase,
    input_filename,
    output_dir,
    save_image,
    **kwargs,
):
    output_dir = os.path.join(
        output_dir,
        *([str(k) + "=" + str(v) for k, v in kwargs.items()]),
    )
    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        realname = pathlib.Path(input_filename).stem
        obin = os.path.join(output_dir, realname + ".bin")
        orec = os.path.join(output_dir, realname + "_rec.png")
        osta = os.path.join(output_dir, realname + "_statistics.json")
    else:
        _, obin = tempfile.mkstemp()
        _, orec = tempfile.mkstemp(suffix=".png")

    ## Encode
    time0 = time.time()
    encoder_returns = engine.encode(
        input_filename,
        obin,
        **kwargs,
    )
    torch.cuda.synchronize()

    if encoder_returns is not None:
        with open(osta, "w") as f:
            json.dump(encoder_returns, f)

    time_enc = time.time() - time0

    # Decoding process; generate recon image
    time_dec_meter = AverageMeter()
    for i in range(3):
        time_start = time.time()
        out_img = engine.decode(obin, orec)  # Decoded image; shape=[3, H, W]
        torch.cuda.synchronize()
        time_end = time.time()
        time_dec = time_end - time_start
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

    if not save_image:
        os.remove(obin)
        os.remove(orec)

    return results


def test_glob(
    engine,
    input_pattern,
    output_dir,
    save_image,
    **kwargs,
):
    input_glob = glob.glob(input_pattern)

    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()
    avg_t_dec = AverageMeter()
    avg_ms_ssim = AverageMeter()

    results = {}

    for filename in input_glob:
        img_result = test_single_image(
            engine,
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


def _config_mapper(config_list, f):
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

        result_single = _config_mapper(config_list, fnew)
        if len(configs) == 1:
            result = result_single
        else:
            result[f"{config_name}={config}"] = result_single
    return result


def test_multiple_configs(
    engine,
    input_pattern,
    output_dir,
    save_image,
    **kwargs,
):
    output_dir = os.path.join(output_dir, "results")

    def _test_glob(
        **kwargs,
    ):
        return test_glob(
            engine,
            input_pattern,
            output_dir,
            save_image=save_image,
            **kwargs,
        )

    configs = [(k, v) for k, v in kwargs.items()]
    results = _config_mapper(configs, _test_glob)
    return results