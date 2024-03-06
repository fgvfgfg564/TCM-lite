import argparse
import tempfile
import os
import json
import pathlib
import glob

import torch
import time

from bin.engine import *
from bin.utils import *
from coding_tools.register import TOOL_GROUPS


def parse_args():
    parser = argparse.ArgumentParser()

    # tester args
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-i", "--input", type=str, required=True, help="input glob")
    parser.add_argument("--w_time", nargs="+", type=float, default=[1.0])
    parser.add_argument("--target_bpp", nargs="+", type=float, default=[1.0])
    parser.add_argument("--save_image", action="store_true")

    parser.add_argument("-a", "--algorithm", type=str, choices=['GA', 'SAv1'], required=True)

    # Engine args
    parser.add_argument(
        "--tools", nargs="+", type=str, default=TOOL_GROUPS.keys()
    )
    parser.add_argument("--tool_filter", nargs="+", type=str, default=None)
    parser.add_argument("--ctu_size", type=int, default=512)
    parser.add_argument('--mosaic', action='store_true', default=False)

    # Encoder config args (GA)
    parser.add_argument("-N", nargs="+", type=int, default=[1000])
    parser.add_argument("--num-gen", nargs="+", type=int, default=[1000])
    parser.add_argument("--boltzmann_k", nargs="+", type=float, default=[0.05])
    parser.add_argument("--method_sigma", nargs="+", type=float, default=[0.2])
    parser.add_argument("--bytes_sigma", nargs="+", type=float, default=[256])
    parser.add_argument("--no_allocation", nargs="+", type=bool, default=[False])

    # Encoder config args (SA)

    args = parser.parse_args()
    return args


def test_single_image(
    engine: EngineBase,
    input_filename,
    output_dir,
    target_bpp,
    w_time,
    save_image,
    **kwargs,
):
    output_dir = os.path.join(
        output_dir,
        "target_bpp="+str(target_bpp),
        "w_time="+str(w_time),
        *([str(k)+"="+str(v) for k, v in kwargs.items()]),
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
    genetic_statistic = engine.encode(
        input_filename,
        obin,
        target_bpp,
        w_time=w_time,
        **kwargs,
    )
    torch.cuda.synchronize()

    with open(osta, "w") as f:
        json.dump(genetic_statistic, f)

    time_enc = time.time() - time0

    # Load bitstream
    fd = open(obin, "rb")
    bitstream = fd.read()
    fd.close()

    # Decoding process; generate recon image
    time_dec_meter = AverageMeter()
    for i in range(10):
        time_start = time.time()
        file_io: FileIO = FileIO.load(bitstream, engine.mosaic, engine.ctu_size)
        out_img = engine.decode(file_io)  # Decoded image; shape=[3, H, W]
        torch.cuda.synchronize()
        time_end = time.time()
        time_dec = time_end - time_start
        time_dec_meter.update(time_dec)

    # Save image
    out_img = dump_image(out_img)
    Image.fromarray(out_img).save(orec)

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
    target_bpp,
    w_time,
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
            target_bpp=target_bpp,
            w_time=w_time,
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
    target_bpp,
    w_time,
    save_image,
    **kwargs,
):
    output_dir = os.path.join(output_dir, "results")

    def _test_glob(
        target_bpp,
        w_time,
        **kwargs,
    ):
        return test_glob(
            engine,
            input_pattern,
            output_dir,
            target_bpp=target_bpp,
            w_time=w_time,
            save_image=save_image,
            **kwargs
        )

    configs = [("target_bpp", target_bpp), ("w_time", w_time)] + [(k, v) for k, v in kwargs.items()]
    results = _config_mapper(configs, _test_glob)
    return results


if __name__ == "__main__":
    """
    Tester for CVPR 2023 paper
    """
    torch.backends.cudnn.enabled = True

    args = parse_args()

    if args.algorithm == 'GA':
        engine = GAEngine1(
            ctu_size=args.ctu_size,
            mosaic=args.mosaic,
            tool_groups=args.tools,
            tool_filter=args.tool_filter,
            dtype=torch.float32,
        )

        results = test_multiple_configs(
            engine,
            args.input,
            args.output_dir,
            args.target_bpp,
            args.w_time,
            args.save_image,
            N=args.N,
            num_gen=args.num_gen,
            no_allocation=args.no_allocation,
            boltzmann_k=args.boltzmann_k,
            method_sigma=args.method_sigma,
            bytes_sigma=args.bytes_sigma,
        )
    elif args.algorithm == 'SAv1':
        engine = SAEngine1(
            ctu_size=args.ctu_size,
            mosaic=args.mosaic,
            tool_groups=args.tools,
            tool_filter=args.tool_filter,
            dtype=torch.float32,
        )

        results = test_multiple_configs(
            engine,
            args.input,
            args.output_dir,
            args.target_bpp,
            args.w_time,
            args.save_image,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    result_filename = os.path.join(args.output_dir, "results.json")
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
