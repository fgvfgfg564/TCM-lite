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


def parse_args():
    parser = argparse.ArgumentParser()

    # tester args
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("-i", "--input", type=str, required=True, help="input glob")

    # Engine args
    parser.add_argument(
        "--tools", nargs="+", type=str, default=Engine.TOOL_GROUPS.keys()
    )
    parser.add_argument("--tool_filter", nargs="+", type=str, default=None)

    # Encoder config args
    parser.add_argument("-N", nargs="+", type=int, default=[1000])
    parser.add_argument("--num-gen", nargs="+", type=int, default=[1000])
    parser.add_argument("--w_time", nargs="+", type=float, default=[1.0])
    parser.add_argument("--bpg_qp", nargs="+", type=int, default=[32])
    parser.add_argument("--boltzmann_k", nargs="+", type=float, default=[0.05])
    parser.add_argument("--method_sigma", nargs="+", type=float, default=[0.2])
    parser.add_argument("--bytes_sigma", nargs="+", type=float, default=[32])
    parser.add_argument("--no_allocation", nargs="+", type=bool, default=[False])

    args = parser.parse_args()
    return args


def test_single_image(
    engine: Engine,
    input_filename,
    output_dir,
    save_image,
    N,
    num_gen,
    bpg_qp,
    w_time,
    no_allocation,
    boltzmann_k,
    method_sigma,
    bytes_sigma,
):
    output_dir = os.path.join(
        output_dir,
        str(N),
        str(num_gen),
        str(bpg_qp),
        str(w_time),
        str(boltzmann_k),
        str(no_allocation),
        str(method_sigma),
        str(bytes_sigma),
    )
    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        realname = pathlib.Path(input_filename).stem
        obin = os.path.join(output_dir, realname + ".bin")
        orec = os.path.join(output_dir, realname + "_rec.png")
    else:
        _, obin = tempfile.mkstemp()
        _, orec = tempfile.mkstemp(suffix=".png")

    ## Encode
    time0 = time.time()
    if not os.path.isfile(obin):
        genetic_statistic = engine.encode(
            input_filename,
            obin,
            N,
            num_gen,
            bpg_qp=bpg_qp,
            w_time=w_time,
            no_allocation=no_allocation,
            boltzmann_k=boltzmann_k,
            method_sigma=method_sigma,
            bytes_sigma=bytes_sigma,
        )
        torch.cuda.synchronize()
    time_enc = time.time() - time0

    # Load bitstream
    fd = open(obin, "rb")
    bitstream = fd.read()
    fd.close()

    # Decoding process; generate recon image
    time_dec_meter = AverageMeter()
    for i in range(10):
        time_start = time.time()
        file_io: FileIO = FileIO.load(bitstream, engine.ctu_size)
        out_img = engine.decode(file_io)  # Decoded image; shape=[3, H, W]
        torch.cuda.synchronize()
        time_end = time.time()
        time_dec = time_end - time_start
        time_dec_meter.update(time_dec)

    # Save image
    out_img = dump_torch_image(out_img)
    Image.fromarray(out_img).save(orec)

    n_bytes = os.path.getsize(obin)
    w, h = get_image_dimensions(input_filename)
    bpp = n_bytes * 8 / (w * h)
    psnr = psnr_with_file(input_filename, orec)

    results = {
        "bpp": bpp,
        "PSNR": psnr,
        "t_enc": time_enc,
        "t_dec": time_dec_meter.avg,
        "genetic_statistic": genetic_statistic,
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
    N,
    num_gen,
    bpg_qp,
    w_time,
    no_allocation,
    boltzmann_k,
    method_sigma,
    bytes_sigma,
):
    input_glob = glob.glob(input_pattern)

    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()
    avg_t_enc = AverageMeter()
    avg_t_dec = AverageMeter()

    results = {}

    for filename in input_glob:
        img_result = test_single_image(
            engine,
            filename,
            output_dir,
            save_image,
            N=N,
            num_gen=num_gen,
            bpg_qp=bpg_qp,
            w_time=w_time,
            no_allocation=no_allocation,
            boltzmann_k=boltzmann_k,
            method_sigma=method_sigma,
            bytes_sigma=bytes_sigma,
        )
        avg_bpp.update(img_result["bpp"])
        avg_psnr.update(img_result["PSNR"])
        avg_t_enc.update(img_result["t_enc"])
        avg_t_dec.update(img_result["t_dec"])

        results[pathlib.Path(filename).stem] = img_result

    results["avg_bpp"] = avg_bpp.avg
    results["avg_psnr"] = avg_psnr.avg
    results["avg_t_enc"] = avg_t_enc.avg
    results["avg_t_dec"] = avg_t_dec.avg

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
    save_image,
    N,
    num_gen,
    bpg_qp,
    w_time,
    no_allocation,
    boltzmann_k,
    method_sigma,
    bytes_sigma,
):
    output_dir = os.path.join(output_dir, "results")

    def _test_glob(
        N,
        num_gen,
        bpg_qp,
        w_time,
        no_allocation,
        boltzmann_k,
        method_sigma,
        bytes_sigma,
    ):
        return test_glob(
            engine,
            input_pattern,
            output_dir,
            save_image,
            N=N,
            num_gen=num_gen,
            bpg_qp=bpg_qp,
            w_time=w_time,
            no_allocation=no_allocation,
            boltzmann_k=boltzmann_k,
            method_sigma=method_sigma,
            bytes_sigma=bytes_sigma,
        )

    configs = [
        ("N", N),
        ("num_gen", num_gen),
        ("bpg_qp", bpg_qp),
        ("w_time", w_time),
        ("no_allocation", no_allocation),
        ("boltzmann_k", boltzmann_k),
        ("method_sigma", method_sigma),
        ("bytes_sigma", bytes_sigma),
    ]
    results = _config_mapper(configs, _test_glob)
    return results


if __name__ == "__main__":
    """
    Tester for CVPR 2023 paper
    """
    torch.backends.cudnn.enabled = True

    args = parse_args()

    engine = Engine(
        tool_groups=args.tools,
        tool_filter=args.tool_filter,
        ignore_tensorrt=True,
        dtype=torch.float32,
    )

    results = test_multiple_configs(
        engine,
        args.input,
        args.output_dir,
        args.save_image,
        args.N,
        args.num_gen,
        args.bpg_qp,
        args.w_time,
        args.no_allocation,
        args.boltzmann_k,
        args.method_sigma,
        args.bytes_sigma,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    result_filename = os.path.join(args.output_dir, "results.json")
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
