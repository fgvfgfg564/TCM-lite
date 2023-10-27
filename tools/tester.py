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
    parser.add_argument("--save_image", action='store_true')
    parser.add_argument("-i", "--input", type=str, required=True, help="input glob")

    # Engine args
    parser.add_argument("-N", type=int, default=1000)
    parser.add_argument("--num-gen", type=int, default=1000)
    parser.add_argument(
        "--tools", nargs="+", type=str, default=Engine.TOOL_GROUPS.keys()
    )
    parser.add_argument(
        "--tool_filter", nargs="+", type=str, default=None
    )
    parser.add_argument("--w_time", nargs='+', type=float, default=[1.0])
    parser.add_argument("--bpg_qp", type=int, default=32)
    parser.add_argument(
        "--save_statistics", action='store_true'
    )
    parser.add_argument(
        "--no_allocation", action='store_true'
    )

    args = parser.parse_args()
    return args

def test_single_image(engine: Engine, input_filename, N, num_gen, output_dir, save_image, bpg_qp, w_time):
    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        realname = pathlib.Path(input_filename).stem
        obin = os.path.join(output_dir, realname+".bin")
        orec = os.path.join(output_dir, realname+"_rec.png")
    else:
        _, obin = tempfile.mkstemp()
        _, orec = tempfile.mkstemp(suffix='.png')

    ## Encode
    time0 = time.time()
    genetic_statistic = engine.encode(input_filename, obin, N, num_gen, bpg_qp=bpg_qp, w_time=w_time)
    torch.cuda.synchronize()
    time_enc = time.time() - time0

    # Load bitstream
    fd = open(obin, "rb")
    bitstream = fd.read()
    fd.close()

    # Decoding process; generate recon image
    time_start = time.time()
    file_io: FileIO = FileIO.load(bitstream, engine.ctu_size)
    out_img = engine.decode(file_io)  # Decoded image; shape=[3, H, W]
    torch.cuda.synchronize()
    time_end = time.time()
    time_dec = time_end - time_start

    # Save image
    out_img = dump_torch_image(out_img)
    Image.fromarray(out_img).save(orec)

    n_bytes = os.path.getsize(obin)
    w, h = get_image_dimensions(input_filename)
    bpp = n_bytes * 8 / (w*h)
    psnr = psnr_with_file(input_filename, orec)

    results = {
        "bpp": bpp,
        "PSNR": psnr,
        "t_enc": time_enc,
        "t_dec": time_dec,
        "genetic_statistic": genetic_statistic
    }

    if not save_image:
        os.remove(obin)
        os.remove(orec)

    return results

def test_glob(engine, input_pattern, N, num_gen, output_dir, save_image, bpg_qp, w_time):
    input_glob = glob.glob(input_pattern)

    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()
    avg_t_enc = AverageMeter()
    avg_t_dec = AverageMeter()
    
    results = {}

    for filename in input_glob:
        img_result = test_single_image(engine, filename, N, num_gen, output_dir, save_image, bpg_qp=bpg_qp, w_time=w_time)
        avg_bpp.update(img_result['bpp'])
        avg_psnr.update(img_result['PSNR'])
        avg_t_enc.update(img_result['t_enc'])
        avg_t_dec.update(img_result['t_dec'])

        results[pathlib.Path(filename).stem] = img_result
    
    results["avg_bpp"] = avg_bpp.avg
    results["avg_psnr"] = avg_psnr.avg
    results["avg_t_enc"] = avg_t_enc.avg
    results["avg_t_dec"] = avg_t_dec.avg

    return results

def test_multiple_w_time(engine, input_pattern, N, num_gen, output_dir, save_image, bpg_qp, w_time):
    output_dir = os.path.join(output_dir, "results")
    results = {}
    for w_time_sample in w_time:
        output_dir_w = os.path.join(output_dir, str(w_time_sample))
        result = test_glob(engine, input_pattern, N, num_gen, output_dir_w, save_image, bpg_qp, w_time_sample)
        results[w_time_sample] = result
    return results

if __name__ == "__main__":
    """
    Tester for CVPR 2023 paper
    """
    torch.backends.cudnn.enabled = True

    args = parse_args()

    engine = Engine(tool_groups=args.tools, tool_filter=args.tool_filter, ignore_tensorrt=True, save_statistic=args.save_statistics, no_allocation=args.no_allocation, dtype=torch.float32)

    if len(args.w_time) == 1:
        results = test_glob(engine, args.input, args.N, args.num_gen, os.path.join(args.output_dir, "results"), args.save_image, args.bpg_qp, args.w_time[0])
    else:
        results = test_multiple_w_time(engine, args.input, args.N, args.num_gen, args.output_dir, args.save_image, args.bpg_qp, args.w_time)

    os.makedirs(args.output_dir, exist_ok=True)
    result_filename = os.path.join(args.output_dir, "results.json")
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    