import enum
import argparse
from pyinstrument import Profiler
import torch
import torch.backends
import json

from coding_tools.register import TOOL_GROUPS
from src.engine import *
from coding_tools.baseline import BPG, VTM, WebP, JPEG
from tester_utils import test_multiple_configs


def parse_args():
    parser = argparse.ArgumentParser()

    # tester args
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-i", "--input", type=str, required=True, help="input glob")
    parser.add_argument("--speedup", nargs="+", type=float, required=True)
    parser.add_argument("--target_bpp", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--loss", nargs=1, type=str, choices=LOSSES.keys(), default=None
    )
    parser.add_argument("--save_image", action="store_true")

    # Profiler args
    parser.add_argument("--profile", action="store_true")

    # Engine args
    parser.add_argument("--tools", nargs="+", type=str, default=TOOL_GROUPS.keys())
    parser.add_argument("--tool_filter", nargs="+", type=str, default=None)
    parser.add_argument("--ctu_size", type=int, default=512)
    parser.add_argument("--mosaic", action="store_true", default=False)

    # Encoder config args (SA)
    parser.add_argument("--num_steps", nargs="+", type=int, default=1000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Basic algorithm tester
    """
    torch.backends.cudnn.enabled = True

    args = parse_args()

    if args.profile:
        profiler = Profiler()
        profiler.start()
    else:
        profiler = None

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
        args.save_image,
        target_bpp=args.target_bpp,
        speedup=args.speedup,
        num_steps=args.num_steps,
        losstype=args.loss,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    result_filename = os.path.join(args.output_dir, "results.json")
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)

    if profiler is not None:
        profiler.stop()
        profile_filename = os.path.join(args.output_dir, "profile.html")
        with open(profile_filename, "w") as f:
            print(profiler.output_html(timeline=False, show_all=False), file=f)
