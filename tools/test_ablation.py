import enum
import argparse
from pyinstrument import Profiler
import torch
import torch.backends
import json

from coding_tools.register import TOOL_GROUPS
from src.SA_scheduler import TimeLapseScheduler
from src.engine import *
from coding_tools.baseline import BPG, VTM, WebP, JPEG
from tester_utils import test_glob, config_mapper


def parse_args():
    parser = argparse.ArgumentParser()

    # tester args
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-i", "--input", type=str, required=True, help="input glob")
    parser.add_argument("--speedup", nargs="+", type=float, default=[1.25])
    parser.add_argument(
        "--qscale", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    parser.add_argument(
        "--loss", nargs=1, type=str, choices=LOSSES.keys(), default=["PSNR"]
    )
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("-o", type=str, default="results.json")

    # Engine args
    parser.add_argument("--tools", nargs="+", type=str, default=TOOL_GROUPS.keys())
    parser.add_argument("--tool_filter", nargs="+", type=str, default=None)
    parser.add_argument("--ctu_size", type=int, default=512)
    parser.add_argument("--mosaic", action="store_true", default=False)

    # Encoder config args (SA)
    parser.add_argument("--time_limits", nargs="+", type=int, required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Basic algorithm tester
    """
    torch.backends.cudnn.enabled = True

    args = parse_args()
    output_dir = os.path.join(args.output_dir, "results")

    engine = SAEngine1(
        ctu_size=args.ctu_size,
        mosaic=args.mosaic,
        tool_groups=args.tools,
        tool_filter=args.tool_filter,
        dtype=torch.float32,
    )

    def _test_glob(
        losstype,
        speedup,
        qscale,
        time_limit,
        level,
        **kwargs,
    ):
        output_dir_local = os.path.join(
            args.output_dir,
            losstype,
            f"level-{level}",
            f"time_limit-{time_limit}",
            f"qscale-{qscale}",
            f"speedup-{speedup}",
        )
        scheduler = TimeLapseScheduler(0.1, 0.99, time_limit)
        return test_glob(
            engine,
            "accelerate",
            args.input,
            output_dir_local,
            save_image=args.save_image,
            losstype=losstype,
            speedup=speedup,
            qscale=qscale,
            scheduler=scheduler,
            level=level,
            **kwargs,
        )

    configs = [
        ("level", args.levels),
        ("losstype", args.loss),
        ("qscale", args.qscale),
        ("speedup", args.speedup),
        ("time_limit", args.time_limits),
    ]
    results = config_mapper(configs, _test_glob)

    os.makedirs(args.output_dir, exist_ok=True)
    result_filename = os.path.join(args.output_dir, args.o)
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
