import enum
import argparse
from pyinstrument import Profiler
import torch
import torch.backends
import json

from coding_tools.register import TOOL_GROUPS
from src.engine import *
from coding_tools.baseline import ANCHORS
from tester_utils import config_mapper, test_glob


def parse_args():
    parser = argparse.ArgumentParser()

    # tester args
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-i", "--input", type=str, required=True, help="input glob")
    parser.add_argument("--target_time", nargs="+", type=float, default=[float("inf")])
    parser.add_argument("--target_bpp", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--loss", nargs=1, type=str, choices=LOSSES.keys(), default=None
    )
    parser.add_argument("--save_image", action="store_true")

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
    )

    # Profiler args
    parser.add_argument("--profile", action="store_true")

    # Engine args
    parser.add_argument("--tools", nargs="+", type=str, default=TOOL_GROUPS.keys())
    parser.add_argument("--tool_filter", nargs="+", type=str, default=None)
    parser.add_argument("--ctu_size", type=int, default=512)
    parser.add_argument("--mosaic", action="store_true", default=False)

    # Encoder config args (GA)
    parser.add_argument("-N", nargs="+", type=int, default=[1000])
    parser.add_argument("--num-gen", nargs="+", type=int, default=[1000])
    parser.add_argument("--boltzmann_k", nargs="+", type=float, default=[0.05])
    parser.add_argument("--method_sigma", nargs="+", type=float, default=[0.2])
    parser.add_argument("--bytes_sigma", nargs="+", type=float, default=[256])
    parser.add_argument("--no_allocation", nargs="+", type=bool, default=[False])

    # Encoder config args (SA)
    parser.add_argument("--num_steps", nargs="+", type=int, default=[1000])

    # Encoder config args (BPG)
    parser.add_argument("--qp", nargs="+", type=int, default=None)
    # Encoder config args (WebP, JPEG, VTM and all quality-based codec)
    parser.add_argument("--quality", nargs="+", type=float, default=None)

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

    output_dir = os.path.join(args.output_dir, "results")

    if args.algorithm == "SAv1":
        engine = SAEngine1(
            ctu_size=args.ctu_size,
            mosaic=args.mosaic,
            tool_groups=args.tools,
            tool_filter=args.tool_filter,
            dtype=torch.float32,
        )

        def _test_glob(
            **kwargs,
        ):
            return test_glob(
                engine,
                "encode",
                args.input,
                output_dir,
                save_image=args.save_image,
                **kwargs,
            )

        schedulers = [NStepsScheduler(1.0, 1e-3, nstep) for nstep in args.num_steps]
        configs = [
            ("target_bpp", args.target_bpp),
            ("target_time", args.target_time),
            ("scheduler", schedulers),
            ("losstype", args.loss),
        ]
        results = config_mapper(configs, _test_glob)
    elif args.algorithm in ANCHORS.keys():
        engine = ANCHORS[args.algorithm]()

        def _test_glob(
            quality,
            **kwargs,
        ):
            output_dir_local = os.path.join(output_dir, f"quality-{quality}")
            return test_glob(
                engine,
                "encode",
                args.input,
                output_dir_local,
                quality=quality,
                save_image=args.save_image,
                **kwargs,
            )

        if args.algorithm == "BPG":
            configs = [("qp", args.qp)]
        else:
            configs = [("quality", args.quality)]
        results = config_mapper(configs, _test_glob)

    os.makedirs(args.output_dir, exist_ok=True)
    result_filename = os.path.join(args.output_dir, "results.json")
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)

    if profiler is not None:
        profiler.stop()
        profile_filename = os.path.join(args.output_dir, "profile.html")
        with open(profile_filename, "w") as f:
            print(profiler.output_html(timeline=False, show_all=False), file=f)
