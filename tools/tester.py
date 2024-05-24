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


class AlgorithmType(enum.Enum):
    GA = enum.auto()
    SAv1 = enum.auto()
    BPG = enum.auto()
    WebP = enum.auto()
    JPEG = enum.auto()
    VTM = enum.auto()


ALGORITHMS = AlgorithmType.__members__.keys()


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
        choices=ALGORITHMS,
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
    parser.add_argument("--level", nargs="+", type=int, default=None)
    # Encoder config args (WebP, JPEG)
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

    algorithm: AlgorithmType = getattr(AlgorithmType, args.algorithm)

    if algorithm == AlgorithmType.GA:
        engine = GAEngine1(
            ctu_size=args.ctu_size,
            mosaic=args.mosaic,
            tool_groups=args.tools,
            tool_filter=args.tool_filter,
            dtype=torch.float32,
        )

        results = test_multiple_configs(
            engine,
            "encode",
            input_pattern=args.input,
            output_dir=args.output_dir,
            target_bpp=args.target_bpp,
            target_time=args.target_time,
            save_image=args.save_image,
            N=args.N,
            num_gen=args.num_gen,
            no_allocation=args.no_allocation,
            boltzmann_k=args.boltzmann_k,
            method_sigma=args.method_sigma,
            bytes_sigma=args.bytes_sigma,
        )
    elif algorithm == AlgorithmType.SAv1:
        engine = SAEngine1(
            ctu_size=args.ctu_size,
            mosaic=args.mosaic,
            tool_groups=args.tools,
            tool_filter=args.tool_filter,
            dtype=torch.float32,
        )

        results = test_multiple_configs(
            engine,
            "encode",
            args.input,
            args.output_dir,
            args.save_image,
            target_bpp=args.target_bpp,
            target_time=args.target_time,
            num_steps=args.num_steps,
            losstype=args.loss,
        )
    elif algorithm == AlgorithmType.BPG:
        engine = BPG()
        results = test_multiple_configs(
            engine,
            "encode",
            args.input,
            args.output_dir,
            args.save_image,
            qp=args.qp,
            level=args.level,
        )
    elif algorithm in [AlgorithmType.JPEG, AlgorithmType.WebP, AlgorithmType.VTM]:
        if algorithm == AlgorithmType.JPEG:
            engine = JPEG()
        elif algorithm == AlgorithmType.WebP:
            engine = WebP()
        elif algorithm == AlgorithmType.VTM:
            engine = VTM()
        results = test_multiple_configs(
            engine,
            "encode",
            args.input,
            args.output_dir,
            args.save_image,
            quality=args.quality,
        )
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_filename = os.path.join(args.output_dir, "results.json")
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)

    if profiler is not None:
        profiler.stop()
        profile_filename = os.path.join(args.output_dir, "profile.html")
        with open(profile_filename, "w") as f:
            print(profiler.output_html(timeline=False, show_all=False), file=f)
