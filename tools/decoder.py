import enum
import argparse
from pyinstrument import Profiler
import torch
import torch.backends
import json

from coding_tools.register import TOOL_GROUPS
from src.engine import *
from coding_tools.baseline import BPG, WebP, JPEG
from tester_utils import test_multiple_configs


class AlgorithmType(enum.Enum):
    GA = enum.auto()
    SAv1 = enum.auto()
    BPG = enum.auto()
    WebP = enum.auto()
    JPEG = enum.auto()


ALGORITHMS = AlgorithmType.__members__.keys()


def parse_args():
    parser = argparse.ArgumentParser()

    # tester args
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output filename"
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="input filename")

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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Basic algorithm tester
    """
    torch.backends.cudnn.enabled = True

    args = parse_args()

    algorithm: AlgorithmType = getattr(AlgorithmType, args.algorithm)

    if algorithm == AlgorithmType.GA:
        engine = GAEngine1(
            ctu_size=args.ctu_size,
            mosaic=args.mosaic,
            tool_groups=args.tools,
            tool_filter=args.tool_filter,
            dtype=torch.float32,
        )
    elif algorithm == AlgorithmType.SAv1:
        engine = SAEngine1(
            ctu_size=args.ctu_size,
            mosaic=args.mosaic,
            tool_groups=args.tools,
            tool_filter=args.tool_filter,
            dtype=torch.float32,
        )
    elif algorithm == AlgorithmType.BPG:
        engine = BPG()
    elif algorithm in [AlgorithmType.JPEG, AlgorithmType.WebP]:
        if algorithm == AlgorithmType.JPEG:
            engine = JPEG()
        else:
            engine = WebP()
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    if args.profile:
        profiler = Profiler()
        profiler.start()
    else:
        profiler = None

    engine.decode(args.input, args.output)

    if profiler is not None:
        profiler.stop()
        profile_filename = os.path.join(args.output + "-profile.html")
        with open(profile_filename, "w") as f:
            print(profiler.output_html(timeline=False, show_all=False), file=f)
