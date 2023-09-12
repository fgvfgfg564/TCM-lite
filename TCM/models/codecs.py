import abc
import io
import os
import platform
import subprocess
import sys
import time

from tempfile import mkstemp
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image as Image
import torch

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

def rgb2yuv(rgb):
    r, g, b = np.split(rgb, 3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = np.concatenate([y, cb, cr], axis=-3)
    return ycbcr

def yuv2rgb(yuv):
    y, cb, cr = np.split(yuv, 3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate([r, g, b], axis=-3)
    return rgb

def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return Image.open(filepath).convert(mode)


def _compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = np.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    metrics: Optional[List[str]] = None,
    max_val: float = 255.0,
) -> Dict[str, float]:
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    out = {}
    out["psnr"] = _compute_psnr(a, b, max_val)
    return out

def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)

def _get_bpg_version(encoder_path):
    rv = run_command([encoder_path, "-h"], ignore_returncodes=[1])
    return rv.split()[4]

class Codec(abc.ABC):
    """Abstract base class"""

    _description = None

    def __init__(self, args):
        self._set_args(args)

    def _set_args(self, args):
        return args

    @classmethod
    def setup_args(cls, parser):
        pass

    @property
    def description(self):
        return self._description

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    @abc.abstractmethod
    def _run_impl(self, img, code, quality, *args, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        in_filepath,
        out_filepath,
        quality: int,
        metrics: Optional[List[str]] = None,
        return_rec: bool = False,
    ):
        info, rec = self._run_impl(in_filepath, out_filepath, quality)
        info.update(compute_metrics(rec, self._load_img(in_filepath), metrics))
        if return_rec:
            return info, rec
        return info

class BinaryCodec(Codec):
    """Call an external binary."""

    fmt = None

    @property
    def name(self):
        raise NotImplementedError()

    def _run_impl(self, in_filepath, out_filepath, quality):
        fd0, png_filepath = mkstemp(suffix=".png")

        # Encode
        start = time.time()
        run_command(self._get_encode_cmd(in_filepath, quality, out_filepath))
        enc_time = time.time() - start
        size = filesize(out_filepath)

        # Decode
        start = time.time()
        run_command(self._get_decode_cmd(out_filepath, png_filepath))
        dec_time = time.time() - start

        # Read image
        rec = read_image(png_filepath)
        os.close(fd0)
        os.remove(png_filepath)
        # os.remove(out_filepath)

        img = self._load_img(in_filepath)
        bpp_val = float(size) * 8 / (img.size[0] * img.size[1])

        out = {
            "bpp": bpp_val,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        return out, rec

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        raise NotImplementedError()

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        raise NotImplementedError()

class BPG(BinaryCodec):
    """BPG from Fabrice Bellard."""

    fmt = ".bpg"

    @property
    def name(self):
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.encoder} "
            f"{self.color_mode}"
        )

    @property
    def description(self):
        return f"BPG. BPG version {_get_bpg_version(self.encoder_path)}"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-m",
            choices=["420", "444"],
            default="444",
            help="subsampling mode (default: %(default)s)",
        )
        parser.add_argument(
            "-b",
            choices=["8", "10"],
            default="8",
            help="bitdepth (default: %(default)s)",
        )
        parser.add_argument(
            "-c",
            choices=["rgb", "ycbcr"],
            default="ycbcr",
            help="colorspace  (default: %(default)s)",
        )
        parser.add_argument(
            "-e",
            choices=["jctvc", "x265"],
            default="x265",
            help="HEVC implementation (default: %(default)s)",
        )
        parser.add_argument("--encoder-path", default="bpgenc", help="BPG encoder path")
        parser.add_argument("--decoder-path", default="bpgdec", help="BPG decoder path")

    def _set_args(self, args):
        args = super()._set_args(args)
        self.color_mode = args.c
        self.encoder = args.e
        self.subsampling_mode = args.m
        self.bitdepth = args.b
        self.encoder_path = args.encoder_path
        self.decoder_path = args.decoder_path
        return args

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        if not 0 <= quality <= 51:
            raise ValueError(f"Invalid quality value: {quality} (0,51)")
        cmd = [
            self.encoder_path,
            "-o",
            out_filepath,
            "-q",
            str(quality),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            in_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd

    def decode(self, out_filepath, in_filepath):
        fd0, png_filepath = mkstemp(suffix=".png")

        # Decode
        run_command(self._get_decode_cmd(out_filepath, png_filepath))

        # Read image
        rec = read_image(png_filepath)
        os.close(fd0)
        os.remove(png_filepath)

        return rec

def get_vtm_encoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "EncoderApp", "Linux": "EncoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


def get_vtm_decoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "DecoderApp", "Linux": "DecoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err

class VTM(Codec):
    """VTM: VVC reference software"""

    fmt = ".bin"

    @property
    def description(self):
        return "VTM"

    @property
    def name(self):
        return "VTM"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-b",
            "--build-dir",
            type=str,
            required=True,
            help="VTM build dir",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="VTM config file",
        )
        parser.add_argument(
            "--rgb", action="store_true", help="Use RGB color space (over YCbCr)"
        )

    def _set_args(self, args):
        args = super()._set_args(args)
        self.encoder_path = get_vtm_encoder_path(args.build_dir)
        self.decoder_path = get_vtm_decoder_path(args.build_dir)
        self.config_path = args.config
        self.rgb = args.rgb
        return args

    def _run_impl(self, in_filepath, out_filepath, quality):
        if not 0 <= quality <= 63:
            raise ValueError(f"Invalid quality value: {quality} (0,63)")

        # Taking 8bit input for now
        bitdepth = 8

        # Convert input image to yuv 444 file
        arr = np.asarray(self._load_img(in_filepath))
        fd, yuv_path = mkstemp(suffix=".yuv")

        arr = arr.transpose((2, 0, 1))  # color channel first

        if not self.rgb:
            # convert rgb content to YCbCr
            rgb = arr.astype(np.float32) / (2 ** bitdepth - 1)
            arr = np.clip(rgb2yuv(rgb), 0, 1)
            arr = (arr * (2 ** bitdepth - 1)).astype(np.uint8)

        with open(yuv_path, "wb") as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]
        cmd = [
            self.encoder_path,
            "-i",
            yuv_path,
            "-c",
            self.config_path,
            "-q",
            quality,
            "-o",
            "/dev/null",
            "-b",
            out_filepath,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=444",
            "--InputBitDepth=8",
            "--ConformanceWindowMode=1",
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]
        start = time.time()
        run_command(cmd)
        enc_time = time.time() - start

        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]
        if self.rgb:
            cmd.append("--OutputInternalColourSpace=GBRtoRGB")

        start = time.time()
        run_command(cmd)
        dec_time = time.time() - start

        # Compute PSNR
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)

        rec_arr = rec_arr.astype(np.float32) / (2**bitdepth - 1)
        if not self.rgb:
            rec_arr = yuv2rgb(rec_arr)

        bpp = filesize(out_filepath) * 8.0 / (height * width)

        # Cleanup
        os.unlink(yuv_path)
        # os.unlink(out_filepath)

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        return out, rec

    def decode(self, out_filepath, in_filepath):
        bitdepth = 8
        arr = np.asarray(self._load_img(in_filepath))
        arr = arr.transpose((2, 0, 1))  # color channel first
        fd, yuv_path = mkstemp(suffix=".yuv")
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]
        run_command(cmd)

        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)

        rec_arr = rec_arr.astype(np.float32) / (2 ** bitdepth - 1)
        if not self.rgb:
            rec_arr = yuv2rgb(rec_arr)
        os.unlink(yuv_path)
        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        return rec