from coding_tools.traditional.tool_base import TraditionalCodingToolBase, PSNR
import os
from PIL import Image
import io
import math
import torch
import numpy as np
from tempfile import mkstemp
from torchvision import transforms
from coding_tools.TCM.models import rgb2yuv, yuv2rgb, run_command
from coding_tools.utils.timer import Timer
import time
from ..register import register_tool

from PIL import Image

# 实现的传统编码工具放在这里

@register_tool('VTM')
class VTMTool(TraditionalCodingToolBase):
    def __init__(self):
        self.encoder_path = "../third_party/VVCSoftware_VTM/bin/EncoderAppStatic"
        self.decoder_path = "../third_party/VVCSoftware_VTM/bin/DecoderAppStatic"
        self.config_path = "../third_party/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"

    def compress_block(self, img_block: torch.Tensor, q_scale: float) -> bytes:
        q_scale = math.ceil(q_scale * 63)
        if not 0.0 <= q_scale <= 63.0:
            raise ValueError(f"Invalid quality value: {q_scale} (0,63)")
        bitdepth = 8
        arr = np.asarray(transforms.ToPILImage()(img_block.squeeze(0)))
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
        arr = arr.transpose((2, 0, 1))
        rgb = arr.astype(np.float32) / (2**bitdepth - 1)
        arr = np.clip(rgb2yuv(rgb), 0, 1)
        arr = (arr * (2**bitdepth - 1)).astype(np.uint8)
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
            q_scale,
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
        run_command(cmd)
        with open(out_filepath, "rb") as f:
            bit_stream = f.read()
        os.close(fd)
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        return bit_stream

    def decompress_block(
        self, bit_stream: bytes, h: int, w: int, q_scale: float
    ) -> torch.Tensor:
        bitdepth = 8
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
        with open(out_filepath, "wb") as f:
            f.write(bit_stream)
        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]
        run_command(cmd)
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape((3, h, w))
        rec_arr = rec_arr.astype(np.float32) / (2**bitdepth - 1)
        rec_arr = yuv2rgb(rec_arr)
        os.close(fd)
        os.unlink(yuv_path)
        os.unlink(out_filepath)
        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        rec = transforms.ToTensor()(rec).unsqueeze(0)

        return rec
    
class PILTool(TraditionalCodingToolBase):
    fmt = None

    def compress_block(self, img_block: np.ndarray, q_scale: float) -> bytes:
        q_scale = 100 * (1 - q_scale)
        if not 0.0 <= q_scale <= 100.0:
            raise ValueError(f"Invalid quality value: {q_scale} (0,100)")
        img = Image.fromarray(img_block)
        # Encode
        tmp = io.BytesIO()
        img.save(tmp, format=self.fmt, quality=int(q_scale))
        tmp.seek(0)
        bit_stream = tmp.read()

        return bit_stream

    def decompress_block(
        self, bit_stream: bytes, h: int, w: int, q_scale: float
    ) -> np.ndarray:
        with Timer("bytesio"):
            image_bytes = io.BytesIO(bit_stream)
        # Decode
        with Timer("open"):
            rec_img = Image.open(image_bytes, formats=(self.fmt,))
        with Timer("load"):
            rec_img.load()
            rec_img = np.asarray(rec_img)

        return rec_img

@register_tool('WebP')
class WebPTool(PILTool):
    fmt = 'webp'


@register_tool('JPEG')
class JPEGTool(PILTool):
    fmt = 'jpeg'

if __name__ == "__main__":
    quality = 0.5
    img_block = Image.open(
        "/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/DOG_4507.png"
    ).convert("RGB")
    img_block = transforms.ToTensor()(img_block)
    img_block = img_block.unsqueeze(0)
    h, w = img_block.size(2), img_block.size(3)
    codec = WebPTool()
    bit_stream = codec.compress_block(img_block, quality)
    bpp = len(bit_stream) * 8 / (h * w)
    print(bpp)

    time0 = time.time()
    rec = codec.decompress_block(bit_stream, h, w, quality)
    time1 = time.time()
    psnr = PSNR(img_block, rec)
    print("PSNR", psnr)
    dec_time = time1 - time0
    print("Dec time:", dec_time)
