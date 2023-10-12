from traditional.tool_base import TraditionalCodingToolBase, PSNR
import os
import io
import torch
import numpy as np
from tempfile import mkstemp
from torchvision import transforms
from TCM.models import rgb2yuv, yuv2rgb, run_command

from PIL import Image

# 实现的传统编码工具放在这里

class VTMTool(TraditionalCodingToolBase):

    def __init__(self):
        self.encoder_path = "../third_party/VVCSoftware_VTM/bin/EncoderAppStatic"
        self.decoder_path = "../third_party/VVCSoftware_VTM/bin/DecoderAppStatic"
        self.config_path = "../third_party/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"

    def compress_block(self, img_block: torch.Tensor, q_scale: float) -> bytes:
        if not 0 <= q_scale <= 63:
            raise ValueError(f"Invalid quality value: {q_scale} (0,63)")

        bitdepth = 8
        arr = np.asarray(transforms.ToPILImage()(img_block.squeeze(0)))
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
        arr = arr.transpose((2, 0, 1))
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
        with open(out_filepath, 'rb') as f:
            bit_stream = f.read()
        os.close(fd)
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        return bit_stream

    def decompress_block(self, bit_stream: bytes, h: int, w: int, q_scale: float) -> torch.Tensor:
        bitdepth = 8
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
        with open(out_filepath, 'wb') as f:
            f.write(bit_stream)
        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]
        run_command(cmd)
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape((3, h, w))
        rec_arr = rec_arr.astype(np.float32) / (2 ** bitdepth - 1)
        rec_arr = yuv2rgb(rec_arr)
        os.close(fd)
        os.unlink(yuv_path)
        os.unlink(out_filepath)
        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        rec = transforms.ToTensor()(rec).unsqueeze(0)

        return rec

class WebPTool(TraditionalCodingToolBase):

    def __init__(self):
        self.fmt = "webp"

    def compress_block(self, img_block: torch.Tensor, q_scale: float) -> bytes:
        img = transforms.ToPILImage()(img_block.squeeze(0))
        # Encode
        tmp = io.BytesIO()
        img.save(tmp, format=self.fmt, quality=int(q_scale))
        tmp.seek(0)
        bit_stream = tmp.read()

        return bit_stream

    def decompress_block(self, bit_stream: bytes, h: int, w: int, q_scale: float) -> torch.Tensor:
        fd, webp_path = mkstemp("." + self.fmt)
        with open(webp_path, 'wb') as f:
            f.write(bit_stream)
        # Decode
        rec_img = Image.open(webp_path)
        rec_img.load()
        rec = transforms.ToTensor()(rec_img).unsqueeze(0)
        os.close(fd)
        os.unlink(webp_path)

        return rec

if __name__ == "__main__":
    img_block = Image.open('/home/ubuntu/PycharmProjects/DataSet/temp/ZR0_0613_0721363630_582EBY_N0301172ZCAM03485_1100LMJ01.png').convert('RGB')
    img_block = transforms.ToTensor()(img_block)
    img_block = img_block.unsqueeze(0)
    h, w = img_block.size(2), img_block.size(3)
    codec = WebPTool()
    bit_stream = codec.compress_block(img_block, 35)
    bpp = len(bit_stream) * 8 / (h * w)
    print(bpp)
    rec = codec.decompress_block(bit_stream, h, w, 35)
    psnr = PSNR(img_block, rec)
    print(psnr)