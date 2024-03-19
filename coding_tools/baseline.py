import abc
import os
import os.path as osp
from .traditional_tools import WebPTool
from PIL import Image
import numpy as np


class CodecBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, input_pth, output_pth, **kwargs):
        """
        Encode
        """

    @abc.abstractmethod
    def decode(self, input_pth, output_pth):
        """
        Decode
        """


class BPG(CodecBase):
    def encode(
        self,
        input_pth,
        output_pth,
        qp,
        level,
    ):
        input_pth = osp.abspath(input_pth)
        output_pth = osp.abspath(output_pth)
        cmd = f"bpgenc \
            {input_pth} \
            -o {output_pth} \
            -q {qp} \
            -m {level}"
        os.system(cmd)

    def decode(self, input_pth, output_pth):
        input_pth = osp.abspath(input_pth)
        output_pth = osp.abspath(output_pth)
        cmd = f"bpgdec \
            {input_pth} \
            -o {output_pth}"
        os.system(cmd)


class WebP(CodecBase):
    tool = WebPTool()

    def encode(self, input_pth, output_pth, quality):
        img = Image.open(input_pth)
        img = np.asarray(img)
        img_compressed = self.tool.compress_block(img, q_scale=quality)
        with open(output_pth, "wb") as f:
            f.write(img_compressed)

    def decode(self, input_pth, output_pth):
        with open(input_pth, "rb") as f:
            bstr = f.read()
        rec = self.tool.decompress_block(bstr, None, None)
        rec_img = Image.fromarray(rec)
        return rec_img
