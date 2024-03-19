from typing_extensions import Union, Any

import abc
import os
import os.path as osp
from .traditional_tools import JPEGTool, WebPTool, TraditionalCodingToolBase
from PIL import Image
import numpy as np


class CodecBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, input_pth, output_pth, **kwargs) -> Union[Any, None]:
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
        print("Encoding ...", cmd)
        os.system(cmd)

    def decode(self, input_pth, output_pth):
        input_pth = osp.abspath(input_pth)
        output_pth = osp.abspath(output_pth)
        cmd = f"bpgdec \
            {input_pth} \
            -o {output_pth}"
        print("Decoding ...", cmd)
        os.system(cmd)


class ToolBasedCodec(CodecBase, abc.ABC):
    @abc.abstractproperty
    def tool() -> TraditionalCodingToolBase:
        pass

    def encode(self, input_pth, output_pth, quality):
        print("Encoding image:", input_pth, "with quality", quality)
        img = Image.open(input_pth)
        img = np.asarray(img)
        img_compressed = self.tool.compress_block(img, q_scale=quality)
        with open(output_pth, "wb") as f:
            f.write(img_compressed)

    def decode(self, input_pth, output_pth):
        print("Decoding image:", input_pth)
        with open(input_pth, "rb") as f:
            bstr = f.read()
        rec = self.tool.decompress_block(bstr, None, None)
        rec_img = Image.fromarray(rec)
        rec_img.save(output_pth)


class WebP(ToolBasedCodec):
    tool = WebPTool()


class JPEG(ToolBasedCodec):
    tool = JPEGTool()
