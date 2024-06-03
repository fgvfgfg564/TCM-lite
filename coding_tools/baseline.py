from io import FileIO
import time
import torch
from typing_extensions import Union, Any, Dict, Type

import abc
import os
import os.path as osp
from .traditional_tools import JPEGTool, VTMTool, WebPTool, TraditionalCodingToolBase
from .register import TOOL_GROUPS
from .coding_tool import LICToolBase
from PIL import Image
import numpy as np
import tempfile

from src.utils import divide_blocks, join_blocks, torch_float_to_np_uint8
from src.fileio import FileIO


class CodecBase(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def encode(self, input_pth, output_pth, **kwargs) -> Union[Any, None]:
        """
        Encode
        """

    def decode(self, input_pth, output_pth):
        """
        Accurate way to compute decoding time: decode into a temporary bmp file and count for time
        """
        tmpbmp = tempfile.mktemp(suffix=".bmp", dir="/tmp")
        t_start = time.time()
        self._decode(input_pth, tmpbmp)
        torch.cuda.synchronize()
        t_end = time.time()
        img = Image.open(tmpbmp)
        img.save(output_pth)
        return t_end - t_start

    @abc.abstractmethod
    def _decode(self, input_pth: str, output_pth: str) -> None:
        """
        Decode
        Returns:
         - decoding time
        """


ANCHORS: Dict[str, Type[CodecBase]] = {}


def register_anchor(name):
    def register(cls):
        ANCHORS[name] = cls
        return cls

    return register


@register_anchor("BPG")
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

    def _decode(self, input_pth, output_pth):
        input_pth = osp.abspath(input_pth)
        output_pth = osp.abspath(output_pth)
        cmd = f"bpgdec \
            {input_pth} \
            -o {output_pth}"
        print("Decoding ...", cmd)
        os.system(cmd)


class ToolBasedCodec(CodecBase, abc.ABC):

    @property
    @abc.abstractmethod
    def tool(self) -> TraditionalCodingToolBase:
        pass

    def encode(self, input_pth, output_pth, quality):
        print("Encoding image:", input_pth, "with quality", quality)
        img = Image.open(input_pth)
        img = np.asarray(img)
        img_compressed = self.tool.compress_block(img, q_scale=quality)
        with open(output_pth, "wb") as f:
            f.write(img_compressed)

    def _decode(self, input_pth, output_pth):
        print("Decoding image:", input_pth)
        with open(input_pth, "rb") as f:
            bstr = f.read()
        rec = self.tool.decompress_block(bstr, None, None)
        rec_img = Image.fromarray(rec)
        rec_img.save(output_pth)


@register_anchor("WebP")
class WebP(ToolBasedCodec):
    @property
    def tool(self):
        return WebPTool()


@register_anchor("JPEG")
class JPEG(ToolBasedCodec):
    @property
    def tool(self):
        return JPEGTool()


@register_anchor("VTM")
class VTM(ToolBasedCodec):
    @property
    def tool(self):
        return VTMTool()


class PatchedToolBasedCodec(CodecBase, abc.ABC):
    def __init__(self, ctu_size=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctu_size = ctu_size

    @property
    @abc.abstractmethod
    def tool(self) -> LICToolBase:
        pass

    def encode(self, input_pth, output_pth, quality):
        print("Encoding image:", input_pth, "with quality", quality)
        img = Image.open(input_pth)
        img = np.asarray(img)
        h, w = img.shape[:2]
        fileio = FileIO(h, w, self.ctu_size, mosaic=False)
        img_blocks = divide_blocks(fileio, h, w, img, torch.float32)

        n_blocks = len(img_blocks)
        method_ids = np.zeros([n_blocks], dtype=np.int32)
        fileio.method_id = method_ids

        bitstreams = []
        for i, blk in enumerate(img_blocks):
            bitstream = self.tool.compress_block(blk.cuda, quality)
            bitstreams.append(bitstream)

        fileio.bitstreams = bitstreams
        fileio.dump(output_pth)

    def _decode(self, input_pth: str, output_pth: str) -> None:
        print("Decoding image:", input_pth)
        fileio = FileIO.load(input_pth, mosaic=False, ctu_size=self.ctu_size)
        img_blocks = []
        for i in range(len(fileio.bitstreams)):
            blk = self.tool.decompress_block(fileio.bitstreams[i], fileio.h, fileio.w)
            blk = torch_float_to_np_uint8(blk)
            img_blocks.append(blk)

        rec = join_blocks(img_blocks, fileio)
        rec_img = Image.fromarray(rec)
        rec_img.save(output_pth)


@register_anchor("TCM")
class TCM(PatchedToolBasedCodec):
    def __init__(self, ctu_size=512, *args, **kwargs):
        super().__init__(ctu_size, *args, **kwargs)
        self._tool = TOOL_GROUPS["TCM"]("TCM_VBR2_ALL", torch.float32, self.ctu_size)

    def tool(self):
        return self.tool


@register_anchor("EVC")
class EVC(PatchedToolBasedCodec):
    def __init__(self, ctu_size=512, *args, **kwargs):
        super().__init__(ctu_size, *args, **kwargs)
        self._tool = TOOL_GROUPS["EVC"]("EVC_LL", torch.float32, self.ctu_size)

    def tool(self):
        return self.tool


@register_anchor("MLICPP")
class MLICPP(PatchedToolBasedCodec):
    def __init__(self, ctu_size=512, *args, **kwargs):
        super().__init__(ctu_size, *args, **kwargs)
        self._tool = TOOL_GROUPS["MLICPP"]("MLICPP_ALL", torch.float32, self.ctu_size)

    def tool(self):
        return self.tool


@register_anchor("QARV")
class QARV(PatchedToolBasedCodec):
    def __init__(self, ctu_size=512, *args, **kwargs):
        super().__init__(ctu_size, *args, **kwargs)
        self._tool = TOOL_GROUPS["QARV"]("QARV", torch.float32, self.ctu_size)

    def tool(self):
        return self.tool
