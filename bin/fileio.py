"""
HEADER format:

IMAGE_H     H   1
IMAGE_W     H   1

for all CTU:
    METHOD_ID   B   2
    NUM_BYTES   I   4
    Q_SCALE     f   4

total: (3 + 10*n_ctu) bytes
"""

import struct
from typing import BinaryIO
import numpy as np
import io


class FileIO:
    meta_str = "HH"
    ctu_str = "BIf"

    def __init__(
        self, h, w, ctu_size, mosaic, method_id=None, q_scale=None, bitstreams=None
    ) -> None:
        self.h = h
        self.w = w
        self.ctu_size = ctu_size
        self.mosaic = mosaic
        self.n_ctu = self.get_n_ctu()

        self.format_str = [self.meta_str]
        for i in range(self.n_ctu):
            self.format_str.append(self.ctu_str)
        self.format_str = "".join(self.format_str)

        self.method_id = method_id
        self.bitstreams = bitstreams
        self.num_bytes = None if bitstreams is None else np.array([len(bitstreams[i] for i in range(self.n_ctu))])
        self.q_scale = q_scale
    
    def get_n_ctu(self):
        ctu_h = (self.h - 1) // self.ctu_size + 1
        ctu_w = (self.w - 1) // self.ctu_size + 1
        n_ctu = ctu_h * ctu_w
        return n_ctu

    @property
    def header_size(self):
        return struct.calcsize(self.format_str)

    def dump(self, filename: str):
        with open(filename, "wb") as fd:
            items = [self.h, self.w]
            for i in range(self.n_ctu):
                items.append(self.method_id[i])
                items.append(len(self.bitstreams[i]))
                items.append(self.q_scale[i])

            bits = struct.pack(self.format_str, *items)
            fd.write(bits)

            for i in range(self.n_ctu):
                fd.write(self.bitstreams[i])

    @classmethod
    def _read_with_format(cls, format_str, fd: BinaryIO):
        l = struct.calcsize(format_str)
        s = fd.read(l)
        return struct.unpack(format_str, s)

    @classmethod
    def load(cls, source: str, mosaic, ctu_size):
        if isinstance(source, bytes):
            fd = io.BytesIO(source)
        else:
            fd = open(source, "rb")

        h, w = cls._read_with_format(cls.meta_str, fd)
        file_io = cls(h, w, mosaic=mosaic, ctu_size=ctu_size)

        file_io.method_id = np.zeros([file_io.n_ctu], dtype=np.uint8)
        file_io.q_scale = np.zeros([file_io.n_ctu], dtype=np.float32)
        num_bytes = np.zeros([file_io.n_ctu], dtype=np.uint32)

        # read CTU header
        for i in range(file_io.n_ctu):
            _method_id, _num_bytes, _q_scale = cls._read_with_format(
                cls.ctu_str, fd
            )
            file_io.method_id[i] = _method_id
            file_io.q_scale[i] = _q_scale
            num_bytes[i] = _num_bytes
        file_io.num_bytes = num_bytes

        # read CTU bytes
        file_io.bitstreams = []
        for i in range(file_io.n_ctu):
            bitstream_tmp = []
            bits_ctu = fd.read(num_bytes[i])
            bitstream_tmp.append(bits_ctu)
            file_io.bitstreams.append(bitstream_tmp)

        fd.close()
        return file_io
