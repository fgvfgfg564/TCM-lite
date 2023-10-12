"""
HEADER format:

IMAGE_H     H   1
IMAGE_W     H   1

for all CTU:
    METHOD_ID   B   2
    NUM_BYTES   I   4
    Q_SCALE     f   4

total: (2 + 10*num_ctu) bytes
"""

import struct
from typing import BinaryIO
import numpy as np
import io

class FileIO:
    meta_str = 'HH'
    ctu_str = 'BIf'

    def __init__(self, h, w, ctu_size, method_id=None, q_scale=None, bitstreams=None) -> None:
        self.h = h
        self.w = w
        self.ctu_size = ctu_size
        self.ctu_h = (h - 1) // ctu_size + 1
        self.ctu_w = (w - 1) // ctu_size + 1
        self.num_ctu = self.ctu_h * self.ctu_w

        self.format_str = [self.meta_str]
        for i in range(self.num_ctu):
            self.format_str.append(self.ctu_str)
        self.format_str = "".join(self.format_str)

        self.method_id = method_id
        self.bitstreams = bitstreams
        self.q_scale = q_scale
    
    @property
    def header_size(self):
        return struct.calcsize(self.format_str)
    
    def dump(self, filename: str):
        with open(filename, "wb") as fd:
            items = [self.h, self.w]
            for i in range(self.ctu_h):
                for j in range(self.ctu_w):
                    items.append(self.method_id[i, j])
                    items.append(len(self.bitstreams[i][j]))
                    items.append(self.q_scale[i, j])

            bits = struct.pack(self.format_str, *items)
            fd.write(bits)

            for i in range(self.ctu_h):
                for j in range(self.ctu_w):
                    fd.write(self.bitstreams[i][j])
    
    @classmethod
    def _read_with_format(cls, format_str, fd: BinaryIO):
        l = struct.calcsize(format_str)
        s = fd.read(l)
        return struct.unpack(format_str, s)
    
    @classmethod
    def load(cls, source: str, ctu_size: int):
        if isinstance(source, bytes):
            fd = io.BytesIO(source)
        else:
            fd = open(source, "rb")
        
        h, w = cls._read_with_format(cls.meta_str, fd)
        file_io = cls(h, w, ctu_size)
        
        file_io.method_id = np.zeros([file_io.ctu_h, file_io.ctu_w], dtype=np.uint8)
        file_io.q_scale = np.zeros([file_io.ctu_h, file_io.ctu_w], dtype=np.float32)
        num_bytes = np.zeros([file_io.ctu_h, file_io.ctu_w], dtype=np.uint32)

        # read CTU header
        for i in range(file_io.ctu_h):
            for j in range(file_io.ctu_w):
                _method_id, _num_bytes, _q_scale = cls._read_with_format(cls.ctu_str, fd)
                file_io.method_id[i, j] = _method_id
                file_io.q_scale[i, j] = _q_scale
                num_bytes[i, j] = _num_bytes

        # read CTU bytes
        file_io.bitstreams = []
        for i in range(file_io.ctu_h):
            bitstream_tmp = []
            for j in range(file_io.ctu_w):
                bits_ctu = fd.read(num_bytes[i, j])
                bitstream_tmp.append(bits_ctu)
            file_io.bitstreams.append(bitstream_tmp)
        
        fd.close()
        return file_io