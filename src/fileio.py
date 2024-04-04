"""
HEADER format:

IMAGE_H     H   2
IMAGE_W     H   2

for all CTU:
    METHOD_ID   B   2
    NUM_BYTES   I   4

"""

import struct
from typing import BinaryIO
import numpy as np
import io


class FileIO:
    meta_str = "HH"
    ctu_str = "BI"

    def __init__(
        self,
        h: int,
        w: int,
        ctu_size: int,
        mosaic: bool,
        method_id=None,
        bitstreams=None,
    ) -> None:
        self.h = h
        self.w = w
        self.num_pixels = h * w
        self.ctu_size = ctu_size
        self.mosaic = mosaic
        self._build_block_partition()

        self.format_str = [self.meta_str]
        for i in range(self.n_ctu):
            self.format_str.append(self.ctu_str)
        self.format_str = "".join(self.format_str)

        self.method_id = method_id
        self.bitstreams = bitstreams
        self.num_bytes = (
            None
            if bitstreams is None
            else np.array([len(bitstreams[i] for i in range(self.n_ctu))])
        )

        self._adjacencyMatrix = None
        self._adjacencyTable = None

    @staticmethod
    def intersects(bb1, bb2):
        # <upper, left, lower, right>
        d1 = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
        d2 = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
        return d1 >= 0 and d2 >= 0 and not (d1 == 0 and d2 == 0)

    @property
    def adjacencyMatrix(self):
        # The adjacency matrix of the blocks
        if self._adjacencyMatrix is None:
            N = self.n_ctu

            AM = np.zeros([N, N], dtype=np.bool_)

            for i in range(N):
                AM[i, i] = False
                for j in range(N):
                    AM[i, j] = AM[j, i] = self.intersects(
                        self.block_indexes[i], self.block_indexes[j]
                    )
            self._adjacencyMatrix = AM
        else:
            AM = self._adjacencyMatrix
        return AM

    @property
    def adjacencyTable(self):
        # The adjacency table of the blocks

        if self._adjacencyTable is None:
            N = self.n_ctu

            AT = []

            for i in range(N):
                at_item = []
                for j in range(N):
                    if i == j:
                        continue
                    if self.intersects(self.block_indexes[i], self.block_indexes[j]):
                        at_item.append(j)
                AT.append(at_item)
            self._adjacencyTable = AT
        else:
            AT = self._adjacencyTable

        return AT

    def _build_block_partition(self):
        if not self.mosaic:
            n_ctu_h, n_ctu_w = self._n_ctu_hw(self.h, self.w, self.ctu_size)
            n_ctu = n_ctu_h * n_ctu_w
        else:
            n_ctu_h, n_ctu_w = self._n_ctu_hw(self.h, self.w, self.ctu_size * 3)
            n_ctu = n_ctu_h * n_ctu_w * 5

        self.block_indexes = []
        block_num_pixels = []
        for i in range(n_ctu):
            upper, left, lower, right = self._block_bb(self.h, self.w, i)
            lower = min(lower, self.h)
            right = min(right, self.w)
            if upper < lower and left < right:
                self.block_indexes.append((upper, left, lower, right))
                block_num_pixels.append((lower - upper) * (right - left))

        self.n_ctu = len(self.block_indexes)
        self.block_num_pixels = np.array(block_num_pixels)

    @property
    def header_size(self):
        return struct.calcsize(self.format_str)

    def dump(self, filename: str):
        with open(filename, "wb") as fd:
            items = [self.h, self.w]
            for i in range(self.n_ctu):
                items.append(self.method_id[i])
                items.append(len(self.bitstreams[i]))

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
        num_bytes = np.zeros([file_io.n_ctu], dtype=np.uint32)

        # read CTU header
        for i in range(file_io.n_ctu):
            _method_id, _num_bytes = cls._read_with_format(cls.ctu_str, fd)
            file_io.method_id[i] = _method_id
            num_bytes[i] = _num_bytes
        file_io.num_bytes = num_bytes

        # read CTU bytes
        file_io.bitstreams = []
        for i in range(file_io.n_ctu):
            bits_ctu = fd.read(num_bytes[i])
            file_io.bitstreams.append(bits_ctu)

        fd.close()
        return file_io

    # Block Division

    def _n_ctu_hw(self, h, w, ctu_size):
        n_ctu_h = (h + ctu_size - 1) // ctu_size + 1
        n_ctu_w = (w + ctu_size - 1) // ctu_size + 1

        return n_ctu_h, n_ctu_w

    def _block_id_hw(self, h, w, block_id_mesh):
        # Only in mesh mod. Return the id of the image in height and width dimension
        n_ctu_h, n_ctu_w = self._n_ctu_hw(h, w, self.ctu_size)

        id_h = block_id_mesh // n_ctu_w
        id_w = block_id_mesh % n_ctu_w

        if id_h >= n_ctu_h:
            return None, None

        return id_h, id_w

    def _block_bb(self, h, w, block_id):
        if not self.mosaic:
            id_h, id_w = self._block_id_hw(h, w, block_id)

            if id_h is None:
                return None, None, None, None

            return (
                id_h * self.ctu_size,
                id_w * self.ctu_size,
                (id_h + 1) * self.ctu_size,
                (id_w + 1) * self.ctu_size,
            )
        else:
            metablk_id = block_id // 5
            miniblk_id = block_id % 5
            n_ctu_h = (h - 1) // (self.ctu_size * 3) + 1
            n_ctu_w = (w - 1) // (self.ctu_size * 3) + 1

            id_h = metablk_id // n_ctu_w
            id_w = metablk_id % n_ctu_w

            upper = id_h * self.ctu_size * 3
            left = id_w * self.ctu_size * 3
            lower = upper
            right = left

            bias = [
                (0, 0, 2, 1),
                (0, 1, 1, 3),
                (1, 1, 2, 2),
                (2, 0, 3, 2),
                (1, 2, 3, 3),
            ]

            upper += bias[miniblk_id][0] * self.ctu_size
            left += bias[miniblk_id][1] * self.ctu_size
            lower += bias[miniblk_id][2] * self.ctu_size
            right += bias[miniblk_id][3] * self.ctu_size

            lower = min(lower, h)
            right = min(right, w)

            return upper, left, lower, right
