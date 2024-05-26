import enum
import statistics
from typing import List, Type

import os
import random
import time
import copy
import abc

from dataclasses import dataclass
from unittest import result
from sympy import rational_interpolate
from typing_extensions import deprecated

import numpy as np
from PIL import Image
import torch
import tqdm

from coding_tools.coding_tool import CodingToolBase
import coding_tools.utils.timer as timer
from coding_tools.traditional_tools import WebPTool
from coding_tools.baseline import CodecBase
from coding_tools import TOOL_GROUPS
from ..loss import LOSSES, LossBase

from ..type import *
from ..utils import *
from ..fileio import FileIO
from ..math_utils import *
from ..async_ops import async_map
from .sa_solver import SolverBase, LagrangeMultiplierSolver
from .toucher import Toucher


Image.MAX_IMAGE_PIXELS = None  # Don't detect decompression bombs

SAFETY_BYTE_PER_CTU = 0

np.seterr(all="raise", under="warn")


@dataclass
class ImageBlock:
    np: np.ndarray  # 0 - 255
    cuda: torch.Tensor  # 0 - 1


class Solution:
    def __init__(self, method_ids: np.ndarray, target_byteses: np.ndarray) -> None:
        self.method_ids = method_ids
        self.target_byteses = target_byteses
        self.n_ctu = len(self.method_ids)


class GASolution(Solution):
    def __init__(self, method_score: np.ndarray, target_byteses: np.ndarray) -> None:
        self.method_score = method_score  # ctu, n_method
        self.method_ids = np.argmax(method_score, axis=0)
        self.target_byteses = target_byteses
        self.n_ctu = len(self.method_ids)

    def normalize_target_byteses(self, min_table, max_table, b_t):
        min_bytes = np.zeros_like(self.target_byteses, dtype=np.float32)
        max_bytes = np.zeros_like(self.target_byteses, dtype=np.float32)

        for i in range(self.n_ctu):
            min_bytes[i] = min_table[self.method_ids[i]][i]
            max_bytes[i] = max_table[self.method_ids[i]][i]

        self.target_byteses = normalize_to_target(
            self.target_byteses, min_bytes, max_bytes, b_t
        )


class EngineBase(CodecBase):
    CACHE_DIR = os.path.join(os.path.split(__file__)[0], "../../cache")

    def __init__(
        self,
        ctu_size,
        mosaic,
        num_qscale_samples=20,
        tool_groups=TOOL_GROUPS.keys(),
        tool_filter=None,
        dtype=torch.float32,
        fitterclass: Type[Fitter] = FitKExp,
    ) -> None:
        self.ctu_size = ctu_size
        self.mosaic = mosaic
        self.methods = []
        self.fitterclass = fitterclass

        self.dtype = dtype

        self._load_models(tool_groups, tool_filter)

        self.num_qscale_samples = num_qscale_samples
        self.qscale_samples = np.linspace(0, 1, num_qscale_samples, dtype=np.float32)[
            ::-1
        ]

    def _load_models(self, valid_groups, tool_filter):
        idx = 0
        # Load models
        for group_name in valid_groups:
            engine_cls = TOOL_GROUPS[group_name]
            print("Loading tool group:", group_name)
            if engine_cls.MODELS is None:
                print("Loading model:", group_name)

                method = engine_cls()

                if tool_filter is None:
                    cnt = 1
                else:
                    cnt = tool_filter.count(group_name)

                for _ in range(cnt):
                    self.methods.append(
                        (
                            method,
                            group_name,
                            idx,
                        )
                    )
                    idx += 1
            else:
                for model_name in engine_cls.MODELS.keys():
                    if not tool_filter or model_name in tool_filter:
                        print("Loading model:", model_name)

                        method = engine_cls.from_model_name(
                            model_name, self.dtype, self.ctu_size
                        )

                        if tool_filter is None:
                            cnt = 1
                        else:
                            cnt = tool_filter.count(model_name)

                        for _ in range(cnt):
                            self.methods.append(
                                (
                                    method,
                                    model_name,
                                    idx,
                                )
                            )
                            idx += 1

        if len(self.methods) == 0:
            raise RuntimeError("No valid coding tool is loaded!")

        self.n_method = len(self.methods)

    def _feed_block(self, method: CodingToolBase, image_block: ImageBlock, qscale):
        if method.PLATFORM == "numpy":
            return method.compress_block(image_block.np, qscale)
        else:
            return method.compress_block(image_block.cuda, qscale)

    def _compress_with_target(self, method, image_block: ImageBlock, target_bytes):
        min_qs = float(1e-5)
        max_qs = float(1.0)
        while min_qs < max_qs - 1e-6:
            mid_qs = (max_qs + min_qs) / 2.0
            bitstream = self._feed_block(method, image_block, mid_qs)
            len_bytes = len(bitstream)
            if len_bytes <= target_bytes:
                max_qs = mid_qs
            else:
                min_qs = mid_qs

        bitstream = self._feed_block(method, image_block, max_qs)
        return bitstream, max_qs

    def _search_init_qscale(self, method, img_blocks, b_t):
        # Given method, search a shared qscale for all blocks and return target bytes for each block
        min_qs = float(1e-5)
        max_qs = float(1.0)
        n_ctu = len(img_blocks)
        target_bytes = np.zeros([n_ctu])

        while min_qs < max_qs - 1e-3:
            mid_qs = (max_qs + min_qs) / 2.0
            total_bytes = 0
            for i in range(n_ctu):
                bitstream = self._feed_block(method, img_blocks[i], mid_qs)
                len_bytes = len(bitstream)
                target_bytes[i] = len_bytes
                total_bytes += len_bytes
            if total_bytes <= b_t:
                max_qs = mid_qs
            else:
                min_qs = mid_qs

        return max_qs, target_bytes

    def _try_compress_decompress(
        self,
        method: CodingToolBase,
        image_block: ImageBlock,
        loss,
        target_bytes=None,
        q_scale=None,
        repeat=1,
    ):
        losscls = LOSSES[loss]
        times = []
        _, c, h, w = image_block.cuda.shape
        ctu_loss = float("inf")
        bitstream = b""
        assert repeat >= 1
        for i in range(repeat):
            if q_scale is None:
                bitstream, q_scale = self._compress_with_target(
                    method, image_block, target_bytes
                )
            else:
                bitstream = self._feed_block(method, image_block, q_scale)
            time0 = time.time()
            recon_img = method.decompress_block(bitstream, h, w)

            torch.cuda.synchronize()
            times.append(time.time() - time0)

            if method.PLATFORM == "torch":
                recon_img = torch_pseudo_quantize_to_uint8(recon_img)
            ctu_loss = losscls.ctu_level_loss(image_block, recon_img)

        return ctu_loss, np.mean(times), bitstream

    def _precompute_score(self, img_blocks, img_size, img_hash, loss: str):
        assert loss in LOSSES
        print(f"Self.methods: {list([x[1] for x in self.methods])}")
        h, w = img_size
        n_ctu = len(img_blocks)
        n_methods = len(self.methods)

        self._precomputed_curve: ImgCurves = {}
        self._minimal_bytes = np.zeros([n_methods, n_ctu], dtype=np.float32)
        self._maximal_bytes = np.zeros([n_methods, n_ctu], dtype=np.float32)

        pbar = tqdm.trange(n_methods * n_ctu * len(self.qscale_samples))
        pbar.set_description("Precomputing score")
        pbar_iter = pbar.__iter__()
        for method, method_name, method_idx in self.methods:
            self._precomputed_curve[method_idx] = {}
            for i in range(n_ctu):
                cache_dir = os.path.join(
                    self.CACHE_DIR,
                    loss,
                    method_name,
                    img_hash,
                    "-".join(
                        [
                            str(self.ctu_size),
                            str(self.mosaic),
                            str(self.num_qscale_samples),
                            str(i),
                        ]
                    ),
                )
                b_e_file = os.path.join(cache_dir, "b_e.npz")
                b_t_file = os.path.join(cache_dir, "b_t.npy")
                min_max_file = os.path.join(cache_dir, "min_max.npz")

                try:
                    # If the caches are complete, load them
                    b_e = self.fitterclass.load(b_e_file)
                    b_t = np.load(b_t_file)
                    min_max = np.load(min_max_file)
                    min_t = min_max["min_t"]
                    max_t = min_max["max_t"]
                    print("Loaded cache from:", cache_dir, flush=True)

                    for qscale in self.qscale_samples:
                        pbar_iter.__next__()
                except FileNotFoundError:
                    # No cache or cache is broken
                    ctu_losses = []
                    num_bytes = []
                    qscales = []
                    times = []

                    for qscale in self.qscale_samples:
                        image_block = img_blocks[i]

                        (
                            ctu_loss,
                            dec_time,
                            bitstream,
                        ) = self._try_compress_decompress(
                            method,
                            image_block,
                            target_bytes=None,
                            q_scale=qscale,
                            repeat=1,
                            loss=loss,
                        )
                        num_byte = len(bitstream)
                        num_bytes.append(num_byte)
                        ctu_losses.append(ctu_loss)
                        times.append(dec_time)
                        qscales.append(qscale)
                        pbar_iter.__next__()

                    make_strictly_increasing(num_bytes)

                    if len(num_bytes) <= 3 or not is_strictly_increasing(num_bytes):
                        raise ValueError(
                            f"num_bytes are too few or not strictly increasing: \nnum_bytes={num_bytes}\nsqes={ctu_losses}\nqscales={qscales}"
                        )

                    print(f"bytes={num_bytes}\ndistortion={ctu_losses}")

                    b_e = self.fitterclass(num_bytes, ctu_losses)
                    # b_t = interpolate.interp1d(num_bytes, times, kind='linear')
                    b_t = np.polyfit(num_bytes, times, 1)

                    min_t = min(num_bytes)
                    max_t = max(num_bytes)

                    # Save to cache
                    os.makedirs(cache_dir, exist_ok=True)
                    b_e.dump(b_e_file)
                    np.save(b_t_file, b_t)
                    np.savez(min_max_file, min_t=min_t, max_t=max_t)
                    print("Cache saved to:", cache_dir, flush=True)

                print(f"b_e[{i}] = {b_e.curve}")

                results: CTUCurves = {"b_e": b_e, "b_t": b_t}
                self._precomputed_curve[method_idx][i] = results
                self._minimal_bytes[method_idx][i] = min_t
                self._maximal_bytes[method_idx][i] = max_t

    def _compress_blocks(self, img_blocks, solution):
        # Calculate q_scale and bitstream for CTUs
        n_ctu = len(img_blocks)
        bitstreams = []
        q_scales = np.zeros([n_ctu], dtype=np.float32)
        for i in range(n_ctu):
            method, method_name, method_id = self.methods[solution.method_ids[i]]
            target = solution.target_byteses[i]
            print(
                f"Encoding CTU[{i}]: method #{method_id}('{method_name}'); Target = {target} B"
            )
            bitstream, q_scale = self._compress_with_target(
                method, img_blocks[i], target
            )
            q_scales[i] = q_scale
            bitstreams.append(bitstream)

        return solution.method_ids, q_scales, bitstreams

    @abc.abstractmethod
    def _solve(
        self,
        img_blocks,
        img_size,
        r_limit,
        t_limit,
        file_io,
        losstype: str,
        **kwargs,
    ) -> Tuple[FileIO, Any]:
        pass

    def read_img(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        rgb_np = Image.open(img_path).convert("RGB")
        rgb_np = np.asarray(rgb_np).copy()
        img_hash = hash_numpy_array(rgb_np)
        return rgb_np, img_hash

    def divide_blocks(self, fileio: FileIO, h, w, img) -> List[ImageBlock]:
        blocks = []
        for i in range(fileio.n_ctu):
            upper, left, lower, right = fileio.block_indexes[i]
            print(f"Block #{i}: ({upper}, {left}) ~ ({lower}, {right})")

            # Move to CUDA
            img_patch_np = img[upper:lower, left:right, :]
            img_patch_cuda = (
                torch.from_numpy(img_patch_np).permute(2, 0, 1).type(self.dtype) / 255.0
            )
            img_patch_cuda = img_patch_cuda.unsqueeze(0)
            img_patch_cuda = img_patch_cuda.cuda()

            blocks.append(ImageBlock(img_patch_np, img_patch_cuda))

        return blocks

    def decode_ctu(self, file_io, i, bounds) -> np.ndarray:
        (upper, left, lower, right) = bounds
        method_id = file_io.method_id[i]
        bitstream = file_io.bitstreams[i]
        method, method_name, _ = self.methods[method_id]
        torch.cuda.synchronize()
        ctu = method.decompress_block(bitstream, lower - upper, right - left)
        if isinstance(ctu, torch.Tensor):
            ctu = torch_float_to_np_uint8(ctu)
        return ctu

    def _estimate_decode_time(self, file_io: FileIO):
        total_time = 0
        for i, bitstream in enumerate(file_io.bitstreams):
            num_bytes = len(bitstream)
            dec_est = np.polyval(
                self._precomputed_curve[file_io.method_id[i]][i]["b_t"], num_bytes
            )
            total_time += dec_est
        return total_time

    def _self_check(self, img_blocks: List[ImageBlock], file_io: FileIO, losstype: str):
        losscls = LOSSES[losstype]
        ctu_losses = []
        # Check the difference between estimated and actual D loss
        for i, bounds in enumerate(file_io.block_indexes):
            ctu = self.decode_ctu(file_io, i, bounds)
            ctu_loss = losscls.ctu_level_loss(img_blocks[i], ctu)
            num_bytes = len(file_io.bitstreams[i])
            sqe_est = self._precomputed_curve[file_io.method_id[i]][i]["b_e"](num_bytes)
            print(
                f"Self check CTU #{i}: Method={file_io.method_id[i]}; Estimated CTU loss: {sqe_est:.6f}; actual CTU loss: {ctu_loss:.6f}"
            )
            ctu_losses.append(ctu_loss)
        estimated_loss = losscls.global_level_loss(file_io, ctu_losses)
        print(
            f"Full self check: global loss={estimated_loss:.6f}. This should be precise."
        )
        return estimated_loss

    def _encode(
        self,
        input_pth,
        target_bpp,
        target_time,
        losstype: str,
        **kwargs,
    ) -> Tuple[FileIO, Any]:
        input_img, img_hash = self.read_img(input_pth)
        h, w, c = input_img.shape
        img_size = (h, w)

        file_io = FileIO(h, w, self.ctu_size, self.mosaic)

        r_limit = target_bpp * h * w // 8
        print(f"Image shape: {h}x{w}")
        print(f"Target={r_limit}B; Target bpp={target_bpp:.4f};")
        print(f"Time limit={target_time}s")

        header_bytes = file_io.header_size
        safety_bytes = SAFETY_BYTE_PER_CTU * file_io.n_ctu
        r_limit -= header_bytes + safety_bytes
        print(
            f"Header bytes={header_bytes}; Safety_bytes={safety_bytes}; CTU bytes={r_limit}"
        )

        img_blocks = self.divide_blocks(file_io, h, w, input_img)

        print("Precompute all scorees", flush=True)
        self._precompute_score(img_blocks, img_size, img_hash, losstype)

        file_io, data = self._solve(
            img_blocks,
            img_size,
            r_limit=r_limit,
            t_limit=target_time,
            file_io=file_io,
            losstype=losstype,
            **kwargs,
        )

        self._self_check(img_blocks, file_io, losstype)

        return file_io, data

    @torch.inference_mode()
    def encode(
        self,
        input_pth,
        output_pth,
        target_bpp,
        target_time,
        losstype: str,
        **kwargs,
    ):
        file_io, data = self._encode(
            input_pth, target_bpp, target_time, losstype, **kwargs
        )
        file_io.dump(output_pth)
        return data

    def join_blocks(self, decoded_ctus, file_io: FileIO):
        h = file_io.h
        w = file_io.w
        recon_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i, ctu in enumerate(decoded_ctus):
            upper, left, lower, right = file_io.block_indexes[i]

            recon_img[upper:lower, left:right, :] = ctu[
                : lower - upper, : right - left, :
            ]
        return recon_img

    @torch.inference_mode()
    def _decode(self, input_pth, output_pth):
        """
        Decode into NumPy array with range 0-1.
        Return:
         - decoding time on CTUs
        """
        fd = open(input_pth, "rb")
        bitstream = fd.read()
        fd.close()

        # Decoding process; generate recon image
        file_io: FileIO = FileIO.load(bitstream, self.mosaic, self.ctu_size)
        with torch.no_grad():
            decoded_ctus = []
            for i, bounds in enumerate(file_io.block_indexes):
                print(f"Decoding CTU #{i}")
                ctu = self.decode_ctu(file_io, i, bounds)
                decoded_ctus.append(ctu)
            recon_img = self.join_blocks(decoded_ctus, file_io)
            out_img = dump_image(recon_img)
            Image.fromarray(out_img).save(output_pth)

    @torch.inference_mode()
    def accelerate(
        self,
        input_pth,
        output_pth,
        qscale,
        speedup,
        losstype: str,
        **kwargs,
    ):
        # Test acceleration with particular percent time
        # First, only enable the 0-th method
        input_img, img_hash = self.read_img(input_pth)
        h, w, c = input_img.shape
        img_size = (h, w)
        fullspeed_file_io = FileIO(h, w, self.ctu_size, self.mosaic)
        img_blocks = self.divide_blocks(fullspeed_file_io, h, w, input_img)

        print("Precompute score of method #0", flush=True)
        method_backup = self.methods
        self.methods = [method_backup[0]]
        self._precompute_score(img_blocks, img_size, img_hash, losstype)

        method, _, __ = self.methods[0]
        bitstreams = []
        method_ids = np.zeros([len(img_blocks)], dtype=np.int32)
        for i, block in enumerate(img_blocks):
            bitstream = self._feed_block(method, block, qscale)
            bitstreams.append(bitstream)
        fullspeed_file_io.method_id = method_ids
        fullspeed_file_io.bitstreams = bitstreams

        # Check the anchor bitstream
        self._self_check(img_blocks, fullspeed_file_io, losstype)

        # restore methods, calculate time budget and target bpp
        self.methods = method_backup
        fulltime = self._estimate_decode_time(fullspeed_file_io)
        num_bytes = len(fullspeed_file_io.dumps())
        target_bpp = num_bytes * 8 / h / w
        time_limit = fulltime / speedup
        print(
            f"Full decode time={fulltime}s. Accelerate {speedup}x. Time limit={time_limit}s"
        )
        file_io, data = self._encode(
            input_pth, target_bpp, time_limit, losstype, **kwargs
        )
        file_io.dump(output_pth)
        return data


class SAEngine1(EngineBase):
    def __init__(
        self,
        ctu_size,
        mosaic,
        num_qscale_samples=20,
        tool_groups=TOOL_GROUPS.keys(),
        tool_filter=None,
        dtype=torch.float32,
        solver: Type[SolverBase] = LagrangeMultiplierSolver,
    ) -> None:
        super().__init__(
            ctu_size, mosaic, num_qscale_samples, tool_groups, tool_filter, dtype
        )
        self.solver = solver()
        self.last_valid_step = None
        self.toucher = Toucher(WebPTool())

    # Older version that considers T loss

    W1 = 100
    W2 = 25
    Wa = 10.0

    def _select_distinctive_block(
        self,
        file_io: FileIO,
        last_ans,
        method=None,
        last_valid_step=None,
    ):
        n_ctu = file_io.n_ctu
        # Initialize change matrix
        P = np.ones([n_ctu, self.n_method], dtype=np.float32)
        if last_valid_step is not None:
            (last_changed_block, last_old_method, last_new_method) = last_valid_step
            # 1. If adjacent block of last move is in the same color, it's likely to be an improvement.
            for blk_id in file_io.adjacencyTable[last_changed_block]:
                if last_ans[blk_id] == last_old_method:
                    P[blk_id, last_new_method] = self.W1
            # 2. Non-adjacent likely to be an improvement.
            for blk_id in range(n_ctu):
                if last_ans[blk_id] == last_old_method:
                    P[blk_id, last_new_method] = np.maximum(
                        P[blk_id, last_new_method], self.W2
                    )

        # 3. Blocks are likely to change into the method of its adjacent ones.
        for blk_id in range(n_ctu):
            count_adjacent = np.zeros([self.n_method], dtype=np.float32)
            for adj_blk_id in file_io.adjacencyTable[blk_id]:
                count_adjacent[last_ans[adj_blk_id]] += 1
            count_adjacent /= count_adjacent.sum()
            for i in range(self.n_method):
                P[blk_id, i] = np.maximum(
                    P[blk_id, i], self.Wa * count_adjacent[i] ** 2
                )

        # Normalize and select
        P /= P.sum()
        select_list: List[Tuple[int, int]] = []
        P_list = []
        for blk_id in range(n_ctu):
            if method is not None:
                method_list = [method]
            else:
                method_list = range(self.n_method)
            for method_id in method_list:
                select_list.append((blk_id, method_id))
                p = P[blk_id, method_id]
                if last_ans[blk_id] == method_id:
                    p = 0
                P_list.append(p)

        selected = random.choices(select_list, weights=P_list, k=1)[0]
        return selected

    def _try_move(
        self, file_io: FileIO, last_ans: np.ndarray, n_method, adaptive_search
    ):
        # Generate a group of new method_ids that move from current state
        # 90% swap, 10% replace
        n_ctu = len(last_ans)
        # Try to update one block
        if adaptive_search:
            selected = self._select_distinctive_block(
                file_io,
                last_ans=last_ans,
                last_valid_step=self.last_valid_step,
            )
            print(
                f"Selected block: {selected[0]}; Old: {last_ans[selected[0]]}; New: {selected[1]}"
            )
            return selected
        else:
            # Pure random
            selected = np.random.random_integers(0, n_ctu - 1)
            new_method = np.random.random_integers(0, n_method - 1)
        return selected, new_method

    def _try_swap(self, file_io: FileIO, last_ans: np.ndarray):
        unique = np.unique(last_ans)
        m1, m2 = np.random.choice(unique, 2, replace=False)
        blk1, _ = self._select_distinctive_block(file_io, last_ans, method=m1)
        blk2, _ = self._select_distinctive_block(file_io, last_ans, method=m2)
        return blk1, blk2

    def _show_solution(self, method_ids, target_byteses, b_t, score, psnr, time):
        n_ctu = len(method_ids)
        total_bytes = np.sum(target_byteses)
        print(
            f"score={score}; total_bytes=[{total_bytes}/{b_t}]({100.0*total_bytes/b_t:.4f}%); PSNR={psnr:.3f}; time={time:.3f}s",
            flush=True,
        )
        for i in range(n_ctu):
            method_id = method_ids[i]
            target_bytes = target_byteses[i]

            curves = self._precomputed_curve[method_id][i]

            min_tb = self._minimal_bytes[method_ids[i]][i]
            max_tb = self._maximal_bytes[method_ids[i]][i]

            est_time = np.polyval(curves["b_t"], target_bytes)
            est_sqe = curves["b_e"](target_bytes)

            print(
                f"- CTU [{i}]:\tmethod_id={method_id}\ttarget_bytes={target_bytes:.1f}(in [{min_tb}, {max_tb}])\tdec_time={1000*est_time:.2f}ms\tsquared_error={est_sqe:.6f};",
                flush=True,
            )

    def _adaptive_init(
        self,
        img_blocks: List[ImageBlock],
        file_io: FileIO,
        n_ctu: int,
        n_method: int,
        r_limit,
        t_limit,
        losstype,
        num_steps=1000,
    ):
        """
        根据图像块的复杂度，自适应地初始化编码方法的选择。

        Args:
            img_blocks (List[ImageBlock]): 图像块列表。
            file_io (FileIO): 文件输入输出接口对象。
            n_ctu (int): CTU (Coding Tree Unit) 数量。
            n_method (int): 编码方法数量。
            r_limit (float): 压缩率的限制值。
            t_limit (float): 时间复杂度的限制值。
            losstype (str): 损失类型。
            num_steps (int, optional): 迭代次数。默认为1000。

        Returns:
            np.ndarray: 编码方法选择结果。

        """
        GRAN = min(100, len(img_blocks))
        # Assign blocks to methods according to their complexity
        print("Starting initialization ...")
        print("Estimating scores for method")

        complexity_scores = []

        for x in tqdm.tqdm(img_blocks, "Touching image blocks"):
            score = self.toucher.touch_complexity(x.np)
            complexity_scores.append(score)
        complexity_scores = np.asarray(complexity_scores)
        complexity_order = np.argsort(complexity_scores)
        print(f"Complexity order: {complexity_order}")

        def _est_speed(method_idx):
            speeds = []
            for i in range(n_ctu):
                ctu_speed = np.polyval(
                    self._precomputed_curve[method_idx][i]["b_t"], 1.0
                )
                speeds.append(ctu_speed)
            return np.asarray(speeds).mean()

        speeds = []
        for i in tqdm.tqdm(range(n_method), "Estimate speed for methods"):
            speeds.append(_est_speed(i))
        speeds = np.asarray(speeds)
        speeds_order = np.argsort(speeds)
        speeds_rank = np.empty_like(speeds_order)
        speeds_rank[speeds_order] = np.arange(n_method)

        # Hill-climbing on method ratios
        # ratio = softmax(w/20.0)
        # Initialize with #0 method
        w = np.zeros((n_method,), dtype=np.int32)
        w[speeds_rank[0]] = GRAN

        def generate_results(_w: np.ndarray):
            ratio = _w / GRAN
            ratio = np.cumsum(ratio)
            ratio[-1] = 1.0
            ratio = np.concatenate([[0.0], ratio])

            # calculate results
            results = np.zeros((n_ctu,), dtype=np.int32)
            for i in range(n_method):
                lb = int(round(n_ctu * ratio[i]))
                rb = int(round(n_ctu * ratio[i + 1]))
                results[complexity_order[lb:rb]] = speeds_order[i]
            return results

        def calc_loss(results):
            return self.solver.find_optimal_target_bytes(
                self._precomputed_curve,
                file_io,
                n_ctu,
                results,
                r_limit,
                t_limit,
                losstype,
            )[1]

        results = generate_results(w)
        loss = calc_loss(results)
        best_results = results.copy()
        best_loss = copy.deepcopy(loss)
        visited = dict()

        hashw = hash_numpy_array(w)
        visited[hashw] = loss

        T = 1.0

        for step in tqdm.tqdm(range(num_steps), "Calculate method ratio"):
            print(f"Weights={w}; Loss={loss}")

            in_idx = random.randrange(n_method)
            out_idx = random.randrange(n_method)
            while in_idx == out_idx or w[out_idx] == 0:
                in_idx = random.randrange(n_method)
                out_idx = random.randrange(n_method)

            w_new = w.copy()
            w_new[out_idx] -= 1
            w_new[in_idx] += 1
            hashnew = hash_numpy_array(w_new)
            result_new = generate_results(w_new)
            if hashnew not in visited:
                loss_new = calc_loss(result_new)
                visited[hashnew] = loss_new
            else:
                loss_new = visited[hashnew]

            if loss_new < loss:
                accept = True
            elif loss_new.r > 0 or loss_new.t > 0:
                accept = False
            else:
                delta = loss_new[2] - loss[2]
                p = safe_SA_prob(delta, T)
                accept = np.random.rand() < p

            print(f"T={T:.6f}; w_new={w_new}; loss={loss_new}; accept={accept}")

            if accept:
                loss = loss_new
                w = w_new
                results = result_new

            if loss < best_loss:
                best_loss = copy.deepcopy(loss)
                best_results = results.copy()

            T *= 0.99

        print(f"Initial method selection: {best_results}")
        return best_results

    def _init(
        self,
        img_blocks,
        init_values,
        adaptive_init,
        n_method,
        n_ctu,
        file_io,
        r_limit,
        t_limit,
        losstype,
    ):
        if n_method == 1:
            ans = np.zeros([n_ctu], dtype=np.int32)
        else:
            if init_values is not None:
                ans = init_values
            else:
                if adaptive_init:
                    ans = self._adaptive_init(
                        img_blocks, file_io, n_ctu, n_method, r_limit, t_limit, losstype
                    )
                else:
                    ans = np.random.random_integers(
                        0,
                        n_method - 1,
                        [
                            n_ctu,
                        ],
                    )
        return ans

    def _sa_body(
        self,
        ans,
        img_blocks,
        img_size,
        file_io,
        losstype,
        r_limit,
        t_limit,
        num_steps,
        adaptive_search,
        T_start,
        T_end,
    ):
        alpha = np.power(T_end / T_start, 1.0 / num_steps)
        n_ctu = len(img_blocks)
        n_method = len(self.methods)
        if adaptive_search:
            self.last_valid_step = None

        target_byteses, loss, psnr, t = self.solver.find_optimal_target_bytes(
            self._precomputed_curve, file_io, n_ctu, ans, r_limit, t_limit, losstype
        )

        T = T_start
        best_ans = ans
        best_loss = loss

        print("Initial ans: ", best_ans)
        print("Initial loss: ", best_loss)
        print(f"Start SA with {num_steps} steps")

        t0 = time.time()
        statistics = []

        # Simulated Annealing
        for step in range(num_steps):
            # 80%: swap blocks
            # 20% update a block
            p = np.random.rand()
            update = p < 0.2
            if np.all(ans == ans[0]):
                # All elements are equal
                update = True
            if best_loss[0] > 0 or best_loss[1] > 0:
                # Failed to meet constraints
                update = True
            if update:
                changed_block, new_method = self._try_move(
                    file_io, ans, n_method, adaptive_search
                )
                next_state = ans.copy()
                next_state[changed_block] = new_method
            else:
                id1, id2 = self._try_swap(file_io, ans)
                next_state = ans.copy()
                next_state[id1], next_state[id2] = next_state[id2], next_state[id1]

            (
                next_target_byteses,
                next_loss,
                next_psnr,
                next_time,
            ) = self.solver.find_optimal_target_bytes(
                self._precomputed_curve,
                file_io,
                n_ctu,
                next_state,
                r_limit,
                t_limit,
                losstype,
            )

            if next_loss < loss:
                accept = True
                if update:
                    self.last_valid_step = (
                        changed_block,
                        ans[changed_block],
                        new_method,
                    )
            else:
                if next_loss[0] == 0 and next_loss[1] == 0:
                    delta = next_loss[2] - loss[2]
                    p = safe_SA_prob(delta, T)
                    accept = np.random.rand() < p
                else:
                    accept = False
                if update:
                    self.last_valid_step = None

            print(f"Loss: {loss}; next_loss: {next_loss}; Accept: {accept}")

            statistics.append(
                {
                    "time": time.time() - t0,
                    "loss": loss.d,
                    "legal": bool(loss.r <= 0 and loss.t <= 0),
                }
            )

            if accept:
                ans = next_state
                target_byteses = next_target_byteses
                loss = next_loss
                psnr = next_psnr
                t = next_time
                if loss < best_loss:
                    best_loss = loss
                    best_ans = ans

            if step % (num_steps // 10) == 0:
                print(f"Results for step: {step}; T={T:.6f}; best_loss={best_loss}")
                self._show_solution(ans, target_byteses, r_limit, loss, psnr, t)

            T *= alpha
        return best_ans, statistics

    def _solve(
        self,
        img_blocks: List[ImageBlock],
        img_size,
        r_limit,
        t_limit,
        file_io: FileIO,
        losstype,
        num_steps=1000,
        T_start=10,
        T_end=1e-6,
        init_values: np.ndarray = None,
        adaptive_init=True,
        adaptive_search=True,
        **kwargs,
    ):
        # Technologies include:
        #
        n_ctu = len(img_blocks)
        n_method = len(self.methods)
        ans = self._init(
            img_blocks,
            init_values,
            adaptive_init,
            n_method,
            n_ctu,
            file_io,
            r_limit,
            t_limit,
            losstype,
        )
        if n_method > 1:
            ans, statistics = self._sa_body(
                ans,
                img_blocks,
                img_size,
                file_io,
                losstype,
                r_limit,
                t_limit,
                num_steps,
                adaptive_search,
                T_start,
                T_end,
            )
        else:
            statistics = None

        target_byteses, score, psnr, time = self.solver.find_optimal_target_bytes(
            self._precomputed_curve, file_io, n_ctu, ans, r_limit, t_limit, losstype
        )
        solution = Solution(ans, target_byteses)

        method_ids, q_scales, bitstreams = self._compress_blocks(img_blocks, solution)

        # update fileio
        file_io.method_id = method_ids
        file_io.bitstreams = bitstreams
        file_io.q_scales = q_scales

        return file_io, statistics


@deprecated("No longer in use")
class GAEngine1(EngineBase):
    def _initial_method_score(self, n_ctu, n_method):
        result = np.random.uniform(0.0, 1.0, [n_method, n_ctu])
        return result

    def _normal_noise_like(self, x, sigma):
        noise = np.random.normal(0, sigma, x.shape).astype(x.dtype)
        return noise

    @classmethod
    def arithmetic_crossover(cls, a: np.ndarray, b: np.ndarray):
        assert a.shape == b.shape
        u = np.random.uniform(0, 1, a.shape)
        result = a * u + b * (1 - u)
        result.astype(a.dtype)
        return result

    def _hybrid(self, header1: GASolution, header2: GASolution, b_t, p):
        new_method_score = self.arithmetic_crossover(
            header1.method_score, header2.method_score
        )
        new_target = self.arithmetic_crossover(
            header1.target_byteses, header2.target_byteses
        )
        new_header = GASolution(new_method_score, new_target)
        new_header.normalize_target_byteses(
            self._minimal_bytes, self._maximal_bytes, b_t
        )

        return new_header

    def _mutate(
        self,
        header: GASolution,
        b_t,
        method_mutate_sigma,
        byte_mutate_sigma,
        inplace=True,
    ):
        if not inplace:
            header = copy.copy(header)

        # Mutate methods used; random swap adjacent methods
        score = header.method_score.copy()
        score_noise = self._normal_noise_like(score, method_mutate_sigma)
        new_score = score + score_noise
        new_score = np.clip(new_score, 0.0, 1.0)

        # Mutate target byterates
        old_target = header.target_byteses.copy()
        byterate_noise = self._normal_noise_like(old_target, byte_mutate_sigma)
        new_target = old_target + byterate_noise

        header.__init__(new_score, new_target)

        header.normalize_target_byteses(self._minimal_bytes, self._maximal_bytes, b_t)

        return header

    def _show_solution(self, solution: GASolution, b_t):
        total_bytes = np.sum(solution.target_byteses)
        print(
            f"score={solution.score}; total_bytes=[{total_bytes}/{b_t}]({100.0*total_bytes/b_t:.4f}%); PSNR={solution.psnr:.3f}; time={solution.time:.3f}s",
            flush=True,
        )
        for i in range(solution.n_ctu):
            method_id = solution.method_ids[i]
            target_bytes = solution.target_byteses[i]

            curves = self._precomputed_curve[method_id][i]

            min_tb = self._minimal_bytes[solution.method_ids[i]][i]
            max_tb = self._maximal_bytes[solution.method_ids[i]][i]

            est_time = np.polyval(curves["b_t"], target_bytes)
            est_sqe = curves["b_e"](target_bytes)

            print(
                f"- CTU [{i}]:\tmethod_id={method_id}\ttarget_bytes={target_bytes:.1f}(in [{min_tb}, {max_tb}])\tdec_time={1000*est_time:.2f}ms\tsquared_error={est_sqe:.6f}; method_scores={solution.method_score[:, i]}"
            )

    def _solve(
        self,
        img_blocks: List,
        img_size,
        b_t,
        file_io,
        N,
        num_gen,
        boltzmann_k,
        no_allocation,
        method_sigma,
        bytes_sigma,
    ):
        # statistics
        gen_score = []
        gen_psnr = []
        gen_time = []

        n_ctu = len(img_blocks)
        n_method = len(self.methods)
        h, w = img_size
        num_pixels = h * w

        DEFAULT_METHOD = 0

        print("Initializing qscale")
        default_qscale, default_target_bytes = self._search_init_qscale(
            self.methods[DEFAULT_METHOD][0], img_blocks, b_t
        )

        if no_allocation:
            method_scores = self._initial_method_score(n_ctu, n_method)
            solution = GASolution(method_scores, default_target_bytes)
            return self._compress_blocks(img_blocks, solution)

        # Generate initial solutions
        solutions = []
        for k in (pbar := tqdm.tqdm(range(N))):
            pbar.set_description(f"Generate initial solutions")
            method_scores = self._initial_method_score(n_ctu, n_method)
            target_byteses = default_target_bytes.copy()
            method = GASolution(method_scores, target_byteses)
            self._mutate(
                method,
                b_t,
                method_sigma,
                bytes_sigma,
            )
            method.normalize_target_byteses(
                self._minimal_bytes, self._maximal_bytes, b_t
            )
            method.score, method.psnr, method.time = self._get_score(
                n_ctu,
                file_io,
                method.method_ids,
                method.target_byteses,
                b_t,
            )
            solutions.append(method)

        solutions.sort(key=lambda x: x.score, reverse=True)

        max_score = solutions[0].score
        best_psnr = solutions[0].psnr
        best_time = solutions[0].time
        initial_score = abs(max_score)

        for k in range(num_gen):
            # show best solution on generation
            best_solution: GASolution = solutions[0]
            gen_psnr.append(best_psnr)
            gen_score.append(max_score)
            gen_time.append(best_time)

            print(f"Best Solution before Gen.{k}:")
            self._show_solution(best_solution, b_t)
            print(flush=True)

            G = num_gen
            R = (G + 2 * np.sqrt(k)) / (3 * G)
            T = 0.75 * (1 - (k / num_gen)) ** 2 + 0.25

            # Elitism, but only keep the best alive
            num_alive = 1
            num_breed = N

            # Elitism strategy
            breedable = solutions[:num_breed]
            solutions = solutions[:num_alive]

            # Calculate probability; Boltzmann selection
            probs = []
            for i in range(num_breed):
                beta = T / boltzmann_k
                t = breedable[i].score * beta / initial_score
                probs.append(t)
            probs = np.array(probs, dtype=np.float32)
            probs = stable_softmax(probs)

            print("Probability for top 10:", probs[:10], flush=True)

            # Hybrid
            idxes = np.arange(num_breed)
            for i in (pbar := tqdm.tqdm(range(N - num_alive))):
                pbar.set_description(
                    f"Calculating score for generation {k+1}; max_score={max_score:.3f}; best_psnr={best_psnr:.3f}; best_time={best_time:.3f}"
                )
                parent_id1 = np.random.choice(idxes, p=probs)
                parent_id2 = np.random.choice(idxes, p=probs)
                newborn = self._hybrid(
                    breedable[parent_id1],
                    breedable[parent_id2],
                    b_t,
                    T * 0.5,
                )
                self._mutate(
                    newborn,
                    b_t,
                    T * method_sigma,
                    T * bytes_sigma,
                )
                newborn.score, newborn.psnr, newborn.time = self._get_score(
                    n_ctu,
                    file_io,
                    newborn.method_ids,
                    newborn.target_byteses,
                    b_t,
                )
                solutions.append(newborn)
                if max_score is None or max_score < newborn.score:
                    max_score = newborn.score
                    best_time = newborn.time
                    best_psnr = newborn.psnr

            solutions.sort(key=lambda x: x.score, reverse=True)

        data = {
            "gen_score": gen_score,
            "gen_psnr": gen_psnr,
            "gen_time": gen_time,
        }

        method_ids, q_scales, bitstreams = self._compress_blocks(
            img_blocks, solutions[0]
        )
        return method_ids, q_scales, bitstreams, data
