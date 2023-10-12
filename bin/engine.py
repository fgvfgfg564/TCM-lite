import os
import numpy as np
import time
from PIL import Image
import torch
import torch.nn.functional as F

import tqdm
import copy
import math
import random
from scipy import interpolate
import einops

from coding_tools.EVC.bin.engine import ModelEngine as EVCModelEngine
from coding_tools.TCM.app.engine import ModelEngine as TCMModelEngine
import coding_tools.utils.timer as timer

from .utils import get_bpg_result, is_strictly_increasing
from .fileio import FileIO

SAFETY_BYTE_PER_CTU = 2
W_TIME = 1

np.seterr(all="raise")


def get_padding_size(height, width, p=768):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


class Solution:
    def __init__(self, method_ids, target_byteses) -> None:
        self.method_ids = method_ids
        self.target_byteses = target_byteses
        self.n_ctu_h, self.n_ctu_w = self.method_ids.shape

    def normalize_target_byteses(self, min_table, max_table, total_target):
        min_bytes = np.zeros_like(self.target_byteses, dtype=np.int32)
        max_bytes = np.zeros_like(self.target_byteses, dtype=np.int32)

        for i in range(self.n_ctu_h):
            for j in range(self.n_ctu_w):
                min_bytes[i, j] = min_table[self.method_ids[i, j]][i][j]
                max_bytes[i, j] = max_table[self.method_ids[i, j]][i][j]

        old_target = self.target_byteses
        old_target -= min_bytes
        old_target = np.maximum(old_target, 1)
        old_target = old_target.astype(np.float32)
        old_target_rate = old_target / np.sum(old_target)

        total_target = np.maximum(total_target, np.sum(min_bytes))

        new_target = old_target_rate * (total_target - np.sum(min_bytes))
        new_target += min_bytes
        new_target = np.minimum(new_target, max_bytes)
        new_target = np.floor(new_target).astype(np.int32)
        self.target_byteses = new_target


class Engine:
    TOOL_GROUPS = {
        "EVC": EVCModelEngine,
        "TCM": TCMModelEngine,
    }

    def __init__(
        self, ctu_size=512, num_qscale_samples=50, tool_groups=TOOL_GROUPS.keys()
    ) -> None:
        self.ctu_size = ctu_size
        self.methods = []

        self._load_models(tool_groups)

        self.num_qscale_samples = num_qscale_samples
        self.qscale_samples = np.linspace(0, 1, num_qscale_samples, dtype=np.float32)[
            ::-1
        ]

    def _load_models(self, valid_groups):
        idx = 0
        # Load models
        for group_name in valid_groups:
            engine_cls = self.TOOL_GROUPS[group_name]
            print("Loading tool group:", group_name)
            for model_name in engine_cls.MODELS.keys():
                print("Loading model:", model_name)
                self.methods.append(
                    (engine_cls.from_model_name(model_name), model_name, idx)
                )
                idx += 1

    def _compress_with_target(self, method, image_block, target_bytes):
        min_qs = float(1e-5)
        max_qs = float(1.0)
        while min_qs < max_qs - 1e-3:
            mid_qs = (max_qs + min_qs) / 2.0
            bitstream = method.compress_block(image_block, mid_qs)
            len_bytes = len(bitstream)
            if len_bytes <= target_bytes:
                max_qs = mid_qs
            else:
                min_qs = mid_qs

        bitstream = method.compress_block(image_block, max_qs)
        return bitstream, max_qs

    @classmethod
    def torch_to_uint8(cls, x):
        x = torch.clamp(x, 0, 1)
        x *= 255
        x = torch.round(x)
        x = x.to(torch.uint8)
        return x

    @classmethod
    def torch_pseudo_quantize_to_uint8(cls, x):
        x = cls.torch_to_uint8(x)
        x = x.to(torch.half) / 255.0
        return x

    def _estimate_score(
        self, method, image_block, target_bytes=None, q_scale=None, repeat=1
    ):
        times = []
        sqes = []
        _, c, h, w = image_block.shape
        for i in range(repeat):
            if q_scale is None:
                bitstream, q_scale = self._compress_with_target(
                    method, image_block, target_bytes
                )
            else:
                bitstream = method.compress_block(image_block, q_scale)
            time0 = time.time()
            recon_img = method.decompress_block(bitstream, h, w, q_scale)
            recon_img = torch.clamp(recon_img, 0, 1)

            torch.cuda.synchronize()
            times.append(time.time() - time0)

            image_block = self.torch_pseudo_quantize_to_uint8(image_block)
            recon_img = self.torch_pseudo_quantize_to_uint8(recon_img)

            sqe = torch.sum((image_block - recon_img) ** 2).detach().cpu().numpy()
            sqes.append(sqe)

        sqe = np.mean(sqes)
        psnr = -10 * np.log10(sqe)

        return sqe, np.mean(times), bitstream, recon_img, q_scale

    def _precompute_score(self, img_blocks):
        n_ctu_h, n_ctu_w, _, c, ctu_h, ctu_w = img_blocks.shape
        n_methods = len(self.methods)

        self._precomputed_curve = {}
        self._minimal_bytes = np.zeros([n_methods, n_ctu_h, n_ctu_w], dtype=np.int32)
        self._maximal_bytes = np.zeros([n_methods, n_ctu_h, n_ctu_w], dtype=np.int32)

        pbar = tqdm.trange(n_methods * n_ctu_h * n_ctu_w * len(self.qscale_samples))
        pbar.set_description("Precomputing score")
        pbar_iter = pbar.__iter__()
        for method, _, idx in self.methods:
            self._precomputed_curve[idx] = {}
            for i in range(n_ctu_h):
                self._precomputed_curve[idx][i] = {}
                for j in range(n_ctu_w):
                    self._precomputed_curve[idx][i][j] = {}
                    sqes = []
                    num_bytes = []
                    qscales = []
                    times = []
                    for qscale in self.qscale_samples:
                        image_block = img_blocks[i, j]

                        sqe, dec_time, bitstream, __, ___ = self._estimate_score(
                            method, image_block, None, qscale, 1
                        )
                        num_byte = len(bitstream)
                        while len(num_bytes) > 0 and num_byte <= num_bytes[-1]:
                            num_bytes.pop()
                            sqes.pop()
                            times.pop()
                            qscales.pop()
                        num_bytes.append(num_byte)
                        sqes.append(sqe)
                        times.append(dec_time)
                        qscales.append(qscale)
                        pbar_iter.__next__()

                    if not is_strictly_increasing(num_bytes):
                        raise ValueError(
                            f"num_bytes are not strictly increasing: \nnum_bytes={num_bytes}\nsqes={sqes}"
                        )

                    self._minimal_bytes[idx, i, j] = num_bytes[0]
                    self._maximal_bytes[idx, i, j] = num_bytes[-1]

                    try:
                        b_e = interpolate.interp1d(num_bytes, sqes, kind="linear")
                        # b_t = interpolate.interp1d(num_bytes, times, kind='linear')
                        b_t = np.polyfit(num_bytes, times, 1)
                        b_q = interpolate.interp1d(num_bytes, qscales, kind="linear")
                    except ValueError as e:
                        print(f"Interpolation error!")
                        print(f"num_bytes={num_bytes}")
                        print(f"sqes={sqes}")
                        raise e

                    self._precomputed_curve[idx][i][j]["b_e"] = b_e
                    self._precomputed_curve[idx][i][j]["b_t"] = b_t
                    self._precomputed_curve[idx][i][j]["min_t"] = min(num_bytes)
                    self._precomputed_curve[idx][i][j]["max_t"] = max(num_bytes)
                    self._precomputed_curve[idx][i][j]["b_q"] = b_q

    def _search(
        self, img_blocks, num_pixels, method_ids, target_byteses, total_target, bpg_psnr
    ):
        if np.sum(target_byteses) > total_target:
            return -np.inf, -np.inf, np.inf
        n_ctu_h, n_ctu_w, _, c, ctu_h, ctu_w = img_blocks.shape

        sqe = 0
        global_time = 0

        for i in range(n_ctu_h):
            for j in range(n_ctu_w):
                img_block = img_blocks[i, j]
                method_id = method_ids[i][j]
                target_bytes = target_byteses[i][j]

                precomputed_results = self._precomputed_curve[method_id][i][j]

                ctu_sqe = precomputed_results["b_e"](target_bytes)
                t = np.polyval(precomputed_results["b_t"], target_bytes)

                global_time += t
                sqe += ctu_sqe

        sqe /= num_pixels * 3
        psnr = -10 * np.log10(sqe)
        return psnr - W_TIME * global_time - bpg_psnr, psnr, global_time

    def _normal_noise_like(self, x, sigma):
        noise = np.random.normal(0, sigma, x.shape).astype(np.int32)
        return noise

    def _hybrid(self, header1, header2, total_target_bytes):
        size = header1.method_ids.shape

        # Mutate methods used
        mutate_method = np.random.choice([0, 1], size)
        new_method_ids = np.select(
            [mutate_method == 0, mutate_method == 1],
            [header1.method_ids, header2.method_ids],
        )

        # Mutate target byterates
        mutate_target = np.random.choice([0, 1], size)
        new_target = np.select(
            [mutate_target == 0, mutate_target == 1],
            [header1.target_byteses, header2.target_byteses],
        )
        new_header = Solution(new_method_ids, new_target)
        new_header.normalize_target_byteses(
            self._minimal_bytes, self._maximal_bytes, total_target_bytes
        )

        return new_header

    def _mutate(
        self,
        header: Solution,
        total_target_bytes,
        method_mutate_p=0.02,
        byte_mutate_sigma=512,
        inplace=True,
    ):
        n_ctu_h, n_ctu_w = header.method_ids.shape
        n_ctus = n_ctu_h * n_ctu_w

        max_method_id = len(self.methods) - 1
        if not inplace:
            header = copy.copy(header)

        # Mutate methods used
        size = header.method_ids.shape
        mutate_method = np.random.choice(
            [0, 1], size, True, [1 - method_mutate_p, method_mutate_p]
        )
        random_header = np.random.random_integers(0, max_method_id, size)
        header.method_ids = np.select(
            [mutate_method == 0, mutate_method == 1], [header.method_ids, random_header]
        )

        # Mutate target byterates
        old_target = header.target_byteses
        byterate_noise = self._normal_noise_like(old_target, byte_mutate_sigma)
        new_target = old_target + byterate_noise
        header.target_byteses = new_target

        header.normalize_target_byteses(
            self._minimal_bytes, self._maximal_bytes, total_target_bytes
        )

        return header

    def _search_init_qscale(self, method, img_blocks, total_target_bytes):
        min_qs = float(1e-5)
        max_qs = float(1.0)
        n_ctu_h, n_ctu_w, _, c, ctu_h, ctu_w = img_blocks.shape
        target_bytes = np.zeros([n_ctu_h, n_ctu_w])

        while min_qs < max_qs - 1e-3:
            mid_qs = (max_qs + min_qs) / 2.0
            total_bytes = 0
            for i in range(n_ctu_h):
                for j in range(n_ctu_w):
                    bitstream = method.compress_block(img_blocks[i, j], mid_qs)
                    len_bytes = len(bitstream)
                    target_bytes[i, j] = len_bytes
                    total_bytes += len_bytes
            if total_bytes <= total_target_bytes:
                max_qs = mid_qs
            else:
                min_qs = mid_qs

        return max_qs, target_bytes

    def _show_solution(self, solution: Solution, total_target_bytes):
        total_bytes = np.sum(solution.target_byteses)
        print(
            f"score={solution.score}; total_bytes=[{total_bytes}/{total_target_bytes}]({100.0*total_bytes/total_target_bytes:.4f}%); PSNR={solution.psnr:.3f}; time={solution.time:.3f}s",
            flush=True,
        )
        for i in range(solution.n_ctu_h):
            for j in range(solution.n_ctu_w):
                method_id = solution.method_ids[i, j]
                target_bytes = solution.target_byteses[i, j]

                curves = self._precomputed_curve[method_id][i][j]

                min_tb = curves["min_t"]
                max_tb = curves["max_t"]

                est_time = np.polyval(curves["b_t"], target_bytes)
                est_qscale = curves["b_q"](target_bytes)
                est_sqe = curves["b_e"](target_bytes)

                print(
                    f"- CTU [{i}, {j}]:\tmethod_id={method_id}\ttarget_bytes={target_bytes}(in [{min_tb}, {max_tb}])\tdec_time={1000*est_time:.2f}ms\tqscale={est_qscale:.5f}\tsquared_error={est_sqe:.6f}"
                )

    def _solve_genetic(
        self,
        img_blocks,
        num_pixels,
        total_target_bytes,
        bpg_psnr,
        N=10000,
        num_generation=10000,
        survive_rate=0.05,
    ):
        n_ctu_h, n_ctu_w, _, c, ctu_h, ctu_w = img_blocks.shape
        n_ctus = n_ctu_h * n_ctu_w

        DEFAULT_METHOD = 0

        print("Initializing qscale")
        default_qscale, default_target_bytes = self._search_init_qscale(
            self.methods[DEFAULT_METHOD][0], img_blocks, total_target_bytes
        )

        print("Precompute all scorees")
        self._precompute_score(img_blocks)

        # Generate initial solutions
        solutions = []
        for k in (pbar := tqdm.tqdm(range(N))):
            pbar.set_description(f"Generate initial solutions")
            method_ids = np.zeros([n_ctu_h, n_ctu_w], dtype=np.int32) + DEFAULT_METHOD
            target_byteses = default_target_bytes
            method = Solution(method_ids, target_byteses)
            self._mutate(
                method,
                total_target_bytes,
                method_mutate_p=0.0 if k == 0 else 1.0,
                byte_mutate_sigma=0.0 if k == 0 else 4096,
            )
            method.score, method.psnr, method.time = self._search(
                img_blocks,
                num_pixels,
                method.method_ids,
                method.target_byteses,
                total_target_bytes,
                bpg_psnr,
            )
            solutions.append(method)
        solutions.sort(key=lambda x: x.score, reverse=True)

        max_score = solutions[0].score
        best_psnr = solutions[0].psnr
        best_time = solutions[0].time

        num_alive = int(math.floor(N * survive_rate))

        for k in range(num_generation):
            # show best solution on generation
            best_solution: Solution = solutions[0]

            print(f"Best Solution before Gen.{k}:")
            self._show_solution(best_solution, total_target_bytes)
            print(flush=True)

            # Kill last solutions
            solutions = solutions[:num_alive]

            # Hybrid
            for i in (pbar := tqdm.tqdm(range(N - num_alive))):
                pbar.set_description(
                    f"Calculating score for generation {k+1}; max_score={max_score:.3f}; best_psnr={best_psnr:.3f}; best_time={best_time:.3f}"
                )
                parent_id1 = random.randint(0, num_alive - 1)
                parent_id2 = random.randint(0, num_alive - 1)
                newborn = self._hybrid(
                    solutions[parent_id1], solutions[parent_id2], total_target_bytes
                )
                self._mutate(newborn, total_target_bytes)
                newborn.score, newborn.psnr, newborn.time = self._search(
                    img_blocks,
                    num_pixels,
                    newborn.method_ids,
                    newborn.target_byteses,
                    total_target_bytes,
                    bpg_psnr,
                )
                solutions.append(newborn)
                if max_score is None or max_score < newborn.score:
                    max_score = newborn.score
                    best_time = newborn.time
                    best_psnr = newborn.psnr
                solutions.sort(key=lambda x: x.score, reverse=True)

        # Calculate q_scale and bitstream for CTUs
        bitstreams = []
        q_scales = np.zeros([n_ctu_h, n_ctu_w], dtype=np.float32)
        best_solution: Solution = solutions[0]
        for i in range(n_ctu_h):
            bitstreams_tmp = []
            for j in range(n_ctu_w):
                method, method_name, method_id = self.methods[
                    best_solution.method_ids[i, j]
                ]
                target = best_solution.target_byteses[i, j]
                print(
                    f"Encoding CTU[{i}, {j}]: method #{method_id}('{method_name}'); Target = {target} B"
                )
                bitstream, q_scale = self._compress_with_target(
                    method, img_blocks[i, j], target
                )
                bitstreams_tmp.append(bitstream)
                q_scales[i, j] = q_scale
            bitstreams.append(bitstreams_tmp)

        return best_solution.method_ids, q_scales, bitstreams

    @staticmethod
    def read_img(img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        rgb = Image.open(img_path).convert("RGB")
        rgb = np.asarray(rgb).astype("float32").transpose(2, 0, 1)
        rgb = rgb / 255.0
        rgb = torch.from_numpy(rgb).type(torch.half)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.cuda()
        return rgb

    def pad_img(self, x):
        pic_height = x.shape[2]
        pic_width = x.shape[3]
        padding_l, padding_r, padding_t, padding_b = get_padding_size(
            pic_height, pic_width, self.ctu_size
        )
        x_padded = F.pad(
            x,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )
        return pic_height, pic_width, x_padded

    def encode(self, input_pth, output_pth, num_generation):
        input_img = self.read_img(input_pth)
        h, w, padded_img = self.pad_img(input_img)

        file_io = FileIO(h, w, self.ctu_size)

        total_target_bytes, bpg_psnr = get_bpg_result(input_pth)
        target_bpp = total_target_bytes * 8 / h / w
        print(f"Image shape: {h}x{w}")
        print(
            f"Target={total_target_bytes}B; Target bpp={target_bpp:.4f}; bpg_psnr={bpg_psnr:.2f}"
        )
        img_blocks = einops.rearrange(
            padded_img,
            "b c (n_ctu_h ctu_size_h) (n_ctu_w ctu_size_w) -> n_ctu_h n_ctu_w b c ctu_size_h ctu_size_w",
            ctu_size_h=self.ctu_size,
            ctu_size_w=self.ctu_size,
        )

        header_bytes = file_io.header_size
        safety_bytes = SAFETY_BYTE_PER_CTU * file_io.num_ctu
        total_target_bytes -= header_bytes + safety_bytes
        print(
            f"Header bytes={header_bytes}; Safety_bytes={safety_bytes}; CTU bytes={total_target_bytes}"
        )

        method_ids, q_scales, bitstreams = self._solve_genetic(
            img_blocks,
            h * w,
            total_target_bytes,
            bpg_psnr,
            num_generation=num_generation,
        )
        file_io.method_id = method_ids
        file_io.bitstreams = bitstreams
        file_io.q_scale = q_scales
        file_io.dump(output_pth)

    def decode(self, file_io):
        decoded_ctus = []
        with timer.Timer("Decode_CTU"):
            for i in range(file_io.ctu_h):
                for j in range(file_io.ctu_w):
                    method_id = file_io.method_id[i, j]
                    q_scale = file_io.q_scale[i, j]
                    bitstream = file_io.bitstreams[i][j]
                    method, method_name, _ = self.methods[method_id]
                    print(
                        f"Decoding CTU[{i}, {j}]: method #{method_id}('{method_name}'); q_scale={q_scale:.6f}"
                    )
                    ctu = method.decompress_block(
                        bitstream, self.ctu_size, self.ctu_size, q_scale
                    )
                    decoded_ctus.append(ctu)
        with timer.Timer("Reconstruct&save"):
            recon_img = torch.cat(decoded_ctus, dim=0)
            recon_img = einops.rearrange(
                recon_img,
                "(n_ctu_h n_ctu_w) c ctu_size_h ctu_size_w -> 1 c (n_ctu_h ctu_size_h) (n_ctu_w ctu_size_w)",
                n_ctu_h=file_io.ctu_h,
            )
            recon_img = recon_img[0, :, : file_io.h, : file_io.w]

        return recon_img
