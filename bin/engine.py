import os
import numpy as np
import time
from PIL import Image
import torch
import torch.nn.functional as F
from dataclasses import dataclass

import tqdm
import copy

from coding_tools.coding_tool import CodingToolBase
import coding_tools.utils.timer as timer

from .utils import *
from .fileio import FileIO, get_padding_size
from .math import *
from ..coding_tools import TOOL_GROUPS

SAFETY_BYTE_PER_CTU = 2

np.seterr(all="raise")

@dataclass
class PaddedBlock:
    padded_block_np: np.ndarray
    padded_block_cuda: torch.Tensor
    h: int
    w: int

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

    def normalize_target_byteses(self, min_table, max_table, total_target):
        min_bytes = np.zeros_like(self.target_byteses, dtype=np.int32)
        max_bytes = np.zeros_like(self.target_byteses, dtype=np.int32)

        for i in range(self.n_ctu):
            min_bytes[i] = min_table[self.method_ids[i]][i]
            max_bytes[i] = max_table[self.method_ids[i]][i]
        
        self.target_byteses = normalize_to_target(self.target_byteses, min_bytes, max_bytes, total_target)

class EngineBase:
    CACHE_DIR = os.path.join(os.path.split(__file__)[0], "../cache")

    def __init__(
        self,
        ctu_size,
        mosaic,
        num_qscale_samples=20,
        tool_groups=TOOL_GROUPS.keys(),
        tool_filter=None,
        dtype=torch.half,
    ) -> None:
        self.ctu_size = ctu_size
        self.mosaic = mosaic
        self.methods = []

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
    
    def _feed_block(method: CodingToolBase, image_block: PaddedBlock, qscale):
        if method.PLATFORM == 'numpy':
            return method.compress_block(image_block.padded_block_np, qscale)
        else:
            return method.compress_block(image_block.padded_block_cuda, qscale)

    def _compress_with_target(self, method, image_block: PaddedBlock, target_bytes):
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
    
    def _search_init_qscale(self, method, img_blocks, total_target_bytes):
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
            if total_bytes <= total_target_bytes:
                max_qs = mid_qs
            else:
                min_qs = mid_qs

        return max_qs, target_bytes

    @classmethod
    def torch_to_uint8(cls, x):
        x = torch.clamp(x, 0, 1)
        x *= 255
        x = torch.round(x)
        x = x.to(torch.uint8)
        return x

    def torch_pseudo_quantize_to_uint8(self, x):
        x = self.torch_to_uint8(x)
        x = x.to(self.dtype) / 255.0
        return x

    def _try_compress_decompress(
        self,
        method: CodingToolBase,
        image_block: PaddedBlock,
        target_bytes=None,
        q_scale=None,
        repeat=1,
    ):
        times = []
        sqes = []
        _, c, h, w = image_block.padded_block.shape
        for i in range(repeat):
            if q_scale is None:
                bitstream, q_scale = self._compress_with_target(
                    method, image_block.padded_block, target_bytes
                )
            else:
                bitstream = self._feed_block(method, image_block, q_scale)
            time0 = time.time()
            recon_img = method.decompress_block(bitstream, h, w, q_scale)
            recon_img = torch.clamp(recon_img, 0, 1)

            torch.cuda.synchronize()
            times.append(time.time() - time0)

            ref = self.torch_pseudo_quantize_to_uint8(image_block.padded_block)
            recon_img = self.torch_pseudo_quantize_to_uint8(recon_img)

            ref = ref[:, :image_block.h, :image_block.w, :]
            recon_img = recon_img[:, :image_block.h, :image_block.w, :]

            sqe = torch.sum((ref - recon_img) ** 2).detach().cpu().numpy()
            sqes.append(sqe)

        sqe = np.mean(sqes)
        psnr = -10 * np.log10(sqe)

        return sqe, np.mean(times), bitstream, recon_img, q_scale

    def _precompute_score(self, img_blocks, img_size, img_hash):
        h, w = img_size
        n_ctu = len(img_blocks)
        n_methods = len(self.methods)

        self._precomputed_curve = {}
        self._minimal_bytes = np.zeros([n_methods, n_ctu], dtype=np.int32)
        self._maximal_bytes = np.zeros([n_methods, n_ctu], dtype=np.int32)

        pbar = tqdm.trange(n_methods * n_ctu * len(self.qscale_samples))
        pbar.set_description("Precomputing score")
        pbar_iter = pbar.__iter__()
        for method, method_name, method_idx in self.methods:
            self._precomputed_curve[method_idx] = {}
            for i in range(n_ctu):
                self._precomputed_curve[method_idx][i] = {}
                cache_dir = os.path.join(self.CACHE_DIR, method_name, img_hash, "-".join([str(self.ctu_size),str(self.mosaic),str(self.num_qscale_samples), str(i)]))
                b_e_file = os.path.join(cache_dir, 'b_e.npz')
                b_t_file = os.path.join(cache_dir, 'b_t.npy')
                b_q_file = os.path.join(cache_dir, 'b_q.npz')
                min_max_file = os.path.join(cache_dir, 'min_max.npz')

                try:
                    # If the caches are complete, load them
                    b_e = LinearInterpolation.load(b_e_file)
                    b_q = LinearInterpolation.load(b_q_file)
                    b_t = np.load(b_t_file)
                    min_max = np.load(min_max_file)
                    min_t = min_max['min_t']
                    max_t = min_max['max_t']
                    print("Loaded cache from:", cache_dir, flush=True)

                    for qscale in self.qscale_samples:
                        pbar_iter.__next__()
                except:
                    # No cache or cache is broken
                    sqes = []
                    num_bytes = []
                    qscales = []
                    times = []

                    for qscale in self.qscale_samples:
                        image_block = img_blocks[i]

                        sqe, dec_time, bitstream, __, ___ = self._try_compress_decompress(
                            method,
                            image_block,
                            target_bytes=None,
                            q_scale=qscale,
                            repeat=1,
                        )
                        num_byte = len(bitstream)
                        num_bytes.append(num_byte)
                        sqes.append(sqe)
                        times.append(dec_time)
                        qscales.append(qscale)
                        pbar_iter.__next__()

                    make_strictly_increasing(num_bytes)

                    if len(num_bytes) <= 3 or not is_strictly_increasing(num_bytes):
                        raise ValueError(
                            f"num_bytes are too few or not strictly increasing: \nnum_bytes={num_bytes}\nsqes={sqes}\nqscales={qscales}"
                        )

                    try:
                        b_e = LinearInterpolation(num_bytes, sqes)
                        # b_t = interpolate.interp1d(num_bytes, times, kind='linear')
                        b_t = np.polyfit(num_bytes, times, 1)
                        b_q = LinearInterpolation(num_bytes, qscales)
                    except ValueError as e:
                        print(f"Interpolation error!")
                        print(f"num_bytes={num_bytes}")
                        print(f"sqes={sqes}")
                        raise e
                    
                    min_t = min(num_bytes)
                    max_t = max(num_bytes)

                    # Save to cache
                    os.makedirs(cache_dir, exist_ok=True)
                    b_e.dump(b_e_file)
                    b_q.dump(b_q_file)
                    np.save(b_t_file, b_t)
                    np.savez(min_max_file, min_t=min_t, max_t=max_t)
                    print("Cache saved to:", cache_dir, flush=True)

                self._precomputed_curve[method_idx][i]["b_e"] = b_e    # interpolate.interp1d; Linear
                self._precomputed_curve[method_idx][i]["b_t"] = b_t    # Linear fitting
                self._minimal_bytes[method_idx][i] = min_t
                self._maximal_bytes[method_idx][i] = max_t
                self._precomputed_curve[method_idx][i]["b_q"] = b_q

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

    def _get_score(
        self, n_ctu, num_pixels, method_ids, target_byteses, total_target
    ):
        # Returns score given method ids and target bytes
        if np.sum(target_byteses) > total_target:
            return -np.inf, -np.inf, np.inf

        sqe = 0
        global_time = 0

        for i in range(n_ctu):
            method_id = method_ids[i]
            target_bytes = target_byteses[i]

            precomputed_results = self._precomputed_curve[method_id][i]

            ctu_sqe = precomputed_results["b_e"](target_bytes)
            t = np.polyval(precomputed_results["b_t"], target_bytes)

            global_time += t
            sqe += ctu_sqe

        sqe /= num_pixels * 3
        psnr = -10 * np.log10(sqe)
        return psnr - self.w_time * global_time, psnr, global_time


    def _solve(
            self, 
            img_blocks,
            img_size,
            total_target_bytes,
            file_io,
            **kwargs,
        ):
        raise NotImplemented

    def read_img(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        rgb_np = Image.open(img_path).convert("RGB")
        rgb_np = np.asarray(rgb_np)
        img_hash = hash_numpy_array(rgb_np)
        return rgb_np, img_hash

    def pad_img(self, x):
        pic_height = x.shape[0]
        pic_width = x.shape[1]
        padding_l, padding_r, padding_t, padding_b = get_padding_size(
            pic_height, pic_width, self.ctu_size
        )
        x_padded = np.pad(
            x,
            ((padding_t, padding_b), (padding_l, padding_r), (0, 0)),
            mode="constant",
            value=0,
        )
        return pic_height, pic_width, x_padded
    
    def divide_blocks(self, fileio: FileIO, h, w, padded_img):
        blocks = []
        for i in range(fileio.n_ctu):
            upper, left, lower, right = fileio.block_indexes[i]
            lower_real = min(lower, h)
            right_real = min(right, w)
            print(f"Block #{i}: ({upper}, {left}) ~ ({lower_real}, {right_real})")
            
            # Move to CUDA
            img_patch_np = padded_img[upper:lower, left:right, :]
            img_patch_cuda = torch.from_numpy(img_patch_np).type(self.dtype)
            img_patch_cuda = img_patch_cuda.unsqueeze(0)
            img_patch_cuda = img_patch_cuda.cuda()

            blocks.append(PaddedBlock(img_patch_np, img_patch_cuda, lower_real - upper, right_real - left))
        
        return blocks

    @torch.inference_mode()
    def encode(
        self,
        input_pth,
        output_pth,
        target_bpp,
        w_time=1.0,
        **kwargs,
    ):
        input_img, img_hash = self.read_img(input_pth)
        h, w, padded_img = self.pad_img(input_img)
        img_size = (h, w)

        file_io = FileIO(h, w, self.ctu_size, self.mosaic)

        # Set loss
        self.w_time = w_time

        total_target_bytes = target_bpp * h * w // 8
        print(f"Image shape: {h}x{w}")
        print(
            f"Target={total_target_bytes}B; Target bpp={target_bpp:.4f};"
        )

        header_bytes = file_io.header_size
        safety_bytes = SAFETY_BYTE_PER_CTU * file_io.n_ctu
        total_target_bytes -= header_bytes + safety_bytes
        print(
            f"Header bytes={header_bytes}; Safety_bytes={safety_bytes}; CTU bytes={total_target_bytes}"
        )

        img_blocks = self.divide_blocks(file_io, h, w, padded_img)

        print("Precompute all scorees", flush=True)
        self._precompute_score(img_blocks, img_size, img_hash)

        method_ids, q_scales, bitstreams, data = self._solve(
            img_blocks,
            img_size,
            total_target_bytes,
            file_io,
            **kwargs,
        )
        file_io.method_id = method_ids
        file_io.bitstreams = bitstreams
        file_io.q_scale = q_scales
        file_io.dump(output_pth)

        return data

    def join_blocks(self, decoded_ctus, file_io: FileIO):
        h = file_io.h
        w = file_io.w
        recon_img = np.zeros(size=(h, w, 3), dtype=np.float32)
        for i, ctu in enumerate(decoded_ctus):
            upper, left, lower, right = file_io.block_indexes[i]
            lower_real = min(lower, h)
            right_real = min(right, w)

            recon_img[upper:lower_real, left:right_real, :] = ctu[:lower_real-upper, :right_real-left, :]
        return recon_img

    @torch.inference_mode()
    def decode(self, file_io: FileIO):
        """
        Decode into NumPy array with range 0-1
        """
        with torch.no_grad():
            decoded_ctus = []
            with timer.Timer("Decode_CTU"):
                for i, (upper, left, lower, right) in enumerate(file_io.block_indexes):
                    method_id = file_io.method_id[i]
                    q_scale = file_io.q_scale[i]
                    bitstream = file_io.bitstreams[i]
                    method, method_name, _ = self.methods[method_id]
                    print(
                        f"Decoding CTU #{i}: ({upper}, {left}) ~ ({lower}, {right})  method #{method_id}('{method_name}'); q_scale={q_scale:.6f}; len_bitstream={len(bitstream)}B"
                    )
                    torch.cuda.synchronize()
                    ctu = method.decompress_block(
                        bitstream, lower-upper, right-left, q_scale
                    )
                    if isinstance(ctu, torch.Tensor):
                        ctu = ctu[0].permute(1, 2, 0).detach().cpu().numpy()
                    decoded_ctus.append(ctu)
            with timer.Timer("Reconstruct&save"):
                recon_img = self.join_blocks(decoded_ctus, file_io)

            return recon_img

class SAEngine1(EngineBase):

    @staticmethod
    def _calc_gradient_psnr(sqes: np.ndarray):
        r"""
        $ PSNR(sqes) = - 10 * \log_{10}{(\frac{\sum X}{num\_pixels})} $
        """

        return -10 / (sqes.sum() * np.log(10))


    def _find_optimal_target_bytes(
        self, file_io: FileIO, n_ctu, num_pixels, method_ids, total_target, learning_rate=1e5, num_steps=1000, init_value=None
    ):
        """
        Find the optimal target bytes given CTU methods
        """

        min_bytes = np.zeros([n_ctu,], dtype=np.int32)
        max_bytes = np.zeros([n_ctu,], dtype=np.int32)

        for i in range(n_ctu):
            min_bytes[i] = self._minimal_bytes[method_ids[i]][i]
            max_bytes[i] = self._maximal_bytes[method_ids[i]][i]

        if init_value is None:
            bpp = total_target / num_pixels
            ans = file_io.block_num_pixels * bpp
        else:
            ans = init_value
        
        ans = normalize_to_target(ans, min_bytes, max_bytes, total_target)

        for step in range(num_steps):
            gradients = np.zeros_like(ans)

            # Gradient item on sqe
            sqes = []
            for i in range(n_ctu):
                method_id = method_ids[i]
                precomputed_results = self._precomputed_curve[method_id][i]
                sqes.append(precomputed_results["b_e"](ans[i]))
            
            sqe_gradient = self._calc_gradient_psnr(np.array(sqes))
            
            for i in range(n_ctu):
                b_e: LinearInterpolation = self._precomputed_curve[method_id][i]["b_e"]
                b_t: np.ndarray = self._precomputed_curve[method_id][i]["b_t"]
                gradients[i] = self.w_time * b_t[0] - sqe_gradient * b_e.derivative(ans[i])
            
            # Gradient descent            
            ans = ans - gradients * learning_rate
            # Normalize
            ans = normalize_to_target(ans, min_bytes, max_bytes, total_target)
            
        score, psnr, time = self._get_score(n_ctu, num_pixels, method_ids, ans, total_target)

        return ans, score, psnr, time
    
    def _try_move(self, method_ids: np.ndarray, n_method):
        # Generate a group of new method_ids that move from current state

        # Pure random
        n_ctu = len(method_ids)
        selected = np.random.random_integers(0, n_ctu - 1)
        new_method = np.random.random_integers(0, n_method - 1)
        new_method_ids = method_ids.copy()
        new_method_ids[selected] = new_method
        return new_method_ids

    def _show_solution(self, method_ids, target_byteses, total_target_bytes, score, psnr, time):
        n_ctu = len(method_ids)
        total_bytes = np.sum(target_byteses)
        print(
            f"score={score}; total_bytes=[{total_bytes}/{total_target_bytes}]({100.0*total_bytes/total_target_bytes:.4f}%); PSNR={psnr:.3f}; time={time:.3f}s",
            flush=True,
        )
        for i in range(n_ctu):
            method_id = method_ids[i]
            target_bytes = target_byteses[i]

            curves = self._precomputed_curve[method_id][i]

            min_tb = self._minimal_bytes[method_ids[i]][i]
            max_tb = self._maximal_bytes[method_ids[i]][i]

            est_time = np.polyval(curves["b_t"], target_bytes)
            est_qscale = curves["b_q"](target_bytes)
            est_sqe = curves["b_e"](target_bytes)

            print(
                f"- CTU [{i}]:\tmethod_id={method_id}\ttarget_bytes={target_bytes:.1f}(in [{min_tb}, {max_tb}])\tdec_time={1000*est_time:.2f}ms\tqscale={est_qscale:.5f}\tsquared_error={est_sqe:.6f};",
                flush=True
            )

    def _solve(self, img_blocks, img_size, total_target_bytes, file_io, num_steps=1000, T_start=1e-3, T_end=1e-6, ):
        n_ctu = len(img_blocks)
        n_method = len(self.methods)
        h, w = img_size
        num_pixels = h * w
        alpha = np.power(T_end / T_start, 1. / num_steps)

        ans = np.random.random_integers(0, n_method-1, [n_ctu,])
        target_byteses, score, psnr, time = self._find_optimal_target_bytes(file_io, n_ctu, num_pixels, ans, total_target_bytes, num_steps=10000)

        T = T_start

        # Simulated Annealing
        for step in range(num_steps):
            next_state = self._try_move(ans, n_method)
            next_target_byteses, next_score, next_psnr, next_time = self._find_optimal_target_bytes(
                file_io, n_ctu, num_pixels, next_state, total_target_bytes, init_value=target_byteses, num_steps=100)

            if next_score > score:
                accept = True
            else:
                delta = score - next_score
                p = safe_SA_prob(delta, T)
                accept = (np.random.rand() < p)
            
            if accept:
                ans = next_state
                target_byteses = next_target_byteses
                score = next_score
                psnr = next_psnr
                time = next_time
            
            if step % (num_steps // 20) == 0:
                print(f"Results for step: {step}; T={T:.6f}")
                self._show_solution(ans, target_byteses, total_target_bytes, score, psnr, time)
            
            T *= alpha
        
        solution = Solution(ans, target_byteses)

        method_ids, q_scales, bitstreams = self._compress_blocks(img_blocks, solution)
        return method_ids, q_scales, bitstreams, []

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

    def _hybrid(self, header1: GASolution, header2: GASolution, total_target_bytes, p):
        new_method_score = self.arithmetic_crossover(
            header1.method_score, header2.method_score
        )
        new_target = self.arithmetic_crossover(
            header1.target_byteses, header2.target_byteses
        )
        new_header = GASolution(new_method_score, new_target)
        new_header.normalize_target_byteses(
            self._minimal_bytes, self._maximal_bytes, total_target_bytes
        )

        return new_header

    def _mutate(
        self,
        header: GASolution,
        total_target_bytes,
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

        header.normalize_target_byteses(
            self._minimal_bytes, self._maximal_bytes, total_target_bytes
        )

        return header

    def _show_solution(self, solution: GASolution, total_target_bytes):
        total_bytes = np.sum(solution.target_byteses)
        print(
            f"score={solution.score}; total_bytes=[{total_bytes}/{total_target_bytes}]({100.0*total_bytes/total_target_bytes:.4f}%); PSNR={solution.psnr:.3f}; time={solution.time:.3f}s",
            flush=True,
        )
        for i in range(solution.n_ctu):
            method_id = solution.method_ids[i]
            target_bytes = solution.target_byteses[i]

            curves = self._precomputed_curve[method_id][i]

            min_tb = self._minimal_bytes[solution.method_ids[i]][i]
            max_tb = self._maximal_bytes[solution.method_ids[i]][i]

            est_time = np.polyval(curves["b_t"], target_bytes)
            est_qscale = curves["b_q"](target_bytes)
            est_sqe = curves["b_e"](target_bytes)

            print(
                f"- CTU [{i}]:\tmethod_id={method_id}\ttarget_bytes={target_bytes:.1f}(in [{min_tb}, {max_tb}])\tdec_time={1000*est_time:.2f}ms\tqscale={est_qscale:.5f}\tsquared_error={est_sqe:.6f}; method_scores={solution.method_score[:, i]}"
            )

    def _solve(
        self,
        img_blocks,
        img_size,
        total_target_bytes,
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
            self.methods[DEFAULT_METHOD][0], img_blocks, total_target_bytes
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
                total_target_bytes,
                method_sigma,
                bytes_sigma,
            )
            method.normalize_target_byteses(
                self._minimal_bytes, self._maximal_bytes, total_target_bytes
            )
            method.score, method.psnr, method.time = self._get_score(
                n_ctu,
                num_pixels,
                method.method_ids,
                method.target_byteses,
                total_target_bytes,
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
            self._show_solution(best_solution, total_target_bytes)
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
                    total_target_bytes,
                    T * 0.5,
                )
                self._mutate(
                    newborn,
                    total_target_bytes,
                    T * method_sigma,
                    T * bytes_sigma,
                )
                newborn.score, newborn.psnr, newborn.time = self._get_score(
                    n_ctu,
                    num_pixels,
                    newborn.method_ids,
                    newborn.target_byteses,
                    total_target_bytes,
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

        method_ids, q_scales, bitstreams = self._compress_blocks(img_blocks, solutions[0])
        return method_ids, q_scales, bitstreams, data