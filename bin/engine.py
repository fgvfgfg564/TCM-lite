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

from EVC.bin.engine import ModelEngine, MODELS

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

class Header:
    def __init__(self, method_ids, target_bitses) -> None:
        self.method_ids = method_ids
        self.target_bitses = target_bitses
    
    def normalize_target_bitses(self, min_table, max_table, total_target):
        n_block_h, n_block_w = self.method_ids.shape
        min_bits = np.zeros_like(self.target_bitses,dtype=np.int32)
        max_bits = np.zeros_like(self.target_bitses,dtype=np.int32)

        for i in range(n_block_h):
            for j in range(n_block_w):
                min_bits[i, j] = min_table[self.method_ids[i, j]][i][j]
                max_bits[i, j] = max_table[self.method_ids[i, j]][i][j]
        
        old_target = self.target_bitses
        old_target -= min_bits
        old_target = np.maximum(old_target, 0)
        old_target = old_target.astype(np.float32)
        old_target_rate = old_target / np.sum(old_target)

        new_target = old_target_rate * (total_target - np.sum(min_bits))
        new_target = np.floor(new_target).astype(np.int32)
        new_target += min_bits
        new_target = np.minimum(new_target, max_bits)
        self.target_bitses = new_target
        assert(np.sum(self.target_bitses) <= total_target)

class Engine:
    def __init__(self, ctu_size=512, num_qscale_samples=20) -> None:
        self.ctu_size = ctu_size
        self.methods = []
        idx = 0

        # EVC models
        for model_name in MODELS.keys():
            self.methods.append((ModelEngine.from_model_name(model_name), model_name, idx))
            idx += 1
        
        self.num_qscale_samples = num_qscale_samples
        self.qscale_samples = np.linspace(0, 1, num_qscale_samples)[::-1]
    
    def _compress_with_bitrate(self, method, image_block, target_bits):
        min_qs = 1e-5
        max_qs = 1.

        while min_qs < max_qs - 1e-3:
            mid_qs = (max_qs + min_qs) / 2.
            bits = method.compress_block(image_block, mid_qs)
            len_bits = len(bits) * 8
            if len_bits <= target_bits:
                max_qs = mid_qs
            else:
                min_qs = mid_qs
        
        bits = method.compress_block(image_block, max_qs)
        return bits, max_qs
    
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

    def _estimate_loss(self, method, image_block, target_bits=None, q_scale=None, repeat=1): 
        times = []
        mses = []
        _, c, h, w = image_block.shape
        for i in range(repeat):
            if q_scale is None:
                bits, q_scale = self._compress_with_bitrate(method, image_block, target_bits)
            else:
                bits = method.compress_block(image_block, q_scale)
            time0 = time.time()
            recon_img = method.decompress_block(bits, h, w, q_scale)
            recon_img = torch.clamp(recon_img, 0, 1)

            torch.cuda.synchronize()
            times.append(time.time() - time0)

            image_block = self.torch_pseudo_quantize_to_uint8(image_block)
            recon_img = self.torch_pseudo_quantize_to_uint8(recon_img)

            mse = torch.mean((image_block - recon_img)**2).detach().cpu().numpy()
            mses.append(mse)

        mse = np.mean(mses)
        psnr = -10*np.log10(mse)
        
        return mse, psnr, np.mean(times), bits, recon_img, q_scale
    
    def _precompute_loss(self, img_blocks):
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape
        n_methods = len(self.methods)

        self._precomputed_curve = {}
        self._minimal_bits = np.zeros([n_methods, n_block_h, n_block_w], dtype=np.int32)
        self._maximal_bits = np.zeros([n_methods, n_block_h, n_block_w], dtype=np.int32)

        pbar = tqdm.trange(n_methods * n_block_h * n_block_w)
        pbar.set_description("Precomputing loss")
        pbar_iter = pbar.__iter__()
        for method, _, idx in self.methods:
            self._precomputed_curve[idx] = {}
            for i in range(n_block_h):
                self._precomputed_curve[idx][i] = {}
                for j in range(n_block_w):
                    self._precomputed_curve[idx][i][j] = {}
                    mses = []
                    num_bits = []
                    times = []
                    for qscale in self.qscale_samples:
                        image_block = img_blocks[i, j]
                        
                        mse, psnr, dec_time, bits, __, ___ = self._estimate_loss(method, image_block, None, qscale, 5)
                        num_bit = len(bits) * 8
                        num_bits.append(num_bit)
                        mses.append(mse)
                        times.append(dec_time)

                    self._minimal_bits[idx, i, j] = num_bits[0]
                    self._maximal_bits[idx, i, j] = num_bits[-1]

                    b_m = interpolate.interp1d(num_bits, mses, kind='cubic')
                    b_t = interpolate.interp1d(num_bits, times, kind='cubic')
                    self._precomputed_curve[idx][i][j]['b_m'] = b_m
                    self._precomputed_curve[idx][i][j]['b_t'] = b_t
                    pbar_iter.__next__()
    
    def _search(self, img_blocks, method_ids, target_bitses):
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape

        global_mse = []
        global_time = 0

        for i in range(n_block_h):
            for j in range(n_block_w):
                img_block = img_blocks[i, j]
                method_id = method_ids[i][j]
                target_bits = target_bitses[i][j]

                precomputed_results = self._precomputed_curve[method_id][i][j]

                mse = precomputed_results['b_m'](target_bits)
                t = precomputed_results['b_t'](target_bits)

                global_time += t
                global_mse.append(mse)
        
        global_mse = np.mean(global_mse)
        psnr = -10*np.log10(global_mse)
        return psnr - global_time, psnr, global_time
    
    def _normal_noise_like(self, x, sigma):
        noise = np.random.normal(0, sigma, x.shape).astype(np.int32)
        return noise
    
    def _hybrid(self, header1, header2, total_target_bits):
        size = header1.method_ids.shape

        # Mutate methods used
        mutate_method = np.random.choice([0, 1], size)
        new_method_ids = np.select([mutate_method == 0, mutate_method == 1], [header1.method_ids, header2.method_ids])

        # Mutate target bitrates
        mutate_target = np.random.choice([0, 1], size)
        new_target = np.select([mutate_target == 0, mutate_target == 1], [header1.target_bitses, header2.target_bitses])
        new_header = Header(new_method_ids, new_target)
        new_header.normalize_target_bitses(self._minimal_bits, self._maximal_bits, total_target_bits)

        return new_header
    
    def _mutate(self, header: Header, total_target_bits, method_mutate_p=0.5, bit_mutate_sigma=8192, inplace=True):
        n_block_h, n_block_w = header.method_ids.shape
        n_blocks = n_block_h * n_block_w

        max_method_id = len(self.methods) - 1
        if not inplace:
            header = copy.copy(header)
        
        # Mutate methods used
        size = header.method_ids.shape
        mutate_method = np.random.choice([0, 1], size, True, [1 - method_mutate_p, method_mutate_p])
        random_header = np.random.random_integers(0, max_method_id, size)
        header.method_ids = np.select([mutate_method == 0, mutate_method == 1], [header.method_ids, random_header])

        # Mutate target bitrates
        old_target = header.target_bitses
        bitrate_noise = self._normal_noise_like(old_target, bit_mutate_sigma)
        new_target = old_target + bitrate_noise
        header.target_bitses = new_target

        header.normalize_target_bitses(self._minimal_bits, self._maximal_bits, total_target_bits)

        return header
    
    def _search_init_qscale(self, method, img_blocks, total_target_bits):
        min_qs = 1e-5
        max_qs = 1.
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape
        target_bits = np.zeros([n_block_h, n_block_w])

        while min_qs < max_qs - 1e-3:
            mid_qs = (max_qs + min_qs) / 2.
            total_bits = 0
            for i in range(n_block_h):
                for j in range(n_block_w):
                    bits = method.compress_block(img_blocks[i, j], mid_qs)
                    len_bits = len(bits) * 8
                    target_bits[i, j] = len_bits
                    total_bits += len_bits
            if total_bits <= total_target_bits:
                max_qs = mid_qs
            else:
                min_qs = mid_qs
        
        return max_qs, target_bits

    def _solve_genetic(self, img_blocks, total_target_bits, N=10000, num_generation=10000, survive_rate=0.1):
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape
        n_blocks = n_block_h * n_block_w

        print("Initializing qscale")
        default_qscale, default_target_bits = self._search_init_qscale(self.methods[0][0], img_blocks, total_target_bits)

        print("Precompute all losses")
        self._precompute_loss(img_blocks)
        
        # Generate initial solves
        solves = []
        for k in range(N):
            method_ids = np.zeros([n_block_h, n_block_w], dtype=np.int32)
            target_bitses = default_target_bits
            method = Header(method_ids, target_bitses)
            self._mutate(method, total_target_bits)
            solves.append(method)
        
        max_score = -1
        best_psnr = -1
        best_time = -1

        for u in (pbar := tqdm.tqdm(solves)):
            pbar.set_description(f"Calculating loss for generation 0; max_score={max_score:.3f}; best_psnr={best_psnr:.3f}; best_time={best_time:.3f}")
            if not hasattr(u, 'loss'):
                u.loss, u.psnr, u.time = self._search(img_blocks, u.method_ids, u.target_bitses)
                if max_score is None or max_score < u.loss:
                    max_score = u.loss
                    best_time = u.time
                    best_psnr = u.psnr
        
        num_alive = int(math.floor(N*survive_rate))

        for k in range(num_generation):
            solves.sort(key=lambda x:x.loss, reverse=True)

            # show best solution on generation
            best_solution:Header = solves[0]
            print(f"best loss on generation {k}: {best_solution.loss}; total_bits={np.sum(best_solution.target_bitses)}", flush=True)
            print("method_ids =", best_solution.method_ids.flatten())
            print("target_bits =", best_solution.target_bitses.flatten())

            # Kill last solutions
            solves = solves[:num_alive]

            # Hybrid
            for i in (pbar := tqdm.tqdm(range(N - num_alive))):
                pbar.set_description(f"Calculating loss for generation {k+1}; max_score={max_score:.3f}; best_psnr={best_psnr:.3f}; best_time={best_time:.3f}")
                parent_id1 = random.randint(0, num_alive - 1)
                parent_id2 = random.randint(0, num_alive - 1)
                newborn = self._hybrid(solves[parent_id1], solves[parent_id2], total_target_bits)
                self._mutate(newborn, total_target_bits)
                newborn.loss, newborn.psnr, newborn.time = self._search(img_blocks, newborn.method_ids, newborn.target_bitses)
                solves.append(newborn)
                if max_score is None or max_score < newborn.loss:
                    max_score = newborn.loss
                    best_time = newborn.time
                    best_psnr = newborn.psnr
                    

    @staticmethod
    def read_img(img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        rgb = Image.open(img_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        rgb = torch.from_numpy(rgb).type(torch.half)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.cuda()
        return rgb
    
    def pad_img(self, x):
        pic_height = x.shape[2]
        pic_width = x.shape[3]
        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, self.ctu_size)
        x_padded = F.pad(
            x,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )
        return pic_height, pic_width, x_padded

    @staticmethod
    def save_torch_image(img, save_path):
        img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(save_path)

    def encode(self, input_pth, output_pth, target_bpp):
        input_img = self.read_img(input_pth)
        h, w, padded_img = self.pad_img(input_img)
        total_target_bits = np.floor(target_bpp * h * w).astype(np.int32)

        img_blocks = padded_img.unfold(2, self.ctu_size, self.ctu_size).unfold(3, self.ctu_size, self.ctu_size)

        img_blocks = torch.permute(img_blocks, (2, 3, 0, 1, 4, 5))
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape
        n_block = n_block_h * n_block_w

        self._solve_genetic(img_blocks, total_target_bits)