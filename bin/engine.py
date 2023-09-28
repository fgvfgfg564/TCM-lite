import os
import numpy as np
import time
from PIL import Image
import torch
import torch.nn.functional as F

import tqdm
import copy

from EVC.bin.engine import ModelEngine, MODELS

MINIMAL_BITS = 512

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


class Engine:
    def __init__(self, ctu_size=512) -> None:
        self.ctu_size = ctu_size
        self.methods = []
        idx = 0

        # EVC models
        for model_name in MODELS.keys():
            self.methods.append((ModelEngine.from_model_name(model_name), model_name, idx))
            idx += 1
    
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

    def _estimate_loss(self, method, image_block, target_bits, repeat=1): 
        times = []
        mses = []
        _, c, h, w = image_block.shape
        for i in range(repeat):
            bits, q_scale = self._compress_with_bitrate(method, image_block, target_bits)
            time0 = time.time()
            recon_img = method.decompress_block(bits, h, w, q_scale)

            torch.cuda.synchronize()
            times.append(time.time() - time0)
            mses.append(torch.mean((image_block - recon_img) ** 2).detach().cpu().numpy().item())
        
        return np.mean(mses), np.mean(times), bits, recon_img, q_scale
    
    def _search(self, img_blocks, method_ids, target_bitses):
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape

        global_mse = []
        global_time = 0

        for i in range(n_block_h):
            for j in range(n_block_w):
                img_block = img_blocks[i, j]
                method_id = method_ids[i][j]
                target_bits = target_bitses[i][j]

                method = self.methods[method_id][0]
                mse, t, _, __, ___ = self._estimate_loss(method, img_block, target_bits)

                global_time += t
                global_mse.append(mse)
        
        global_mse = np.mean(global_mse)
        psnr = -10*np.log10(global_mse)
        return psnr - global_time
    
    def _normal_noise_like(self, x, sigma):
        noise = np.random.normal(0, sigma, x.shape).astype(np.int32)
        return noise
    
    def _normalize_target_bits(self, x, total_target_bits):
        x = np.maximum(x, MINIMAL_BITS)
        new_total = np.sum(x)
        valid = x > MINIMAL_BITS
        num_valid = np.sum(valid)
        bias_per_block = np.floor((total_target_bits - new_total) / num_valid).astype(np.int32)
        x[valid] += bias_per_block
        assert(np.sum(x) <= total_target_bits)
        return x

    
    def _hybrid(self, header1, header2, total_target_bits):
        size = header1.method_ids.shape

        # Mutate methods used
        mutate_method = np.random.choice([0, 1], size)
        new_method_ids = np.select([mutate_method == 0, mutate_method == 1], [header1.method_ids, header2.method_ids])

        # Mutate target bitrates
        mutate_target = np.random.choice([0, 1], size)
        new_target = np.select([mutate_target == 0, mutate_target == 1], [header1.target_bitses, header2.target_bitses])
        new_target = self._normalize_target_bits(new_target, total_target_bits)

        return Header(new_method_ids, new_target)
    
    def _mutate(self, header: Header, total_target_bits, method_mutate_p=0.1, bit_mutate_sigma=1024, inplace=True):
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
        new_target = self._normalize_target_bits(new_target, total_target_bits)

        header.target_bitses = new_target
        return header

    def _solve_genetic(self, img_blocks, total_target_bits, N=100, num_generation=100):
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape
        n_blocks = n_block_h * n_block_w

        default_target_bits = total_target_bits // n_blocks
        
        # Generate initial solves
        solves = []
        for k in range(N*3):
            method_ids = np.zeros([n_block_h, n_block_w], dtype=np.int32)
            target_bitses = default_target_bits + np.zeros([n_block_h, n_block_w], dtype=np.int32)
            method = Header(method_ids, target_bitses)
            self._mutate(method, total_target_bits)
            solves.append(method)
        
        for u in (pbar := tqdm.tqdm(solves)):
            pbar.set_description(f"Calculating loss for generation 0")
            if not hasattr(u, 'loss'):
                u.loss = self._search(img_blocks, u.method_ids, u.target_bitses)
        
        for k in range(num_generation):
            solves.sort(key=lambda x:x.loss, reverse=True)
            print(f"best loss on generation {k}: {solves[0].loss}; total_bits={np.sum(solves[0].target_bitses)}", flush=True)

            # Kill last solutions
            solves = solves[:2*N]

            # Hybrid
            pbar = tqdm.tqdm(range(N))
            pbar.set_description(f"Calculating loss for generation {k+1}")
            for i in pbar:
                newborn = self._hybrid(solves[i*2], solves[i*2+1], total_target_bits)
                self._mutate(newborn, total_target_bits)
                newborn.loss = self._search(img_blocks, newborn.method_ids, newborn.target_bitses)
                solves.append(newborn)
                    

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