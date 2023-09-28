import os
import numpy as np
import time
from PIL import Image
import torch
import torch.nn.functional as F

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

class Engine:
    def __init__(self, ctu_size=512) -> None:
        self.ctu_size = ctu_size
        self.methods = []
        idx = 0

        # EVC models
        for model_name in MODELS.keys():
            self.methods.append((ModelEngine.from_model_name(model_name, True), model_name, idx))
            idx += 1
    
    def _compress_with_bitrate(self, method, image_block, target_bits):
        min_qs = 1e-5
        max_qs = 1.

        while min_qs < max_qs - 1e-7:
            mid_qs = (max_qs + min_qs) / 2.
            bits = method.compress_block(image_block, mid_qs)
            len_bits = len(bits) * 8
            if len_bits <= target_bits:
                max_qs = mid_qs
            else:
                min_qs = mid_qs
        
        bits = method.compress_block(image_block, max_qs)
        return bits, max_qs

    def _estimate_loss(self, method, image_block, target_bits, repeat=3): 
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

        img_blocks = padded_img.unfold(2, self.ctu_size, self.ctu_size).unfold(3, self.ctu_size, self.ctu_size)

        img_blocks = torch.permute(img_blocks, (2, 3, 0, 1, 4, 5))
        n_block_h, n_block_w, _, c, ctu_h, ctu_w = img_blocks.shape
        for i in range(n_block_h):
            for j in range(n_block_w):
                print("Encoding with method: ", self.methods[0][1])
                block = img_blocks[i, j]
                method = self.methods[0][0]
                target_bits = np.floor(target_bpp * ctu_h * ctu_w).astype(np.int32)
                result = self._estimate_loss(method, block, target_bits)
                print(result[0], result[1], len(result[2])*8, target_bits, "q_scale =", result[4])