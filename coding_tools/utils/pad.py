import numpy as np
import torch
import torch.nn.functional as F


def get_padded_hw(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return new_h, new_w


def get_padding_size(height, width, p):
    new_h, new_w = get_padded_hw(height, width, p)
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def pad_np_img(x, p):
    pic_height = x.shape[0]
    pic_width = x.shape[1]
    padding_l, padding_r, padding_t, padding_b = get_padding_size(
        pic_height, pic_width, p
    )
    x_padded = np.pad(
        x,
        ((padding_t, padding_b), (padding_l, padding_r), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return x_padded, (pic_height, pic_width)


def pad_torch_img(x: torch.Tensor, p):
    b, c, h, w = x.shape
    padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w, p)
    x = F.pad(x, (padding_l, padding_r, padding_t, padding_b))
    return x, (h, w)
