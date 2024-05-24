from PIL import Image
import os
import numpy as np

from src.loss import PSNRLoss
from src.utils import psnr_with_file
from src.engine import ImageBlock
from src.engine import FileIO


# 使用示例
img1 = "images/6720x4480/IMG_3227.png"
img2 = "images/6720x4480/IMG_1813.png"

print(psnr_with_file(img1, img2))

lossfunc = PSNRLoss()
img1 = Image.open(img1)
img2 = Image.open(img2)

img1 = np.asarray(img1)
img2 = np.asarray(img2)

fileio1 = FileIO(4480, 6720, 512, False)
local_level_losses = []
for i, (upper, left, bottom, right) in enumerate(fileio1.block_indexes):
    img_ref = img1[upper:bottom, left:right, :]
    img_rec = img2[upper:bottom, left:right, :]
    img_ref = ImageBlock(img_ref, None)
    local_loss = lossfunc.ctu_level_loss(img_ref, img_rec)
    local_level_losses.append(local_loss)

img1_tmp = img1.astype(np.float32)
img2_tmp = img2.astype(np.float32)
mse2 = np.mean(np.square(img1_tmp - img2_tmp)) / (255.0**2)
print("MSE=", mse2)
print(
    f"h:{fileio1.h}; w:{fileio1.w}; hxw: {6720*4480}; num_pixels: {fileio1.num_pixels}"
)
print("MSE_LOSS=", np.sum(local_level_losses) / fileio1.num_pixels)

print(lossfunc.global_level_loss(fileio1, local_level_losses))
