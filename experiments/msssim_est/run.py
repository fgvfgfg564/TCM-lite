from typing import Callable

from PIL import Image
from lpips.lpips import *
import torch
import pytorch_msssim
import numpy as np
from io import BytesIO
import os
import matplotlib.pyplot as plt
import math
from src.ba_ms_ssim import ba_ms_ssim, BoundaryInfo
import glob

ROOTDIR = os.path.dirname(__file__)


def compress_image_jpeg(input_array, quality):
    """
    Compresses an image using JPEG compression.

    :param input_array: A NumPy array representing the image.
    :param quality: An integer from 0 to 100 indicating the JPEG quality.
    :return: A NumPy array of the compressed image.
    """
    # Ensure the array is in the range 0-255
    input_array = (input_array * 255).astype("uint8")

    # Convert the NumPy array to a PIL image
    image = Image.fromarray(input_array)

    # Compress the image using JPEG
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)

    # Read the compressed image back into a NumPy array
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    compressed_array = np.array(compressed_image).astype("float32") / 255

    return compressed_array


def load_images_from_folder(folder_path):
    """
    Loads all images from a given folder and converts them into NumPy arrays.

    :param folder_path: Path to the folder containing images.
    :return: A list of NumPy arrays representing the images.
    """
    images = []
    for filename in glob.glob(folder_path):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
        ):
            print(filename)
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                img_array = np.array(img).astype("float32") / 255
                images.append(img_array)

    return images


from math import ceil


def pad(img, patch_size=256):
    """
    Pads and patches a NumPy image into 256x256 square patches.

    :param img: NumPy array of the image.
    :param patch_size: Size of the patches, default is 256.
    :return: A NumPy array of image patches.
    """
    # Calculate padding sizes
    pad_height = patch_size - (img.shape[0] % patch_size)
    pad_width = patch_size - (img.shape[1] % patch_size)
    pad_height = pad_height if pad_height != patch_size else 0
    pad_width = pad_width if pad_width != patch_size else 0

    # Pad the image
    padded_img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant")

    return padded_img


# Example usage
# img = np.random.rand(500, 500, 3)  # Replace with your actual image
# patches = pad_and_patchify(img)
# print(patches.shape)  # This will print (2, 2, 256, 256, 3) for a 500x500x3 image


def original_msssim_func(img1: torch.Tensor, img2: torch.Tensor, _: BoundaryInfo):
    return pytorch_msssim.ms_ssim(img1, img2, 1.0)


@torch.no_grad()
def compute_msssim(
    func: Callable[[torch.Tensor, torch.Tensor, BoundaryInfo], torch.Tensor],
    img1,
    img2,
    boundary,
):
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0)
    return func(img1.cuda(), img2.cuda(), boundary).cpu().numpy().item()


def R2(X, Y):
    X_mean = np.mean(X)
    SST = np.sum((X - X_mean) ** 2)
    SSR = np.sum((X - Y) ** 2)
    R2 = 1 - (SSR / SST)
    return R2


def plot(images, func, title):
    gt = []
    patched = []
    for i, img in enumerate(images):
        for q in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            compressed_img = compress_image_jpeg(img, q)
            gt_msssim = compute_msssim(original_msssim_func, img, compressed_img, None)

            for patch_size in [256]:
                padded_img = pad(img, patch_size)
                padded_compressed_img = pad(compressed_img, patch_size)

                h_patches = padded_img.shape[0] // patch_size
                w_patches = padded_compressed_img.shape[1] // patch_size

                patched_msssims = []

                for i in range(h_patches):
                    for j in range(w_patches):
                        x = padded_img[
                            i * patch_size : (i + 1) * patch_size,
                            j * patch_size : (j + 1) * patch_size,
                            :,
                        ]
                        y = padded_compressed_img[
                            i * patch_size : (i + 1) * patch_size,
                            j * patch_size : (j + 1) * patch_size,
                            :,
                        ]
                        top = i == 0
                        left = j == 0
                        bottom = i == (h_patches - 1)
                        right = j == (w_patches - 1)
                        patched_msssim = compute_msssim(
                            func, x, y, BoundaryInfo(top, left, bottom, right)
                        )
                        patched_msssims.append(patched_msssim)
                        del patched_msssim

                patched_msssim = np.asarray(patched_msssims).mean()
                # print("GT:", gt_msssim, "Patched:", patched_msssim)

                gt.append(-10 * math.log10(1.0 - gt_msssim))
                patched.append(-10 * math.log10(1.0 - patched_msssim))
        # break
    gt = np.asarray(gt)
    patched = np.asarray(patched)

    max = 40
    plt.figure(figsize=(12, 8))
    plt.plot([0.0, max], [0.0, max], color="red")
    plt.scatter(gt, patched)
    plt.title(title)
    plt.xlim(0, max)
    plt.ylim(0, max)
    plt.xlabel("GT")
    plt.ylabel("Patched")
    plt.savefig(os.path.join(ROOTDIR, title), dpi=300)
    plt.close()

    r2 = R2(gt, patched)
    print(f"{title}: R^2 = {r2:.8f}")

    mse = np.mean((gt - patched) ** 2)
    print(f"{title}: MSE = {mse:.8f}")


if __name__ == "__main__":
    images = load_images_from_folder(os.path.expanduser("~/dataset/kodak/*.png"))
    plot(images, original_msssim_func, "MS-SSIM")
    plot(images, lambda x, y, b: ba_ms_ssim(x, y, b, 1.0), "BA-MS-SSIM")
