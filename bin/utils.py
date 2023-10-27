import os
import tempfile
from PIL import Image
import subprocess
import numpy as np
import torch
import math

def stable_softmax(x: np.ndarray):
    x -= x.max()
    x = np.maximum(x, -75)
    x = np.exp(x)
    x /= x.sum()
    return x

def get_bpg_result(img_filename, qp=28):
    img_filename = os.path.abspath(img_filename)
    with tempfile.NamedTemporaryFile(
        "w+b"
    ) as f_bin, tempfile.NamedTemporaryFile() as f_recon:
        bin_filename = f_bin.name
        recon_filename = f_recon.name
        enc_cmd = f"bpgenc -q {qp} {img_filename} -o {bin_filename}"
        dec_cmd = f"bpgdec {bin_filename} -o {recon_filename}"

        print("BPG encoding:", enc_cmd)
        print("BPG decoding:", dec_cmd)

        subprocess.run(enc_cmd.split())
        subprocess.run(dec_cmd.split())

        file_size_bytes = os.path.getsize(bin_filename)
        num_bits = file_size_bytes

        img1 = np.array(Image.open(img_filename)).astype(np.int32)
        img2 = np.array(Image.open(recon_filename)).astype(np.int32)

        mse = np.mean((img1 - img2) ** 2)
        psnr = -10 * np.log10(mse / (255**2))

    return num_bits, psnr

def make_strictly_increasing(arr):     
    n = len(arr)
    min_bytes = None
    for t in range(n-1, -1, -1):
        if min_bytes is None or min_bytes > arr[t]:
            min_bytes = arr[t]
        else:
            arr[t] = min_bytes
        min_bytes -= 1

def is_strictly_increasing(arr):
    """
    Check if a 1-D NumPy array is strictly increasing.

    Parameters:
    arr (numpy.ndarray): The input 1-D NumPy array.

    Returns:
    bool: True if the array is strictly increasing, False otherwise.
    """
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            return False
    return True

def dump_torch_image(img):
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    return img

def psnr_with_file(image_path1, image_path2):
    """
    Calculate the PSNR of two image files
    """
    # Open and load the images using PIL
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Ensure the images have the same dimensions
    if image1.size != image2.size:
        raise ValueError("Both images must have the same dimensions.")

    # Convert the PIL images to NumPy arrays
    image1 = np.array(image1).astype(np.float32)
    image2 = np.array(image2).astype(np.float32)

    # Calculate the mean squared error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # Calculate the maximum possible pixel value
    max_pixel_value = 255  # Assuming 8-bit images

    # Calculate PSNR using the formula: PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr

def get_image_dimensions(image_path):
    """
    Read the height and width of an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple (width, height) representing the image's dimensions.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count