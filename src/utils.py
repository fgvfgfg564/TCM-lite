from typing_extensions import List

import os
import tempfile
from PIL import Image
import subprocess
import numpy as np
import torch
from torchvision.transforms import ToTensor
import math
import pytorch_msssim
import hashlib
from dataclasses import dataclass
from .fileio import FileIO


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

        img1 = np.array(Image.open(img_filename)).astype(np.float32)
        img2 = np.array(Image.open(recon_filename)).astype(np.float32)

        mse = np.mean((img1 - img2) ** 2)
        psnr = -10 * np.log10(mse / (255**2))

    return num_bits, psnr


def make_strictly_increasing(arr):
    n = len(arr)
    min_bytes = None
    for t in range(n - 1, -1, -1):
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


def dump_image(img: np.ndarray):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).detach().cpu().numpy()
    if img.dtype != np.uint8:
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
        return float("inf")
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr


def msssim_with_file(image_path1, image_path2):
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
    image1 = ToTensor()(image1).unsqueeze(0)
    image2 = ToTensor()(image2).unsqueeze(0)

    # Calculate the mean squared error (MSE)
    msssim = pytorch_msssim.ms_ssim(image1, image2, 1.0).detach().cpu().numpy().item()
    return msssim


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


def hash_numpy_array(array, hash_function="sha256"):
    """
    Computes the hash of a NumPy array.

    :param array: NumPy array to be hashed.
    :param hash_function: Name of the hash function to use (e.g., 'md5', 'sha1', 'sha256').
    :return: Hexadecimal hash of the array.
    """
    if hash_function not in hashlib.algorithms_available:
        raise ValueError(f"Hash function {hash_function} is not available.")

    # Convert the array to bytes
    array_bytes = array.tobytes()

    # Create a hash object and update it with the array bytes
    hash_obj = hashlib.new(hash_function)
    hash_obj.update(array_bytes)

    # Return the hexadecimal digest of the hash
    return hash_obj.hexdigest()


def hash_list_of_arrays(array_list, hash_function="sha256"):
    """
    Computes the hash of a list of NumPy arrays.

    :param array_list: List of NumPy arrays to be hashed.
    :param hash_function: Name of the hash function to use.
    :return: Hexadecimal hash of the list of arrays.
    """
    combined_hash = hashlib.new(hash_function)

    for array in array_list:
        array_hash = hash_numpy_array(array, hash_function)
        combined_hash.update(array_hash.encode())

    return combined_hash.hexdigest()


def torch_to_uint8(x):
    """
    将输入的Tensor从[0, 1]的浮点型归一化到[0, 255]的uint8类型。

    Args:
        x (torch.Tensor): 输入的Tensor，数据类型为float32，值域在[0, 1]之间。

    Returns:
        torch.Tensor: 转换后的Tensor，数据类型为uint8，值域在[0, 255]之间。

    """
    x = torch.clamp(x, 0, 1)
    x *= 255
    x = torch.round(x)
    x = x.to(torch.uint8)
    return x


def torch_float_to_np_uint8(x):
    """
    将PyTorch的FloatTensor类型的张量转换为NumPy的uint8类型的数组。

    Args:
        x (torch.FloatTensor): 输入的FloatTensor类型的张量，其形状应为(1, C, H, W)。

    Returns:
        np.ndarray: 转换后的NumPy uint8类型的数组，其形状为(H, W, C)。

    """
    x = torch_to_uint8(x)
    x = x[0].permute(1, 2, 0).detach().cpu().numpy()
    return x


def torch_pseudo_quantize_to_uint8(x):
    """
    将输入张量 x 进行伪量化到 uint8，然后再转换回float。

    Args:
        x (torch.Tensor): 待伪量化的输入张量。

    Returns:
        torch.Tensor: 伪量化到 uint8 后的张量，数据类型与输入张量相同。

    """
    dtype = x.dtype
    x = torch_to_uint8(x)
    x = x.to(dtype) / 255.0
    return x


@dataclass
class ImageBlock:
    np: np.ndarray  # 0 - 255
    cuda: torch.Tensor  # 0 - 1


def divide_blocks(fileio: FileIO, h, w, img: np.ndarray, dtype) -> List[ImageBlock]:
    """
    将图片按照CTU大小划分为多个图像块，并返回包含每个图像块信息的列表。

    Args:
        fileio (FileIO): 包含CTU大小和索引信息的文件IO对象。
        h (int): 图片的高度。
        w (int): 图片的宽度。
        img (np.ndarray): 待划分的图像数据。形状为[h, w, c]
        dtype (torch.dtype): 图像块数据的数据类型。

    Returns:
        List[ImageBlock]: 包含每个图像块信息的列表，每个元素为ImageBlock类型。

    """
    blocks = []
    for i in range(fileio.n_ctu):
        upper, left, lower, right = fileio.block_indexes[i]

        # Move to CUDA
        img_patch_np = img[upper:lower, left:right, :]
        img_patch_cuda = (
            torch.from_numpy(img_patch_np).permute(2, 0, 1).type(dtype) / 255.0
        )
        img_patch_cuda = img_patch_cuda.unsqueeze(0)
        img_patch_cuda = img_patch_cuda.cuda()

        blocks.append(ImageBlock(img_patch_np, img_patch_cuda))

    return blocks


def join_blocks(decoded_ctus: List[np.ndarray], file_io: FileIO):
    h = file_io.h
    w = file_io.w
    recon_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, ctu in enumerate(decoded_ctus):
        upper, left, lower, right = file_io.block_indexes[i]

        recon_img[upper:lower, left:right, :] = ctu[: lower - upper, : right - left, :]
    return recon_img
