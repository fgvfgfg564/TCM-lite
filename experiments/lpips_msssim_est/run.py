from PIL import Image
from lpips.lpips import *
import torch
import pytorch_msssim
import numpy as np
from io import BytesIO
import os

def compress_image_jpeg(input_array, quality):
    """
    Compresses an image using JPEG compression.

    :param input_array: A NumPy array representing the image.
    :param quality: An integer from 0 to 100 indicating the JPEG quality.
    :return: A NumPy array of the compressed image.
    """
    # Ensure the array is in the range 0-255
    input_array = (input_array * 255).astype('uint8')

    # Convert the NumPy array to a PIL image
    image = Image.fromarray(input_array)

    # Compress the image using JPEG
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)

    # Read the compressed image back into a NumPy array
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    compressed_array = np.array(compressed_image).astype('float32') / 255

    return compressed_array

def load_images_from_folder(folder_path):
    """
    Loads all images from a given folder and converts them into NumPy arrays.

    :param folder_path: Path to the folder containing images.
    :return: A list of NumPy arrays representing the images.
    """
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                img_array = np.array(img).astype('float32') / 255
                images.append(img_array)

    return images

from math import ceil

def pad_and_patchify(img, patch_size=256):
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
    padded_img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    # Number of patches in each dimension
    num_patches_x = ceil(padded_img.shape[1] / patch_size)
    num_patches_y = ceil(padded_img.shape[0] / patch_size)

    # Initialize the array to hold the patches
    patches = np.zeros((num_patches_y, num_patches_x, patch_size, patch_size, img.shape[2]), dtype=img.dtype)

    # Fill in the patches
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patches[i, j] = padded_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

    return patches

# Example usage
# img = np.random.rand(500, 500, 3)  # Replace with your actual image
# patches = pad_and_patchify(img)
# print(patches.shape)  # This will print (2, 2, 256, 256, 3) for a 500x500x3 image


@torch.no_grad()
def compute_msssim(img1, img2):
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0)
    return pytorch_msssim.ms_ssim(img1, img2, 1.).numpy().item()

LPIPSNET = lpips.LPIPS()

@torch.no_grad()
def compute_lpips(img1, img2):
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0)
    return LPIPSNET(img1, img2, normalize=True)[0,0,0,0].numpy().item()

if __name__ == '__main__':
    images = load_images_from_folder(os.path.expanduser('~/dataset/kodak'))
    for img in images:
        for q in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
            compressed_img = compress_image_jpeg(img, q)
            gt_msssim = compute_msssim(img, compressed_img)
            gt_lpips = compute_lpips(img, compressed_img)
            print(gt_msssim, gt_lpips)

            for patch_size in [64, 128, 256, 512]:
                patched_img = pad_and_patchify(img, patch_size)
                patched_compressed_img = pad_and_patchify(compressed_img, patch_size)