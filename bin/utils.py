import os
import tempfile
from PIL import Image
import subprocess
import numpy as np

def get_bpg_result(img_filename, qp=28):
    img_filename = os.path.abspath(img_filename)
    with tempfile.NamedTemporaryFile('w+b') as f_bin, tempfile.NamedTemporaryFile() as f_recon:
        bin_filename = f_bin.name
        recon_filename = f_recon.name
        enc_cmd = f'bpgenc -q {qp} {img_filename} -o {bin_filename}'
        dec_cmd = f'bpgdec {bin_filename} -o {recon_filename}'

        print("BPG encoding:", enc_cmd)
        print("BPG decoding:", dec_cmd)
        
        subprocess.run(enc_cmd.split())
        subprocess.run(dec_cmd.split())

        file_size_bytes = os.path.getsize(bin_filename)
        num_bits = file_size_bytes * 8

        img1 = np.array(Image.open(img_filename)).astype(np.int32)
        img2 = np.array(Image.open(recon_filename)).astype(np.int32)

        mse = np.mean((img1-img2) ** 2)
        psnr = -10*np.log10(mse / (255 ** 2))
    
    return num_bits, psnr

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