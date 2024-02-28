import json
import os
from pathlib import Path
from PIL import Image
import shutil
import glob

basedir = os.path.split(__file__)[0]

test_images = glob.glob(os.path.join(basedir, "../../images/6720x4480/*.png"))

def calculate_bpp(bin_path, img_path):
    # Get the size (in bytes) of binary file A
    file_A_size_bytes = os.path.getsize(bin_path)

    # Open and load image B using Pillow (PIL)
    image_B = Image.open(img_path)

    # Get the width and height (dimensions) of image B
    width, height = image_B.size

    # Calculate the total number of pixels in image B
    total_pixels = width * height

    # Calculate the number of bits in file A
    bits_in_A = file_A_size_bytes * 8

    # Calculate the ratio (bits in A divided by pixels in B)
    ratio = bits_in_A / total_pixels

    return ratio

def move_and_compress(input_path, output_path, quality=80, downsample_factor=4):
    try:
        # Open the image using Pillow
        img = Image.open(input_path)

        # Calculate the new dimensions after down-sampling
        width, height = img.size
        new_width = width // downsample_factor
        new_height = height // downsample_factor

        # Perform the down-sampling
        img = img.resize((new_width, new_height))

        # Save the down-sampled image with JPEG compression and the specified quality
        img.save(output_path, format='JPEG', quality=quality)

        print(f"Image down-sampled by {downsample_factor}x, compressed, and saved to {output_path} with quality {quality}")

    except Exception as e:
        print(f"Error: {str(e)}")

latex_images_dir = os.path.join(basedir, "images")
os.makedirs(latex_images_dir, exist_ok=True)

def add_image(filename):
    return r"\includegraphics[width=\linewidth]{"+filename+"}"

script = ""

valid_images = ["IMG_8176", "IMG_3515", "IMG_3799"]

for image_raw_name in test_images:
    basename = Path(image_raw_name).stem

    if basename in valid_images:
        continue

    image_new_name = os.path.join(latex_images_dir, f"{basename}.jpg")
    move_and_compress(image_raw_name, image_new_name)

    line0 = basename.replace("_", r"\_") + " & "
    line1 = "\centering Method & "
    s = add_image(f"images/{basename}.jpg")
    line2 = "\centering " + s + " & "
    line3 = "\centering Target bitrate & "
    
    for w_time in [0.000, 0.005, 0.010, 0.025, 0.050, 1.0]:
        folder = os.path.join(basedir, f"results/1000/100/30/{w_time}/0.01/False/0.2/512/")
        statistics_filename = os.path.join(folder, f"{basename}_statistics.json")
        bin_filename = os.path.join(folder, f"{basename}.bin")
        recon_filename = os.path.join(folder, f"{basename}_rec.png")
        method_filename = os.path.join(folder, f"{basename}_method.png")
        num_bytes_filename = os.path.join(folder, f"{basename}_num_bytes.png")

        new_recon_filename = os.path.join(latex_images_dir, f"example_{basename}_{w_time}_rec.jpg")
        new_method_filename = os.path.join(latex_images_dir, f"example_{basename}_{w_time}_method.jpg")
        new_num_bytes_filename = os.path.join(latex_images_dir, f"example_{basename}_{w_time}_num_bytes.jpg")

        if not os.path.isfile(bin_filename):
            PSNR = -1
            time = -1
            bpp = -1
        else:
            with open(statistics_filename, "r") as f:
                data = json.load(f)

            PSNR = float(data["gen_psnr"][-1])
            time = float(data["gen_time"][-1])
            bpp = float(calculate_bpp(bin_filename, recon_filename))

            move_and_compress(recon_filename, new_recon_filename)
            move_and_compress(method_filename, new_method_filename)
            move_and_compress(num_bytes_filename, new_num_bytes_filename)

        title = f"{bpp:.3f}/{PSNR:.2f}dB/{time*1000:.0f}ms"
        line0 += title + " & "
        line1 += add_image(f"images/example_{basename}_{w_time}_method.jpg") + " & "
        line2 += add_image(f"images/example_{basename}_{w_time}_rec.jpg") + " & "
        line3 += add_image(f"images/example_{basename}_{w_time}_num_bytes.jpg") + " & "
    
    line0 = line0[:-2] + r"\\" + '\n'
    line1 = line1[:-2] + r"\\" + '\n'
    line2 = line2[:-2] + r"\\" + '\n'
    line3 = line3[:-2] + r"\\" + '\n'

    script = script + line0 + line2 + line1 + line3 + "\n\\hline\n"

script = r"""\begin{table}[]
\begin{tabular}{ccccccc}
""" + script + r"""\end{tabular}
\end{table}
"""

with open(os.path.join(basedir, "script_supp.txt"), "w") as f:
    print(script, file=f)