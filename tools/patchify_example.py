import os
from PIL import Image


def split_image_into_blocks(image_path, output_folder, block_size=512):
    # Open the input image
    image = Image.open(image_path)
    image_width, image_height = image.size
    os.makedirs(output_folder, exist_ok=True)

    # Calculate the number of blocks needed
    num_blocks_x = image_width // block_size
    num_blocks_y = image_height // block_size

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the image into blocks and save each block
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            left = x * block_size
            upper = y * block_size
            right = left + block_size
            lower = upper + block_size

            # Crop the image to the current block
            block = image.crop((left, upper, right, lower))

            # Save the block to the output folder
            block_filename = os.path.join(output_folder, f"block_{y}_{x}.png")
            block.save(block_filename)

    print(
        f"Image split into {num_blocks_x * num_blocks_y} blocks of size {block_size}x{block_size} and saved to {output_folder}"
    )


# Example usage
input_image_path = "images/6720x4480/IMG_6726.png"
output_blocks_folder = "images/IMG_6726_blocks_512x512"
split_image_into_blocks(input_image_path, output_blocks_folder)
