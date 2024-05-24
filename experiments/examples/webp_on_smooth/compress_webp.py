from PIL import Image
import sys


def compress_image_to_webp(input_image_path, output_image_path, quality=20):
    """
    Compresses an image and saves it in WebP format.

    :param input_image_path: Path to the input image file.
    :param output_image_path: Path to save the compressed WebP image.
    :param quality: Quality setting for the WebP compression (0-100).
    """
    try:
        # Open the input image
        image = Image.open(input_image_path)

        # Save the image in WebP format with the specified quality
        image.save(output_image_path, "webp", quality=quality)

        print(f"Image successfully compressed and saved as {output_image_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python compress_image.py <input_image_path> <output_image_path> <quality>"
        )
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        quality = int(sys.argv[3])

        compress_image_to_webp(input_image_path, output_image_path, quality)
