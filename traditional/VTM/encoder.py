import subprocess

def compress_image(input_image, compressed_output):
    # Command to compress the image using VTM
    compress_command = [
        "vtmexe",  # Replace with the path to your VTM executable
        "--InputFile=" + input_image,
        "--OutputFile=" + compressed_output,
        "--Profile=main_10",
        "--QP=22",  # Adjust the quantization parameter as needed
        "--FramesToBeEncoded=1",
    ]

    # Run the compression command
    subprocess.run(compress_command)

def decompress_image(compressed_input, decompressed_output):
    # Command to decompress the image using VTM
    decompress_command = [
        "vtmexe",  # Replace with the path to your VTM executable
        "--BitstreamFile=" + compressed_input,
        "--ReconFile=" + decompressed_output,
    ]

    # Run the decompression command
    subprocess.run(decompress_command)

if __name__ == "__main__":
    input_image = "input.png"  # Replace with the path to your input image
    compressed_output = "compressed.bin"  # Output compressed file
    decompressed_output = "decompressed.png"  # Output decompressed image

    # Compress the image
    compress_image(input_image, compressed_output)

    # Decompress the compressed image
    decompress_image(compressed_output, decompressed_output)

    print("Compression and decompression completed.")