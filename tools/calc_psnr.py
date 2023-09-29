from PIL import Image
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", type=str)
    parser.add_argument("img2", type=str)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    img1 = args.img1
    img2 = args.img2

    img1 = np.array(Image.open(img1))
    img2 = np.array(Image.open(img2))

    print(img1.shape)

    mse = np.mean((img1-img2) ** 2)

    print(-10*np.log10(mse / (255 ** 2)))

if __name__ == "__main__":
    main()