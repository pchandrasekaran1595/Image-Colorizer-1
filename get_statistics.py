import os
import sys
import numpy as np

from CLI.utils import breaker

def main():
    args: tuple = ("--path", "-p")
    path: str = "data"
    if args in sys.argv: path = sys.argv[sys.argv.index(args) + 1]

    assert "rgb_images.npy" in os.listdir(path) and "gray_images.npy" in os.listdir(path), "Please run python np_make.py"

    rgb_images = np.load(os.path.join(path, "rgb_images.npy"))
    gray_images = np.load(os.path.join(path, "gray_images.npy"))


    breaker()
    print("RGB Images\n\n")
    print("Mean\n")
    print(f"Red Channel Mean   : {rgb_images[:, :, 0].mean() / 255}")
    print(f"Green Channel Mean : {rgb_images[:, :, 1].mean() / 255}")
    print(f"Blue Channel Mean  : {rgb_images[:, :, 2].mean() / 255}")
    breaker()
    print("Standard Deviation\n")
    print(f"Red Channel Std   : {rgb_images[:, :, 0].std() / 255}")
    print(f"Green Channel Std : {rgb_images[:, :, 1].std() / 255}")
    print(f"Blue Channel Std  : {rgb_images[:, :, 2].std() / 255}")

    breaker()
    print("Gray Images\n\n")
    print("Mean\n")
    print(f"Red Channel Mean   : {gray_images[:, :, 0].mean() / 255}")
    print(f"Green Channel Mean : {gray_images[:, :, 1].mean() / 255}")
    print(f"Blue Channel Mean  : {gray_images[:, :, 2].mean() / 255}")

    breaker()
    print("Standard Deviation\n")
    print(f"Red Channel Std   : {gray_images[:, :, 0].std() / 255}")
    print(f"Green Channel Std : {gray_images[:, :, 1].std() / 255}")
    print(f"Blue Channel Std  : {gray_images[:, :, 2].std() / 255}")

    breaker()

if __name__ == "__main__":
    sys.exit(main() or 0)