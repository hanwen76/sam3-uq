from pathlib import Path

import numpy as np


def main() -> None:
    out = Path(__file__).resolve().parent
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 1] = 24
    image[18:44, 20:46, 0] = 180
    image[18:44, 20:46, 1] = 80

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18:44, 20:46] = 1
    np.save(out / "sample_image.npy", image)
    np.save(out / "sample_mask.npy", mask)


if __name__ == "__main__":
    main()
