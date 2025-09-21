import numpy as np
from PIL import Image


def ensure_ndarray_rgb(img: np.ndarray) -> Image.Image:
    arr = img
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):  # noqa: PLR2004
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 3 and arr.shape[2] == 3:  # noqa: PLR2004
        rgb = arr[:, :, ::-1]
        return Image.fromarray(rgb, mode="RGB")

    if arr.ndim == 3 and arr.shape[2] == 4:  # noqa: PLR2004
        rgba = arr[:, :, [2, 1, 0, 3]]
        return Image.fromarray(rgba, mode="RGBA").convert("RGB")

    raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
