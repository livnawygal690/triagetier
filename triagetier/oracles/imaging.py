from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    array: NDArray[np.uint8] = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)
