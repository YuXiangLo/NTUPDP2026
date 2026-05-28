from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def save_image(path: str | Path, image: np.ndarray) -> None:
    clipped = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(path)
