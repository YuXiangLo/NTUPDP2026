from __future__ import annotations

import numpy as np


def carve_seams_cpu(image: np.ndarray, num_seams: int, show_progress: bool = False) -> np.ndarray:
    if num_seams <= 0:
        return image
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")
    height, width = image.shape[:2]
    if num_seams >= width:
        raise ValueError("num_seams must be smaller than image width")

    carved = image.astype(np.float32, copy=False)
    
    indices = range(num_seams)
    if show_progress:
        from tqdm import tqdm
        indices = tqdm(indices, desc="Carving (CPU)", leave=False)
        
    for _ in indices:
        energy = _compute_energy(carved)
        seam = _find_vertical_seam(energy)
        carved = _remove_vertical_seam(carved, seam)
    return carved


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _compute_energy(image: np.ndarray) -> np.ndarray:
    gray = _to_grayscale(image)
    left = np.pad(gray, ((0, 0), (1, 0)), mode="edge")[:, :-1]
    right = np.pad(gray, ((0, 0), (0, 1)), mode="edge")[:, 1:]
    up = np.pad(gray, ((1, 0), (0, 0)), mode="edge")[:-1, :]
    down = np.pad(gray, ((0, 1), (0, 0)), mode="edge")[1:, :]
    dx = np.abs(right - left)
    dy = np.abs(down - up)
    return dx + dy


def _find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    height, width = energy.shape
    cost = energy.copy()
    backtrack = np.zeros((height, width), dtype=np.int32)

    for i in range(1, height):
        for j in range(width):
            left = max(j - 1, 0)
            right = min(j + 1, width - 1)
            prev_slice = cost[i - 1, left : right + 1]
            idx = int(np.argmin(prev_slice))
            backtrack[i, j] = left + idx
            cost[i, j] += prev_slice[idx]

    seam = np.zeros(height, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for i in range(height - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]
    return seam


def _remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=bool)
    mask[np.arange(height), seam] = False
    if image.ndim == 3:
        channels = image.shape[2]
        return image[mask].reshape(height, width - 1, channels)
    return image[mask].reshape(height, width - 1)
