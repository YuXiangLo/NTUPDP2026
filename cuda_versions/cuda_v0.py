from __future__ import annotations

import torch

from seam_carving.cuda_registry import CudaSeamCarver, register_cuda


@register_cuda("CUDA_v0")
class CUDAV0(CudaSeamCarver):
    def carve(self, image: torch.Tensor, num_seams: int, show_progress: bool = False) -> torch.Tensor:
        if num_seams <= 0:
            return image
        if image.ndim not in (2, 3):
            raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

        # Ensure image is on CUDA
        carved = image.to(device="cuda", dtype=torch.float32)
        height, width = carved.shape[:2]
        if num_seams >= width:
            raise ValueError("num_seams must be smaller than image width")

        indices = range(num_seams)
        if show_progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc="Carving (CUDA_v0)", leave=False)

        for _ in indices:
            energy = self._compute_energy(carved)
            seam = self._find_vertical_seam(energy)
            carved = self._remove_vertical_seam(carved, seam)
        return carved

    def _to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[:, :, 0]
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _compute_energy(self, image: torch.Tensor) -> torch.Tensor:
        gray = self._to_grayscale(image)
        left = torch.cat([gray[:, :1], gray[:, :-1]], dim=1)
        right = torch.cat([gray[:, 1:], gray[:, -1:]], dim=1)
        up = torch.cat([gray[:1, :], gray[:-1, :]], dim=0)
        down = torch.cat([gray[1:, :], gray[-1:, :]], dim=0)
        dx = (right - left).abs()
        dy = (down - up).abs()
        return dx + dy

    def _find_vertical_seam(self, energy: torch.Tensor) -> torch.Tensor:
        height, width = energy.shape
        cost = energy.clone()
        backtrack = torch.zeros(
            (height, width), dtype=torch.int64, device=energy.device
        )

        for i in range(1, height):
            prev = cost[i - 1]
            left = torch.cat([prev[:1], prev[:-1]], dim=0)
            right = torch.cat([prev[1:], prev[-1:]], dim=0)
            stacked = torch.stack([left, prev, right], dim=0)
            min_vals, idxs = torch.min(stacked, dim=0)
            cost[i] = cost[i] + min_vals
            backtrack[i] = torch.arange(width, device=energy.device) + (idxs - 1)
            backtrack[i].clamp_(0, width - 1)

        seam = torch.zeros(height, dtype=torch.int64, device=energy.device)
        seam[-1] = torch.argmin(cost[-1])
        for i in range(height - 2, -1, -1):
            seam[i] = backtrack[i + 1, seam[i + 1]]
        return seam

    def _remove_vertical_seam(self, image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[:2]
        mask = torch.ones((height, width), dtype=torch.bool, device=image.device)
        mask[torch.arange(height, device=image.device), seam] = False
        if image.ndim == 3:
            channels = image.shape[2]
            return image[mask].view(height, width - 1, channels)
        return image[mask].view(height, width - 1)
