from __future__ import annotations

import torch

from seam_carving.cuda_registry import CudaSeamCarver, register_cuda


@register_cuda("CUDA_v1")
class CUDAV1(CudaSeamCarver):
    """
    CUDA_v1: Basic warp tiling improvements (simulated via PyTorch optimizations).
    
    In a raw CUDA implementation, "warp tiling" would involve:
    1. Using shared memory to load tiles of the image/cost matrix.
    2. Using warp shuffles to find the minimum of neighboring costs.
    3. Processing multiple rows/columns per thread block to maximize reuse.
    
    This PyTorch version uses optimized kernels (conv2d) and sliding window
    operations (unfold) that leverage similar hardware-level tiling.
    """

    def carve(self, image: torch.Tensor, num_seams: int, show_progress: bool = False) -> torch.Tensor:
        if num_seams <= 0:
            return image
        if image.ndim not in (2, 3):
            raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

        carved = image.to(device="cuda", dtype=torch.float32)
        height, width = carved.shape[:2]
        if num_seams >= width:
            raise ValueError("num_seams must be smaller than image width")

        indices = range(num_seams)
        if show_progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc="Carving (CUDA_v1)", leave=False)

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
        # Weighted sum for grayscale
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        return (image * weights).sum(dim=2)

    def _compute_energy(self, image: torch.Tensor) -> torch.Tensor:
        gray = self._to_grayscale(image)
        h, w = gray.shape
        
        # Using 2D convolution for energy calculation.
        # Conv2D is highly optimized with tiling and shared memory in cuDNN.
        gray_4d = gray.view(1, 1, h, w)
        
        # Scharr/Sobel-like gradient kernels (simple difference for now to match v0)
        kernel_x = torch.tensor([[[-1, 0, 1]]], dtype=torch.float32, device=image.device)
        kernel_y = torch.tensor([[[-1], [0], [1]]], dtype=torch.float32, device=image.device)
        
        dx = torch.nn.functional.conv2d(
            torch.nn.functional.pad(gray_4d, (1, 1, 0, 0), mode="replicate"), 
            kernel_x.view(1, 1, 1, 3)
        ).abs()
        dy = torch.nn.functional.conv2d(
            torch.nn.functional.pad(gray_4d, (0, 0, 1, 1), mode="replicate"), 
            kernel_y.view(1, 1, 3, 1)
        ).abs()
        
        return (dx + dy).view(h, w)

    def _find_vertical_seam(self, energy: torch.Tensor) -> torch.Tensor:
        height, width = energy.shape
        cost = energy.clone()
        backtrack = torch.zeros((height, width), dtype=torch.int64, device=energy.device)

        # Pre-allocate range for backtrack calculation
        indices = torch.arange(width, device=energy.device)

        for i in range(1, height):
            prev = cost[i - 1]
            
            # Use unfold to create a sliding window of size 3 (left, center, right)
            # This mimics a "tiled" approach where each element looks at its neighbors.
            # Manual padding for 1D tensor as torch.nn.functional.pad(mode='replicate') 
            # requires at least a 3D tensor.
            padded = torch.cat([prev[:1], prev, prev[-1:]])
            windows = padded.unfold(0, 3, 1) # Shape: (width, 3)
            
            min_vals, idxs = torch.min(windows, dim=1)
            cost[i] += min_vals
            
            # idxs is in [0, 1, 2] -> map to offset [-1, 0, 1]
            backtrack[i] = indices + (idxs - 1)
            backtrack[i].clamp_(0, width - 1)

        seam = torch.zeros(height, dtype=torch.int64, device=energy.device)
        seam[-1] = torch.argmin(cost[-1])
        for i in range(height - 2, -1, -1):
            seam[i] = backtrack[i + 1, seam[i + 1]]
        return seam

    def _remove_vertical_seam(self, image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[:2]
        # Use a boolean mask to remove the seam efficiently
        mask = torch.ones((height, width), dtype=torch.bool, device=image.device)
        mask[torch.arange(height, device=image.device), seam] = False
        
        if image.ndim == 3:
            channels = image.shape[2]
            return image[mask].view(height, width - 1, channels)
        return image[mask].view(height, width - 1)
