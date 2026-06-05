from __future__ import annotations

import os
import sys

import torch

from seam_carving.cuda_registry import CudaSeamCarver, register_cuda

# Make the compiled extension (cuda_kernels/seam_cuda.so) importable regardless
# of cwd. Run `make` in cuda_kernels/ on the server to build it.
_KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cuda_kernels")
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

try:
    import seam_cuda as _seam_cuda  # type: ignore
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - depends on build env
    _seam_cuda = None
    _IMPORT_ERR = exc


@register_cuda("CUDA_v2")
class CUDAV2(CudaSeamCarver):
    """Fused-DP CUDA seam carver.

    Energy and seam removal stay in PyTorch (cheap, fully parallel). The DP
    that finds each vertical seam — the real bottleneck — is replaced by a
    single custom CUDA kernel launch per seam instead of ~height tiny kernels.
    """

    def carve(self, image: torch.Tensor, num_seams: int, show_progress: bool = False) -> torch.Tensor:
        if _seam_cuda is None:
            raise NotImplementedError(
                f"seam_cuda extension not built ({_IMPORT_ERR}). "
                f"Run `make` in cuda_kernels/."
            )
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
            indices = tqdm(indices, desc="Carving (CUDA_v2)", leave=False)

        for _ in indices:
            energy = self._compute_energy(carved)
            seam = _seam_cuda.find_seam(energy)
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
        return (dx + dy).contiguous()

    def _remove_vertical_seam(self, image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[:2]
        mask = torch.ones((height, width), dtype=torch.bool, device=image.device)
        mask[torch.arange(height, device=image.device), seam] = False
        if image.ndim == 3:
            channels = image.shape[2]
            return image[mask].view(height, width - 1, channels)
        return image[mask].view(height, width - 1)
