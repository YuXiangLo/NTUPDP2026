from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type

import torch


@dataclass
class CudaSeamCarver:
    name: str

    def carve(self, image: torch.Tensor, num_seams: int, show_progress: bool = False) -> torch.Tensor:
        raise NotImplementedError


_CUDA_REGISTRY: Dict[str, Type[CudaSeamCarver]] = {}


def register_cuda(name: str):
    def decorator(cls: Type[CudaSeamCarver]) -> Type[CudaSeamCarver]:
        _CUDA_REGISTRY[name] = cls
        return cls

    return decorator


def get_cuda_implementation(name: str) -> Optional[CudaSeamCarver]:
    impl = _CUDA_REGISTRY.get(name)
    if impl is None:
        return None
    return impl(name=name)


def available_cuda_versions() -> list[str]:
    return sorted(_CUDA_REGISTRY.keys())
