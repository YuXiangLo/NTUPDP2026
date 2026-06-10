from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from seam_carving import carve_seams_cpu, carve_seams_torch, load_image, save_image

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CUDA_BINARY = ROOT_DIR / "build" / "cuda"


def _resolve_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return nvcc

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError("nvcc not found on PATH and CUDA_HOME/bin/nvcc is unavailable")


def _ensure_cuda_binary(binary_path: Path, source_path: Optional[Path]) -> Path:
    if binary_path.exists() and (source_path is None or binary_path.stat().st_mtime >= source_path.stat().st_mtime):
        return binary_path

    if source_path is None:
        raise FileNotFoundError("CUDA source is required when the binary is missing or out of date")
    if not source_path.exists():
        raise FileNotFoundError(f"Missing CUDA source: {source_path}")

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    nvcc = _resolve_nvcc()
    subprocess.run(
        [nvcc, "-O2", "-std=c++17", str(source_path), "-o", str(binary_path)],
        check=True,
    )
    return binary_path


def _write_raw_image(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.float32, order="C")
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    elif image.ndim == 3:
        height, width, channels = image.shape
    else:
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

    header = np.array([height, width, channels], dtype=np.int32)
    with path.open("wb") as handle:
        header.tofile(handle)
        image.reshape(-1).tofile(handle)


def _read_raw_image(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=3)
        if header.size != 3:
            raise ValueError(f"Invalid CUDA output header in {path}")
        height, width, channels = (int(v) for v in header)
        data = np.fromfile(handle, dtype=np.float32)

    expected = height * width * channels
    if data.size != expected:
        raise ValueError(f"Expected {expected} floats in {path}, found {data.size}")

    if channels == 1:
        return data.reshape(height, width)
    return data.reshape(height, width, channels)


def _run_cuda(binary_path: Path, image: np.ndarray, num_seams: int, show_progress: bool) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="cuda_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.bin"
        output_path = tmpdir_path / "output.bin"
        _write_raw_image(input_path, image)

        command = [
            str(binary_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seams",
            str(num_seams),
        ]
        if show_progress:
            command.append("--progress")

        subprocess.run(command, check=True)
        return _read_raw_image(output_path)


def _measure(
    label: str,
    runner: Callable[[], object],
    runs: int,
    warmup: int,
    sync: Optional[Callable[[], None]] = None,
    show_progress: bool = False,
) -> tuple[float, float]:
    from tqdm import tqdm

    if warmup > 0:
        for _ in tqdm(range(warmup), desc=f"{label:<12} (warmup)", leave=False, disable=not show_progress):
            runner()
            if sync:
                sync()

    times = []
    for _ in tqdm(range(runs), desc=f"{label:<12} (runs)", disable=not show_progress):
        start = time.perf_counter()
        runner()
        if sync:
            sync()
        times.append(time.perf_counter() - start)

    mean = statistics.mean(times)
    std = statistics.pstdev(times)
    print(f"{label:<12} mean={mean:.6f}s  std={std:.6f}s  runs={runs}")
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark seam carving variants.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--seams", type=int, default=10, help="Number of seams to remove.")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs.")
    parser.add_argument(
        "--torch-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for the PyTorch baseline.",
    )
    parser.add_argument(
        "--cuda-binary",
        type=Path,
        default=DEFAULT_CUDA_BINARY,
        help="Path to the compiled CUDA executable.",
    )
    parser.add_argument(
        "--cuda-source",
        type=Path,
        help="Path to the CUDA source file used when compiling the executable.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save output images to --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for saving outputs.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars for runs and seam removal.",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Skip the CPU benchmark.",
    )
    args = parser.parse_args()

    image_np = load_image(args.image)
    output_dir = Path(args.output_dir)

    print("Running seam carving benchmark")
    print(f"Image: {args.image}  Seams: {args.seams}")
    print("")

    cpu_output: Optional[np.ndarray] = None
    torch_output: Optional[np.ndarray] = None
    cuda_output: Optional[np.ndarray] = None

    if not args.skip_cpu:

        def cpu_runner() -> np.ndarray:
            nonlocal cpu_output
            cpu_output = carve_seams_cpu(image_np.copy(), args.seams, show_progress=args.progress)
            return cpu_output

        _measure("CPU", cpu_runner, args.runs, args.warmup, show_progress=args.progress)
    else:
        print("CPU         skipped (--skip-cpu)")

    device = torch.device(args.torch_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available but --torch-device=cuda was set")

    image_torch = torch.from_numpy(image_np)

    def torch_runner() -> np.ndarray:
        nonlocal torch_output
        torch_output = carve_seams_torch(
            image_torch.clone(), args.seams, device=device, show_progress=args.progress
        ).detach().cpu().numpy()
        return torch_output

    torch_sync = torch.cuda.synchronize if device.type == "cuda" else None
    _measure("PyTorch", torch_runner, args.runs, args.warmup, sync=torch_sync, show_progress=args.progress)

    binary_path = _ensure_cuda_binary(args.cuda_binary, args.cuda_source)

    def cuda_runner() -> np.ndarray:
        nonlocal cuda_output
        cuda_output = _run_cuda(binary_path, image_np, args.seams, args.progress)
        return cuda_output

    _measure("CUDA", cuda_runner, args.runs, args.warmup, show_progress=args.progress)

    if args.save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        if cpu_output is not None:
            save_image(output_dir / "cpu.png", cpu_output)
        if torch_output is not None:
            save_image(output_dir / "pytorch.png", torch_output)
        if cuda_output is not None:
            save_image(output_dir / "cuda.png", cuda_output)
        print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
