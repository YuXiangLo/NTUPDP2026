from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

import cuda_versions  # noqa: F401
from seam_carving import carve_seams_cpu, carve_seams_torch, load_image, save_image
from seam_carving.cuda_registry import available_cuda_versions, get_cuda_implementation


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
    # Use tqdm for runs if show_progress is enabled
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
    print("Starting main...")
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
        "--cuda-version",
        default=None,
        help=f"CUDA implementation name ({', '.join(available_cuda_versions())}). If not specified, all versions are benchmarked.",
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
    torch_output: Optional[torch.Tensor] = None
    cuda_output: Optional[torch.Tensor] = None

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

    def torch_runner() -> torch.Tensor:
        nonlocal torch_output
        torch_output = carve_seams_torch(
            image_torch.clone(), args.seams, device=device, show_progress=args.progress
        )
        return torch_output

    torch_sync = torch.cuda.synchronize if device.type == "cuda" else None
    _measure("PyTorch", torch_runner, args.runs, args.warmup, sync=torch_sync, show_progress=args.progress)

    if not torch.cuda.is_available():
        print("CUDA        skipped (torch.cuda unavailable)")
    else:
        versions_to_run = [args.cuda_version] if args.cuda_version else available_cuda_versions()
        image_cuda = image_torch.to(device="cuda")

        for version in versions_to_run:
            cuda_impl = get_cuda_implementation(version)
            if cuda_impl is None:
                print(f"CUDA ({version:<7}) skipped (not registered)")
                continue

            def cuda_runner_factory(impl):
                def cuda_runner() -> torch.Tensor:
                    nonlocal cuda_output
                    cuda_output = impl.carve(image_cuda.clone(), args.seams, show_progress=args.progress)
                    return cuda_output
                return cuda_runner

            try:
                _measure(
                    f"CUDA ({version})",
                    cuda_runner_factory(cuda_impl),
                    args.runs,
                    args.warmup,
                    sync=torch.cuda.synchronize,
                    show_progress=args.progress,
                )
            except NotImplementedError as exc:
                print(f"CUDA ({version:<7}) skipped ({exc})")

    if args.save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        if cpu_output is not None:
            save_image(output_dir / "cpu.png", cpu_output)
        if torch_output is not None:
            save_image(output_dir / "pytorch.png", torch_output.detach().cpu().numpy())
        if cuda_output is not None:
            save_image(output_dir / "cuda.png", cuda_output.detach().cpu().numpy())
        print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
