"""Runtime helpers for seeds, devices, directories, and JSON serialization."""

from __future__ import annotations

import json
import platform
import random
import sys
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    """Choose an explicit device, falling back to CPU when unavailable."""
    if device is not None:
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "Requested device 'cuda' is unavailable; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for an output path if needed."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def to_serializable(value: Any) -> Any:
    """Convert tensors, arrays, and paths into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON payload with stable formatting."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _package_version(package_name: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_environment_metadata(
    device: torch.device | str,
    requested_device: str | None = None,
) -> dict[str, object]:
    """Collect lightweight environment metadata for reproducible experiment outputs."""
    resolved_device = torch.device(device)
    cuda_available = torch.cuda.is_available()
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_index: int | None = None
    cuda_device_name: str | None = None
    if cuda_available:
        try:
            if resolved_device.type == "cuda":
                cuda_device_index = int(resolved_device.index) if resolved_device.index is not None else 0
            else:
                cuda_device_index = 0
            cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))
        except Exception:
            cuda_device_index = 0
            try:
                cuda_device_name = str(torch.cuda.get_device_name(0))
            except Exception:
                cuda_device_name = None

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "device": str(resolved_device),
        "requested_device": None if requested_device is None else str(requested_device),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_index": cuda_device_index,
        "cuda_device_name": cuda_device_name,
        "device_resolution": {
            "requested": None if requested_device is None else str(requested_device),
            "resolved": str(resolved_device),
            "used_fallback": requested_device is not None and str(requested_device) != str(resolved_device),
        },
        "packages": {
            "matplotlib": _package_version("matplotlib"),
            "numpy": _package_version("numpy"),
            "pot": _package_version("pot"),
            "pyvene": _package_version("pyvene"),
            "scipy": _package_version("scipy"),
            "torch": _package_version("torch"),
            "tqdm": _package_version("tqdm"),
        },
    }
