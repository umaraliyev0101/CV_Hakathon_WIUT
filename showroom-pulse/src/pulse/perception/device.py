"""onnxruntime provider choice for the whole pipeline (env ORT_DEVICE)."""
from __future__ import annotations

import os

import onnxruntime as ort


def default_providers() -> tuple[str, ...]:
    """CPU, or CUDA first when ORT_DEVICE=cuda and onnxruntime-gpu is installed (else CPU)."""
    if os.environ.get("ORT_DEVICE", "").strip().lower() == "cuda":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA/cuDNN from the nvidia-* pip wheels
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)
