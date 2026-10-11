"""ORT_DEVICE picks the onnxruntime providers; without onnxruntime-gpu it always stays on the CPU."""
import onnxruntime as ort

from pulse.perception.device import default_providers


def test_default_is_cpu(monkeypatch):
    monkeypatch.delenv("ORT_DEVICE", raising=False)
    assert default_providers() == ("CPUExecutionProvider",)


def test_cuda_unavailable_falls_back(monkeypatch):
    monkeypatch.setenv("ORT_DEVICE", "cuda")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert default_providers() == ("CPUExecutionProvider",)


def test_cuda_available(monkeypatch):
    monkeypatch.setenv("ORT_DEVICE", "cuda")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda *a, **k: None, raising=False)
    assert default_providers() == ("CUDAExecutionProvider", "CPUExecutionProvider")
