"""ORT_DEVICE picks the onnxruntime providers; without onnxruntime-gpu it always stays on the CPU."""
import onnxruntime as ort

from showroom_coach.perception import rtm_onnx as R


def test_default_is_cpu(monkeypatch):
    monkeypatch.delenv("ORT_DEVICE", raising=False)
    assert R.providers() == ["CPUExecutionProvider"]


def test_cuda_requested(monkeypatch):
    monkeypatch.setenv("ORT_DEVICE", " CUDA ")
    expected = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"])
    assert R.providers() == expected


def test_cuda_unavailable_falls_back(monkeypatch):
    monkeypatch.setenv("ORT_DEVICE", "cuda")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert R.providers() == ["CPUExecutionProvider"]
