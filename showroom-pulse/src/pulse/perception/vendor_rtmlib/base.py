# Vendored from rtmlib 0.0.16 (Apache-2.0); modified: imports, onnxruntime-only base.
# Original copyright: OpenMMLab / rtmlib authors (see LICENSE in this folder).
"""Minimal onnxruntime-only BaseTool: takes a local model path, never downloads."""
import os
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import onnxruntime as ort


class BaseTool(metaclass=ABCMeta):

    def __init__(self, onnx_model: str, model_input_size: tuple = None,
                 mean: tuple = None, std: tuple = None,
                 providers=("CPUExecutionProvider",)):
        if not os.path.exists(onnx_model):
            raise FileNotFoundError(onnx_model)
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 1
        self.session = ort.InferenceSession(str(onnx_model), sess_options=so,
                                            providers=list(providers))
        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def inference(self, img: np.ndarray):
        """img: HxWx3 already normalised; returns list of output arrays."""
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        inp = img[None, :, :, :]
        names = [o.name for o in self.session.get_outputs()]
        return self.session.run(names, {self.session.get_inputs()[0].name: inp})
