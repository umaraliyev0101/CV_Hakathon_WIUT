"""Export the fine-tuned Nano checkpoint to ONNX (same graph settings as models/export_onnx.py).

Usage from repo root:
    .venv/bin/python models/finetune/export_ft.py data/finetune_runs/nano_mix/checkpoint_best_ema.pth models/showroom_detector_ft.onnx
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.export_onnx import sha256  # noqa: E402

if __name__ == "__main__":
    from rfdetr import RFDETRNano

    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = RFDETRNano(pretrain_weights=str(ckpt))
    with tempfile.TemporaryDirectory(prefix="rfdetr_export_") as tmp:
        p = Path(model.export(output_dir=tmp, opset_version=17, batch_size=1, dynamic_batch=False, verbose=False))
        import onnx
        from onnxsim import simplify as onnx_simplify

        m2, ok = onnx_simplify(onnx.load(str(p)))
        if ok:
            onnx.save(m2, str(p))
        shutil.copyfile(p, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) sha256={sha256(out)}")
