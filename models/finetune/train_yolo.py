"""Benchmark only (Ultralytics is AGPL-3.0, not for the product): fine-tune COCO YOLO11n on the same data.

Usage from repo root: CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/finetune/train_yolo.py
"""

from ultralytics import YOLO

if __name__ == "__main__":
    YOLO("data/finetune_runs/yolo11n.pt").train(data="data/finetune_yolo/data.yaml", epochs=15, imgsz=640, batch=32, device=0,
                             workers=8, project="data/finetune_runs", name="yolo11n_mix", exist_ok=True,
                             patience=5, plots=False, verbose=False)
