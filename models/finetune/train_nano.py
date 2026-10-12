"""Fine-tune COCO RF-DETR Nano on person boxes: reviewed real Pexels frames + synthetic CCTV frames.

Dataset: data/finetune/ (built from data/real_labels/coco and data/synthetic/coco; real train repeated
3x, every 4th synthetic frame). Usage from repo root:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/finetune/train_nano.py
"""

from rfdetr import RFDETRNano

if __name__ == "__main__":
    model = RFDETRNano()
    model.train(dataset_dir="data/finetune", output_dir="data/finetune_runs/nano_mix", epochs=20, batch_size=16,
                grad_accum_steps=1, lr=1e-4, num_workers=8, early_stopping=True, early_stopping_patience=5,
                class_names=["person"], tensorboard=False, progress_bar=None)
