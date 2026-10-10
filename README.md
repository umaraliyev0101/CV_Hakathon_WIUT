# Showroom Pulse

Local customer-experience analytics for car showrooms: footfall, dwell per car, customer
journeys and unserved interest, from anonymous, aggregated camera data, with evidence-backed
recommendations.

Work in progress (full documentation and licence table come in Task 21).

## Licences

| Component | Licence | Trained on | Where used |
|---|---|---|---|
| YOLOX-nano (Megvii) | Apache-2.0 | COCO 2017 | Person/vehicle detector candidate and fallback (`models/yolox_nano`) |
| YOLOX-tiny (Megvii) | Apache-2.0 | COCO 2017 | Detector candidate (`models/yolox_tiny`) |
| YOLOX-s (Megvii) | Apache-2.0 | COCO 2017 | Default detector, chosen by the M0 benchmark (`models/yolox_s`) |
| YOLOX-m (Megvii) | Apache-2.0 | COCO 2017 | Detector candidate (`models/yolox_m`) |
| RF-DETR-N (Roboflow, ONNX via saifkhichi96/opendetect) | Apache-2.0 | COCO 2017 (Objects365 pretrain) | Detector candidate (`models/rfdetr_n`) |
| RF-DETR-S (Roboflow, ONNX via saifkhichi96/opendetect) | Apache-2.0 | COCO 2017 (Objects365 pretrain) | Detector candidate (`models/rfdetr_s`) |
| RTMPose-s body7 (OpenMMLab) | Apache-2.0 | Body7 (COCO, AIC, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18) | Default pose estimator (`models/rtmpose_s`) |
| RTMPose-m body7 (OpenMMLab) | Apache-2.0 | Body7 (COCO, AIC, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18) | Optional higher-accuracy pose (`models/rtmpose_m`) |
| DINOv2-small (Meta; ONNX export by Xenova) | Apache-2.0 | LVD-142M | Appearance embedding for identity merge (`models/dinov2_small`) |
| rtmlib 0.0.16 (subset, vendored) | Apache-2.0 | n/a (code) | `src/pulse/perception/vendor_rtmlib` |
| MEVA dataset (Kitware/IARPA; Corona et al., WACV 2021) | CC BY 4.0 | n/a (evaluation data) | Dev/test clips and KPF annotations, `data/meva` (not committed) |
| Stock clips "4th International Auto Show, Bangalore (2025)" 01/04/07 (Gpkp, Wikimedia Commons) | CC BY-SA 4.0 | n/a (demo data) | Showroom-like demo and detector car check (`data/stock`, not committed) |
| Stock clip "Largest BYD flagship store in Manila opens" (China News Service / cnsvideo, Wikimedia Commons) | CC BY 3.0 | n/a (demo data) | Showroom demo and detector car check (`data/stock`, not committed) |
