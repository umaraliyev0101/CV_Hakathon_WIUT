# DATA_SOURCES fragment — real open footage + detector weights

Fragment to merge into the project DATA_SOURCES file. All clips were downloaded 2026-10-11 through
`https://www.pexels.com/download/video/<id>/`, which redirects to `videos.pexels.com` with no login.
Licence for every clip: **Pexels License** (https://www.pexels.com/license/): free to use, no attribution
required (credit given anyway), no account. Clips are kept in `data/real_open/`, which is gitignored
through `data/`.

**Purpose: demo and evaluation only. Do NOT train on these clips.**

| Pexels id | page | author | resolution | fps | duration | local path | sha256 |
|---|---|---|---|---|---|---|---|
| 7154214 | https://www.pexels.com/video/salesman-talking-to-a-customer-7154214/ | Antoni Shkraba | 1920x1080 | 25 | 11.6 s (289 fr) | `data/real_open/pexels_7154214_1920_1080_25fps.mp4` | `dc4e30ee6987bb4eb0e25a467024451bbd4c5d88a8300840d685df448e579e9e` |
| 7154217 | https://www.pexels.com/video/couple-talking-to-a-car-dealer-7154217/ | Antoni Shkraba | 1920x1080 | 25 | 12.5 s (312 fr) | `data/real_open/pexels_7154217_1920_1080_25fps.mp4` | `4baeb2b04f383ffe7f3d393ba4083b82b203ce0b56d641b039032e2dfc645a2a` |
| 7154241 | https://www.pexels.com/video/salesman-talking-to-a-customer-7154241/ | Antoni Shkraba | 1066x1920 (vertical) | 25 | 17.0 s (425 fr) | `data/real_open/pexels_7154241_1066_1920_25fps.mp4` | `8013de02df9dfebf8df842d4087fc5f55806d4502e4081ea7386b993bc86fc74` |
| 36512656 | https://www.pexels.com/video/businesswoman-entering-car-dealership-building-36512656/ | cherishma reddy | 2160x3840 (4K vertical) | 50 | 9.6 s (480 fr) | `data/real_open/pexels_36512656_2160_3840_50fps.mp4` | `fc6b0fc038345178385d5c7b0eea824694f04e6df067a7c8484e90e6d516406a` |

Notes:
- 36512656: the download endpoint only offers the 4K rendition. Guessed 1080p/720p CDN paths returned 403.
  It is only 13.5 MB, so we kept the 4K file. Downscale at read time if needed.
- Author names come from the Pexels pages. The pages are behind a Cloudflare check for curl, so we read them
  with a web fetcher.
- Duration and resolution were measured with OpenCV on the downloaded files.

## Detector weights

| item | source | licence |
|---|---|---|
| RF-DETR Nano, COCO-pretrained (`rf-detr-nano.pth`) | `pip install rfdetr==1.11.2`; weights auto-downloaded from https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth (md5 `fb6504cce7fbdc783f7a46991f07639f`) | Apache-2.0 (Roboflow RF-DETR, https://github.com/roboflow/rf-detr). Trained on COCO 2017 (annotations CC BY 4.0, images under Flickr terms) |
| `models/showroom_detector.onnx` | derived: exported with `models/export_onnx.py` (opset 17, onnxsim) | Apache-2.0 (derived from the above) |

No ultralytics/AGPL code or weights are used.
