# Data sources and licences

Everything used by this repo's synthetic showroom pipeline (`sim/`), the baseline detector (`models/`)
and the demo/evaluation footage. The per-file list with URLs is `assets/MANIFEST.csv` (160 rows,
checked by the `qa_assets` gate). `assets/`, `data/` and model weights are **never committed**
(`.gitignore`); raw third-party files stay on the machine. No accounts were created, nothing was paid
for, nothing was uploaded.

## 1. 3D assets used to render the synthetic data

| What | Files | Source | Author | Licence |
|---|---|---|---|---|
| 12 rigged human avatars (FBX + 2k textures) | `assets/people/characters/<Name>/` | https://github.com/microsoft/Microsoft-Rocketbox | Microsoft / Rocketbox Studios | **MIT** (c) 2020 Microsoft (`assets/people/ROCKETBOX_LICENSE.md`) |
| 31 motion-capture clips (walk, idle, talk, listen, look around, turns, wave, presenting) | `assets/people/anims/*.fbx` | same repo (`Animations/`) | Microsoft / Rocketbox Studios | **MIT** |
| Retargeted actions (derived: baked onto the avatars' A-pose rest) | `assets/people/anims/rocketbox_actions_retargeted.blend` | made here | derived from the above | MIT (derived) |
| Ferrari 458 Italia | `assets/cars/ferrari_458.glb` | three.js examples (`examples/models/gltf/ferrari.glb`) | vicent091036 (Sketchfab) | **CC BY 4.0** — credit: "Ferrari 458 Italia" by vicent091036 |
| Car Concept | `assets/cars/khronos_carconcept.glb` | KhronosGroup/glTF-Sample-Assets `Models/CarConcept` | Eric Chadwick, Darmstadt Graphics Group | **CC BY 4.0** |
| Lamborghini Urus | `assets/cars/lamborghini_urus.glb` | pmndrs/examples (`building-dynamic-envmaps/src/lambo.glb`) | Steven Grey (Sketchfab) | **CC BY-NC 4.0** — non-commercial; fine for this hackathon demo, must be replaced for any commercial use |
| Porsche 911 (930) Turbo | `assets/cars/porsche_930_turbo.glb` | pmndrs/examples (`envmap-ground-projection/src/porsche-transformed.glb`) | believed Lionsharp Studios (Sketchfab) | **unknown** (credit not embedded; believed CC BY 4.0) — demo only, replace before any other use |
| HDRIs `wide_street_01`, `urban_street_01`, `sunset_jhbcentral` (2k) | `assets/hdri/` | https://polyhaven.com | Poly Haven artists (see manifest) | **CC0** |
| Textures `marble_01`, `tiled_floor_001`, `beige_wall_001`, `concrete_floor_worn_001` (2k) | `assets/textures/` | https://polyhaven.com | Poly Haven artists (see manifest) | **CC0** |

Mixamo was not used (it needs an Adobe account). No SMPL/AMASS, no CrowdHuman, no ultralytics.
Sketchfab CC BY attributions above must be kept wherever renders are shown.

## 2. How the synthetic data was generated

`sim/gen_configs.py` samples 12 scene configs by seeded rejection sampling (rooms, 4–6 cars, desk,
scripted people, 2 cameras each). `sim/run_all.py` turns each config into a frame-exact timeline
(`sim/behaviour.py`), renders it in Blender 4.2.23 LTS (EEVEE RGB for three lighting variants and
Workbench instance masks), derives labels from the masks (`sim/labels.py`), applies a CCTV look and
encodes H.264 (`sim/post.py`). Event ground truth is scripted; rule events come from
`sim/gt_events.py` + `sim/configs/presets.json`. Every clip records the git sha, config and seed in
`gt.json`. Renders are synthetic and contain no real people.

## 3. Real open footage (demo and evaluation only — never trained on)

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

## 4. Detector weights

| item | source | licence |
|---|---|---|
| RF-DETR Nano, COCO-pretrained (`rf-detr-nano.pth`) | `pip install rfdetr==1.11.2`; weights auto-downloaded from https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth (md5 `fb6504cce7fbdc783f7a46991f07639f`) | Apache-2.0 (Roboflow RF-DETR, https://github.com/roboflow/rf-detr). Trained on COCO 2017 (annotations CC BY 4.0, images under Flickr terms) |
| `models/showroom_detector.onnx` | derived: exported with `models/export_onnx.py` (opset 17, onnxsim) | Apache-2.0 (derived from the above) |

No ultralytics/AGPL code or weights are used.
