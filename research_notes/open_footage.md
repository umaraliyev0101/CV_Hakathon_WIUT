# Open footage and datasets for showroom analytics (survey, 2026-10-11)

These sources were checked on the web on 2026-10-11. Items marked **UNVERIFIED** were not confirmed
on the source page. Check the licence again before redistributing anything.

## Bottom line

- **No openly licensed CCTV-angle car-showroom footage exists**, and no public dataset has
  staff/customer role labels. This is why we build the synthetic pipeline (`sim/`): it gives
  exact roles, events and boxes.
- The best real footage is **one Pexels shoot** (creator Antoni Shkraba, IDs 71542xx): a
  salesperson with customers in a dealership. Commercial use is OK, with no attribution required.
  The camera is probably at eye level (stock framing), not CCTV height.
  - Use it for the demo and as a sanity check on real footage (does the detector, tracker and role
    logic hold up?).
  - Its ground truth would have to be hand-labelled.
- The best large datasets that are safe for commercial use are **NVIDIA PhysicalAI-SmartSpaces**
  and **MEVA**, both CC BY 4.0. Datasets marked NC are fine for internal evaluation only.

## 1. Real showroom stock clips

| Clip | URL | Licence | Notes |
|---|---|---|---|
| Salesman talking to a customer | https://www.pexels.com/video/salesman-talking-to-a-customer-7154214/ | Pexels | 1080p, 12 s (verified) |
| Couple talking to a car dealer | https://www.pexels.com/video/couple-talking-to-a-car-dealer-7154217/ | Pexels | 1080p, 12 s (verified) |
| Salesman talking to a customer (2) | https://www.pexels.com/video/salesman-talking-to-a-customer-7154241/ | Pexels | from listing |
| Couple talking to a car agent | https://www.pexels.com/video/couple-talking-to-a-car-agent-7154228/ | Pexels | from listing |
| A couple buying a car | https://www.pexels.com/video/a-couple-buying-a-car-7154230/ | Pexels | from listing |
| Woman negotiating car with a dealer | https://www.pexels.com/video/woman-negotiating-car-with-a-dealer-7154238/ | Pexels | from listing |
| A car dealer talking while standing | https://www.pexels.com/video/a-car-dealer-talking-while-standing-7154235/ | Pexels | from listing |
| Businesswoman entering car dealership | https://www.pexels.com/video/businesswoman-entering-car-dealership-building-36512656/ | Pexels | 4K vertical, 9 s, entry event |
| Car Dealership Success | https://pixabay.com/videos/car-dealership-success-344210/ | Pixabay | **AI-generated**; avoid as "real" |
| Mixkit dealership clips (23476, 23989) | mixkit.co | **Personal use only** | Do not use in the product or demo |

**Pexels licence:** free, commercial use allowed, no attribution required. It forbids showing
identifiable people in a bad light, implying endorsement, and redistributing the clips as-is. Using
the clips to train ML models is **UNVERIFIED**, so use them for demo and evaluation only.
Coverr has no showroom clips with people. Videvo now redirects to Freepik and was not checked.

## 2. Surveillance and retail datasets

| Name | Licence | What it has | Use |
|---|---|---|---|
| [NVIDIA PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) | CC BY 4.0 | Multi-camera, 1080p30, 2D boxes and cross-camera IDs (3D from 2025). Mostly synthetic warehouses; some retail, hospital and lab scenes | Evaluation and training (commercial OK). Download one or two retail scenes only, never all of it |
| [MEVA](https://mevadata.org/) | CC BY 4.0 | 328 h of ground-level camera video with activity labels (MEVID adds boxes and IDs) | Evaluation and training. Mostly outdoor, staged |
| [CAVIAR shopping centre](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) | CC BY-SA (as stated on the page) | Mall corridor CCTV with boxes, IDs and roles/activities | Evaluation and a historical demo; 384×288 |
| MOT17 / MOT20 | CC BY-NC-SA 3.0 (UNVERIFIED) | Pedestrian MOT boxes | Internal evaluation only |
| DanceTrack | Videos non-commercial | Tracking with uniform appearance | Internal evaluation only |
| MERL Shopping | Non-commercial research (UNVERIFIED) | Overhead grocery videos, shelf actions, no tracks | Internal evaluation only |
| WILDTRACK | No licence stated | 7 cameras, outdoor plaza | Research only |
| [CITIC-MTMC](https://zenodo.org/records/17512043) | CC BY-NC-SA 4.0 | 5 indoor cameras, including a research-centre "showroom" camera | Internal evaluation only |
| Roboflow `face-fj9gn/retail-rt0l2` | MIT? (UNVERIFIED) | 524 images, possibly Customer/Employee classes | Check before use |

## 3. Generated video

- **Wan 2.2** (https://github.com/Wan-Video/Wan2.2): Apache 2.0, and the project claims no rights to
  outputs. It runs locally (TI2V-5B at 720p fits on 24 GB), so it could run on our GPU 1. This is
  the safest generator for **pitch B-roll**.
- Free tiers of Veo, Kling and Hailuo are personal use only or watermarked. Avoid them.
- None of these keep a fixed CCTV angle or consistent identities. They are **not ground truth** and
  are not for evaluation.

## Recommended use

| Need | Source |
|---|---|
| Exact ground truth for events and roles (evaluation and tuning for A and B) | Our Blender synthetic clips (`data/synthetic/share/`) |
| Real-footage sanity check and live demo | Pexels 7154214, 7154217, 7154241, 36512656, plus our own staged phone clips |
| Person-detector training on real data | Our staged clips (hand-labelled) plus optional SmartSpaces retail scenes |
| Detector fine-tuning on synthetic data | Synthetic COCO export |
| Pitch B-roll | Wan 2.2 on GPU 1 (optional), labelled "AI-generated" |
