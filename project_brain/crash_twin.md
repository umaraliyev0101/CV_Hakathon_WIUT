# 0. One-page summary

**Crash Twin** turns a 60-second phone video of a road accident into four things: a measured 3D replay of the scene, the impact point and direction for every vehicle, an estimated speed range, and a list of the traffic rules each driver most likely broke, citing the exact clause of Uzbekistan's Road Traffic Rules.

- **Input:** a phone walk-around video or 10-30 photos of the scene. Optionally a dashcam or CCTV clip.
- **Output:** a 3D replay that opens in any browser, the accident diagram, impact and speed estimates with error bars, and a draft violation list ("Vehicle A, clause 105: did not yield to a vehicle approaching from the right, confidence 0.81") with the evidence frames.
- **Users:** insurers (claims), traffic police / the city (clearing minor crashes fast), dealers' body-repair intake (Toyota Royal Motors, BYD Royal Motors).
- **Why computer vision:** every measurement comes from images: 3D reconstruction, vehicle detection and segmentation, damage segmentation lifted into 3D, metric scale from known car sizes, lane-marking and sign recognition, and video tracking. Physics and rules turn those measurements into a decision.
- **Moat (the replication test):** the whole chain, photos to 3D to physics to legal clause, does not exist as a product. Research covers only pieces: crash reports as TEXT to simulation (TRACE, CrashAgent, AccidentSim, AC3R), and photos to 3D damage on ONE car (CrashSplat, 2025). The professional tools (PC-Crash, Virtual CRASH) are manual and cost thousands of dollars per seat. Our assets are (1) the chained pipeline, (2) a rules engine coded from the Uzbek traffic rules, and (3) a test set measured against real crash data (NHTSA CISS) plus our own Tashkent captures.

> Track fit: the brief asks for "a product that turns images or video into a decision or an action for road safety", and names insurers and the city as users and "insurance claim" as a direction. Crash Twin is exactly that: images in, decision out.

# 1. The problem

**1.1 What happens today after a minor crash in Tashkent**
- Both drivers stop where the crash happened and wait for a traffic-police inspector (DYHXX) to come, measure, draw the scheme and decide who is at fault. The cars often block a lane while they wait, which causes secondary congestion and sometimes secondary collisions.
- The scheme is drawn by hand. Measurements are taken with a tape or not at all, and once the cars are moved the evidence is gone.
- The insurer then receives a paper scheme and a few photos, and has to judge the claim, including fraud (staged crashes, damage that does not match the story), with almost no measured evidence.

**1.2 What the world uses**
- Professional reconstruction: PC-Crash (about USD 5,000 for the first licence) and Virtual CRASH. These need a trained expert and days of work per case. They are for courts and serious cases, not for everyday claims.
- Insurer AI (for example Tractable): estimates repair cost from photos of ONE car. It does not reconstruct the scene and does not reason about rules.
- Research prototypes rebuild crashes in a simulator from police report TEXT (TRACE 2026, CrashAgent 2025, AccidentSim 2025, AC3R). None start from photos of the real scene.

**1.3 The gap Crash Twin fills:** a measured, rule-aware reconstruction from the photos any driver can take, in minutes, at a cost of cents.

> Numbers to collect before the pitch (do not invent them): crashes per year in Tashkent and the share that are minor (DYHXX / Ministry of Internal Affairs reports, stat.uz). Average waiting time for an inspector (ask 10 drivers or a police contact). Motor claims per month at one insurer (ask Kapital Sug'urta, Gross, Uzbekinvest, Alfa Invest).

# 2. The product

**2.1 User flow (driver or insurance agent)**
1. Open crashtwin web app on a phone (no install).
2. Follow the on-screen capture guide: walk slowly around the scene for 30-60 s, keeping both cars and the road markings in view, then take a close-up of each damaged area.
3. Optional: upload a dashcam or CCTV clip.
4. Wait about 2-10 minutes. Get a link to the case.
5. The case page shows: 3D replay (orbit, zoom, play the crash), accident diagram (top view), per-vehicle card (make/model, damage, impact direction, speed range), and draft rule verdicts with clause text and evidence images.
6. Export PDF for the insurer or the police. A human (inspector or claims handler) confirms or edits before anything is final.

**2.2 Mode (name it on slide 1)**
- **Recommended: Startup**, B2B SaaS sold to insurers per case. The dealers (Toyota/BYD Royal Motors body shops) are a channel: their repair intake uses the same report.
- Alternative: **Partner product** inside a dealer's body-repair and insurance-claims desk. Unit: after-sales / body repair. Process changed: damage intake and insurer paperwork.

# 3. System architecture

~~~flow  Figure 1 - Crash Twin pipeline
(Phone video or photos (+ optional dashcam/CCTV clip))
|
[M1 Capture QC: blur, coverage, light check; plate/face blur]
|
[M2 3D reconstruction: VGGT poses + points, then gsplat splat]
|
[M3 Vehicles: detect, segment, recognise make/model; fit 3D box]
|
[M4 Metric scale: known car dimensions + lane width as rulers]
|
[M5 Damage: 2D masks on photos, lifted onto the 3D twin]
|
[M6 Road context: lanes, markings, signs, signals + OpenStreetMap]
|
[M7 Physics: impact direction, crush energy, speed range, paths]
|
[M8 Rules engine: Uzbek traffic-rule predicates over the measured scene]
|
(Case page: 3D replay + diagram + speeds + rule verdicts + PDF)
~~~

**3.1 Modules in detail**

**M1 Capture quality control** (CV)
- Rejects or flags frames that are blurred (variance of Laplacian), too dark, or too repetitive; checks that both vehicles and some road markings were seen.
- Blurs licence plates and faces before anything is stored (plate and face detector). This is required by the hackathon rules and by privacy law.

**M2 3D reconstruction** (CV, core)
- Sample 40-120 frames from the video.
- VGGT (Visual Geometry Grounded Transformer, Meta + Oxford, CVPR 2025 best paper) predicts camera poses, depth maps and a point cloud in one forward pass, in seconds. Use the **VGGT-1B-Commercial** checkpoint; the original checkpoint is non-commercial.
- Fallback when VGGT fails: COLMAP or GLOMAP structure-from-motion (BSD licence).
- gsplat (nerfstudio, Apache-2.0) trains a 3D Gaussian Splatting model from those poses for the photo-real replay.
- Output: point cloud + splat + camera poses, in an arbitrary scale.

**M3 Vehicles** (CV)
- Detection + instance segmentation of each vehicle in every frame: YOLO11-seg or RT-DETR plus SAM 2 for clean masks.
- Masks from all views are voted into 3D, which gives each vehicle's 3D points.
- Fit an oriented 3D box (or a CAD template) to each vehicle: position, heading, length, width.
- Make/model recognition for the cars common in Uzbekistan (Chevrolet Cobalt, Nexia 3, Gentra, Spark, Damas, Malibu, Tracker; BYD Song Plus, Chazor, Han; Toyota Camry, RAV4 and others). Fine-tune a classifier on our own photos plus public car datasets (research licences).

**M4 Metric scale** (CV, our trick)
- 3D reconstruction from photos has no real-world scale. We recover it from the scene itself: the recognised car model has a known length and width (manufacturer spec sheet). Every recognised car is a ruler.
- Cross-check with lane width and dashed-line length, and optionally a sheet of A4 paper or a phone placed on the road.
- Report the scale uncertainty, for example "plus or minus 3%", and carry it through to the speeds.

**M5 Damage in 3D** (CV)
- Damage segmentation on the close-up photos: dent, scratch, crack, broken glass, broken lamp, missing part. Train on VehiDE and CarDD (research only), then fine-tune on our own photos.
- Lift the 2D masks onto the 3D vehicle model (the method of CrashSplat, 2025). This gives the damage location on the car body and the crush depth, measured as the deviation from the undamaged template surface.

**M6 Road context** (CV + map)
- Detect lane lines (solid or dashed), stop lines, zebra crossings, arrows, and signs (yield, stop, main road, no U-turn) in the frames; place them in 3D on the ground plane.
- Pull the intersection geometry from OpenStreetMap (ODbL) using the phone's GPS: number of lanes, one-way streets, priority road tags, speed limits where tagged.
- The ground plane (from M2) gives a top-down orthophoto of the scene. This becomes the base of the accident diagram.

**M7 Physics** (math on CV measurements)
- Impact point and direction of force (PDOF) from the 3D damage location and shape on each car.
- Crush depth and crush stiffness give the absorbed energy, which gives delta-V (the speed change in the impact). This is the classical Campbell crush-energy method; stiffness coefficients can be derived from public NHTSA NCAP crash tests.
- Conservation of momentum, using the final rest positions (from M3) and post-impact sliding (friction coefficient range for dry or wet asphalt), gives the impact speeds. Pre-impact paths are fitted backwards.
- Everything is reported as a **range** (Monte Carlo over the uncertain inputs: scale, friction, stiffness, masses).
- If a dashcam or CCTV clip exists: track the vehicles in the video (detector + tracker, reusing our elimination-round code, declared) and measure speeds directly. This overrides the crush estimate.
- Optional: replay the reconstructed crash in CARLA (MIT) or BeamNG.tech (free research licence) to check that the fitted speeds actually produce the observed rest positions and damage.

**M8 Rules engine** (logic on CV measurements; LLM only for wording)
- Each traffic-rule clause that matters in collisions is written as a code predicate over the measured scene (section 4).
- Output: the clause, the vehicle, the verdict (likely violated / likely not / cannot tell), the confidence, and the evidence (which frames and which measurements).
- An LLM is used only to cite the exact clause text and write the explanation in Uzbek or Russian. It never decides the verdict, so the decision is explainable and testable.
- The final decision on fault stays with the police or a court. The product produces a draft that a human signs.

**3.2 Licences of the main components (check each before shipping a product)**

| Component | Use | Licence |
|---|---|---|
| VGGT-1B-Commercial (Meta) | poses + points | Commercial use allowed on approved application; no military use |
| VGGT-1B (original) | same | Non-commercial: avoid in the product |
| COLMAP / GLOMAP | fallback SfM | BSD-3 |
| gsplat (nerfstudio) | Gaussian splatting | Apache-2.0 |
| SAM 2 (Meta) | segmentation | Apache-2.0 |
| Ultralytics YOLO11 | detection/seg | AGPL-3.0 (open repo OK; product needs enterprise licence) |
| RT-DETR / RT-DETRv2 | detection (YOLO alternative) | Apache-2.0 |
| Depth Anything V2 Small | monocular depth backup | Apache-2.0 (Base/Large are non-commercial) |
| CARLA | simulation, synthetic data | MIT code, CC-BY assets |
| BeamNG.tech | soft-body crash simulation | free research licence on application |
| OpenStreetMap | road geometry | ODbL (attribution, share-alike) |
| Web splat viewer (SuperSplat / three.js splat viewers) | 3D replay in browser | check each project (most MIT) |

# 4. Rules engine: Uzbekistan Road Traffic Rules

Source: **Road Traffic Rules, Annex 1 to Cabinet of Ministers Resolution No. 172 of 12 April 2022**, in force from 1 May 2022, official text on lex.uz (https://lex.uz/docs/-5953883). Clause 78 was amended by Resolution No. 140 of 3 April 2023 (60 km/h in Tashkent and Nukus). Clause numbers below were read from the lex.uz text on 10 October 2026; re-check them before the pitch.

**4.1 Clauses that matter in collisions, and what the cameras must measure**

| Clause | Rule (short) | CV evidence needed | Predicate (sketch) |
|---|---|---|---|
| 41 | On a red signal, stop before the stop line (the front must not pass it) | stop line in 3D, vehicle front position, signal state (from video) | front_x > stop_line_x while signal == red |
| 45 | Signal before starting, changing lane, turning, stopping | turn-indicator state in video frames | maneuver_started and not indicator_on |
| 53 | A manoeuvre must not endanger other road users | trajectories (video) or impact geometry | maneuver_by(A) and collision within T s |
| 54 | Leaving an adjacent territory (yard, parking): yield to all on the road | exit location from OSM / photos, paths | A came from adjacent territory and collided |
| 55 | When changing lanes, yield to vehicles moving straight in that lane | lane lines, lateral motion of A | A crossed lane line, B kept its lane, impact on A's side |
| 60 | When paths cross and priority is not set, yield to the vehicle on the right | approach directions, no priority signs | B approached from A's right and A did not yield |
| 62 | U-turns prohibited at listed places | U-turn path, location type (crossing, bridge, poor visibility) | path curvature ~ 180 deg at a prohibited location |
| 65 | On two-way roads with 4+ lanes, do not drive in the lanes for oncoming traffic | lane count + direction (OSM + markings), vehicle lateral position | vehicle center in an opposing lane |
| 64 | Crossing into the oncoming direction across a solid line (1.1, 1.3) is prohibited | solid line detected in 3D, wheel path | path crosses a solid-line polyline |
| 70 | Keep to marked lanes; only dashed lines may be crossed when changing lanes | lane polylines, line type, path | crossing of a solid line not for turning permitted by markings |
| 73 | Keep a following distance that avoids a collision if the leader brakes hard | rear-end impact geometry, speed range | rear impact on B and front impact on A in the same lane |
| 78 | Speed limits: 70 km/h in settlements, 60 km/h in Tashkent and Nukus (2023), 30 km/h near schools/kindergartens (300 m), 20 km/h in residential areas | speed range (M7), location type (OSM, signs) | lower bound of speed range > limit |
| 99 | Turning left or U-turn on a green: yield to oncoming traffic going straight or right | signal state, paths | A turned left, B came straight from opposite, collision |
| 103 | On a green, yield to vehicles still finishing the crossing and to pedestrians finishing | timing (video) | B already in intersection when A entered |
| 104 | At a crossing of unequal roads, the vehicle on the secondary road yields to the main road | priority signs (2.1, 2.4, 2.5) + OSM priority | A on secondary road and collision with B from main road |
| 105 | At a crossing of equal roads, yield to the vehicle approaching from the right | approach directions, no priority signs | B approached from A's right |
| 107 | At an equal-roads crossing, the vehicle turning left yields to oncoming straight/right | paths | A turned left across B |
| 109 | At an unregulated pedestrian crossing, slow down or stop for pedestrians crossing | zebra crossing in 3D, pedestrian presence (video) | pedestrian on crossing and A did not stop |

**4.2 How a verdict is produced**
1. M3-M7 produce a **scene graph**: vehicles (position, heading, path, speed range, damage), road elements (lanes, line types, stop lines, crossings, signs, priority), and signal state if video exists.
2. Each predicate is evaluated with **uncertainty**: we sample the scene 1,000 times inside the error bars from M4 and M7, and the confidence is the share of samples where the predicate is true.
3. Below 0.6 or with missing evidence (for example no video, so no signal state) the verdict is "cannot tell" and the report says exactly which evidence is missing. This keeps it honest and is a strong answer in Q&A.

**4.3 Example predicate (Python)**

```
def clause_105_yield_to_right(scene, a, b):
    """Equal roads: A must yield to B if B approaches from A's right."""
    if scene.priority_road_known and not scene.roads_equal(a, b):
        return Verdict.NOT_APPLICABLE
    rel = scene.bearing(a.approach_heading, b.approach_position)  # deg
    from_right = 20 <= rel <= 160
    a_entered_first = a.entry_time is not None and b.entry_time is not None \
        and a.entry_time < b.entry_time - 1.0
    if from_right and not a_entered_first and scene.collided(a, b):
        return Verdict.LIKELY_VIOLATED("105", actor=a,
               evidence=[b.approach_frames, scene.impact_frame])
    return Verdict.LIKELY_NOT
```

# 5. Data: what we train on and what we test on

**5.1 Training and development data (public)**

| Dataset | What it contains | Used for | Licence / access |
|---|---|---|---|
| NHTSA CISS (2017-2024) | Real US crashes: scene + vehicle photos, measured crush profiles, scene diagrams, reconstructed delta-V, many with EDR (black box) speed | Ground truth for PDOF, crush, delta-V; physics calibration and **test set** | US government public data (personal info removed). crashviewer.nhtsa.dot.gov; search helper github.com/adshelton/ciss-tool |
| NHTSA NCAP crash tests | Lab crashes at known speeds, photos, crush measurements | Crush stiffness coefficients per vehicle class | US government public data |
| VehiDE | 13,945 photos of damaged cars, 32,000+ damage instances, 8 classes | Damage segmentation training | Images from Flickr/Shutterstock; authors state academic use. Kaggle lists Apache-2.0, which does not clear image copyright |
| CarDD | 4,000 high-resolution images, 9,000+ instances, 6 classes (dent, scratch, crack, glass shatter, lamp broken, tire flat) | Damage segmentation training | Non-commercial research only, signed agreement (cardd-ustc.github.io) |
| Nexar Collision Prediction | 1,500 dashcam clips (~40 s, 1280x720, ~30 fps) labelled collision / near-collision / normal, with event and alert times | Video path: tracking and speed from dashcam, impact timing | Nexar Open Data License (read the terms; restricted unethical use) |
| CCD, DoTA, DAD | Public dashcam accident video datasets | Extra video training and tests | Research licences, check each |
| Mapillary Traffic Sign Dataset / Vistas | Street-level images with signs and lane markings labelled | Sign + marking detection | CC BY-NC-SA (research) |
| ApolloCar3D | Street images with 3D car pose and CAD model fits | 3D car pose fitting, CAD templates | Research licence, check |
| OpenStreetMap (Tashkent extract) | Roads, lanes, one-ways, priority, crossings | Road context; synthetic scenes | ODbL |
| Manufacturer spec sheets | Length / width / height / wheelbase of common Uzbek-market cars | Metric scale (M4) | Public facts |

> For a commercial product, the research-only image sets (VehiDE, CarDD, Mapillary NC) must be replaced by our own photos or by insurer data under contract. For the hackathon they are fine, and they must be listed in the README.

**5.2 Synthetic data (we generate it; declared)**
- Import 10-20 real Tashkent intersections from OpenStreetMap into CARLA (OSM to OpenDRIVE conversion is built into CARLA) or BeamNG.
- Script 500+ crashes with known ground truth: speeds, paths, impact point, signal state, and the violated clause (for example a car crossing a solid line, failing to yield from the right, running a red).
- Render each crash from virtual phone walk-around cameras (day, night, rain).
- Use it to (a) train and test the rules engine, (b) stress-test reconstruction, and (c) test night/rain robustness.
- Declare the generation method in the README, as the rules require.

**5.3 Our own data (the local moat)**
- **Parked-car captures:** 50+ walk-around videos of real parked cars in Tashkent (university and mall parking lots), plates blurred, no people. We measure car positions and gaps with a tape measure: this is the ground truth for 3D and scale accuracy.
- **Staged toy-car crashes:** 1:36 or 1:43 die-cast models of Cobalt, Nexia, Camry and BYD on a printed intersection mat (lane lines, zebra, stop lines at scale). Stage 30+ scripted crashes with known positions and known "violations". Because the scale factor is printed on the box, the metric pipeline still works. This is also the live demo.
- **Damage close-ups:** 200+ photos of real dents and scratches on cars in parking lots and at a body shop (ask the Toyota/BYD service centre). Label with CVAT or Label Studio (polygon masks).
- **Later, with partners:** insurers' anonymised claim photos with the final settled fault; DYHXX schemes for the same cases. This is the data no other team can get, and the long-term moat.

**5.4 Test sets and metrics (built before training, never trained on)**

| Test set | Size (target) | Ground truth | Metric |
|---|---|---|---|
| T1 Parked cars (ours) | 50 scenes | tape-measured distances and gaps | scale error %, vehicle position error (cm), heading error (deg) |
| T2 Toy crashes (ours) | 30 scenes | scripted positions, impact points, violations | position error, PDOF error (deg), violation precision / recall |
| T3 NHTSA CISS subset | 100 cases (frontal, side, rear mix) | measured crush, PDOF, CISS delta-V, EDR delta-V | PDOF MAE (deg), delta-V MAE (km/h) and % within range |
| T4 Synthetic (CARLA/BeamNG) | 200 scenarios, 10 intersections | exact speeds, paths, clauses | speed MAE, clause precision / recall per clause, "cannot tell" rate |
| T5 Damage (VehiDE/CarDD val + ours) | 500 images | polygon masks | mask mAP / IoU per class |
| T6 Video (Nexar collision subset) | 100 clips | event and alert time | impact-time error (s), tracking ID switches |
| T7 System | all of the above | wall-clock time and cost per case | seconds per stage, USD per case |

> Show the jury a table of these numbers including the bad ones. "Known failure cases" is part of the technical-complexity criterion: night with no street light, heavy rain on the lens, cars already moved, crashes with only one car visible.

# 6. Sample data formats

**6.1 Capture protocol (printed on the capture screen)**
1. Stand 3-5 m from the cars. Walk one full circle, slowly, phone in landscape, 30-60 s.
2. Keep the road markings in view; tilt down a little.
3. Then take one close-up of each damaged area from about 1 m.
4. If it is dark, turn on the phone torch and walk slower.

**6.2 Annotation file for one scene (our test sets)**

```
{
  "scene_id": "T2_toy_017",
  "capture": {"video": "T2_toy_017.mp4", "fps": 30, "device": "iPhone 13",
              "scale_note": "1:36 die-cast, mat printed at 1:36"},
  "vehicles": [
    {"id": "A", "model": "Chevrolet Cobalt", "rest_xy_m": [3.42, 1.10],
     "rest_heading_deg": 74, "impact_zone": "front-left",
     "pre_speed_kmh": 38, "path": "straight-west-to-east"},
    {"id": "B", "model": "BYD Song Plus", "rest_xy_m": [4.95, 2.60],
     "rest_heading_deg": 182, "impact_zone": "right-side-front-door",
     "pre_speed_kmh": 22, "path": "straight-south-to-north"}
  ],
  "road": {"type": "equal_roads_crossing", "signals": false,
           "markings": ["1.1 centre line on west arm", "zebra north arm"]},
  "ground_truth_violations": [
    {"vehicle": "A", "clause": "105", "note": "B approached from A's right"}
  ]
}
```

**6.3 Output report for one case (what the user gets)**

```
CASE CT-2026-10-11-0042   Chilonzor, Bunyodkor ave. x Muqimiy st.
Vehicles: A Chevrolet Cobalt (white), B BYD Song Plus (grey)
Impact: A front-left into B right side; PDOF A 10 deg, B 280 deg
Speeds at impact (90% range): A 31-44 km/h, B 15-26 km/h
Scale check: car length A measured 4.47 m (spec match 99%)
Rule findings (draft, for human review):
  A  clause 105  did not yield to vehicle from the right   conf 0.81
  A  clause 78   speed above 60 km/h limit                 conf 0.03 (no)
  B  -           no violation found                       conf n/a
Missing evidence: signal state (no video); priority sign not seen.
Evidence: frames 112, 140, 233; 3D replay: crashtwin.app/c/CT-...-0042
```

# 7. Compute and cost

**7.1 Per case (estimates; measure on our own hardware and replace)**

| Stage | Hardware | Time (estimate) |
|---|---|---|
| Frame sampling, QC, blur | CPU | 5-15 s |
| VGGT on 60-100 frames | 1 GPU, 24 GB+ VRAM | 5-30 s |
| Gaussian splat (gsplat, 7k-30k steps) | same GPU | 3-15 min (skip for the instant report; run in background for the replay) |
| Detection + SAM 2 masks on ~100 frames | same GPU | 20-60 s |
| Damage seg, road context, physics Monte Carlo, rules | GPU/CPU | 10-40 s |
| LLM write-up (paid API, in cost model) | API | 5-15 s |

- **Instant report:** about 1-3 minutes without the splat. **Full 3D replay:** about 5-15 minutes.
- **Cost per case:** a rented GPU of the RTX 4090 / L4 / A100 class costs roughly USD 0.5-2 per hour, so GPU cost is roughly USD 0.1-0.5 per case, plus the LLM cost (cents). State these as measured numbers in the deck, not these estimates.

**7.2 For the hackathon**
- Training: fine-tuning a YOLO/RT-DETR damage model on ~5-15k images takes about 1-3 hours on one rented GPU. Make/model classifier: under an hour.
- Serving the demo: one rented GPU server (RunPod, Vast.ai, Lambda) with a FastAPI backend, and a static frontend on Vercel or Netlify. Deploy from the public repository (the rules require it), ideally with GitHub Actions.
- Backup: precompute 3-5 cases so the replay works even if the GPU is busy, and say so honestly (not as a fake live result).

# 8. Demo plan (7 minutes, the jury gives the input)

- **Live input:** the jury arranges two die-cast cars on our printed intersection mat in any crash position they like. We film a 30 s walk-around with a phone, upload it, and show the report: positions, impact direction, which clause applies. The jury can open the public site on their own phones and orbit the 3D replay.
- **Second input (if allowed):** a real parked car outside: we show measured length versus spec sheet (scale accuracy live).
- **Recorded backup** of the whole demo in case the network fails (the rules say technical problems do not stop the clock).

**Recommended 7-minute split (from the brief)**
1. Problem and user (1:00): the crash on Bunyodkor waiting for an inspector; the insurer with a hand-drawn scheme.
2. Live demo (2:00): jury's toy crash, upload, report, 3D replay on their phones.
3. Technology (2:00): pipeline figure; M4 scale trick; damage lifted to 3D; rules engine; test-set table with errors and failure cases.
4. Budget and ROI (1:30): pilot cost, unit economics, payback with stated assumptions.
5. Next steps (0:30): pilot with one insurer and one DYHXX district; data partnership.

# 9. Business case

**9.1 Who pays and why**
- **Insurers:** fewer field-adjuster visits, faster claims, and fraud detection (damage that does not match the claimed crash). Pay per case.
- **Traffic police / city:** minor crashes cleared in minutes; the lane is freed. Pay per licence or a city contract (slower sales).
- **Dealers / body shops:** structured damage intake for insurance repairs; they push the app to their customers.

**9.2 Pilot in 3 months (within existing processes)**
- Month 1: one insurer, one claims team, Crash Twin used side by side with their current process (no decision is automated).
- Month 2: 200+ real cases; compare our draft fault verdicts with the final settled verdicts.
- Month 3: report accuracy, time saved per case, fraud flags confirmed. Decide on paid rollout.
- Regulation: the product is an evidence tool. The fault decision remains with the police or a court, so no law change is needed for the pilot. Personal data: plates and faces blurred, data stored in Uzbekistan if required by the personal-data law (check with the insurer's lawyer).

**9.3 Unit economics template (fill in with real numbers; every value is an assumption to state)**

| Item | Symbol | Example assumption |
|---|---|---|
| Price per case to insurer | P | to set after interviews |
| GPU + LLM cost per case | C | measured, about USD 0.2-0.6 |
| Field visits avoided per 100 cases | V | ask insurer |
| Cost of one field visit | F | ask insurer |
| Fraud cases detected per 1,000 claims | D | ask insurer |
| Average fraudulent payout avoided | Z | ask insurer |
| Insurer saving per case | S = (V/100)*F + (D/1000)*Z | computed |

- Pilot budget: 3 people x 3 months, one GPU server, cloud storage, LLM API, legal review. Payback = pilot cost / (monthly cases x (P - C)). Show the formula and every input on one slide.

# 10. Risks and honest limits

- **Speed from crush is uncertain.** Even NHTSA's own reconstructions overestimated low-speed rear impacts by about 36% on average against the EDR (Watson et al., SAE 2022). We always show a range, and we prefer video when it exists.
- **The scene may already be cleared.** If the cars were moved, rest positions are lost; we fall back to damage-only analysis plus any video.
- **Night and rain.** Reconstruction degrades in low light and with rain on the lens. Mitigation: capture guide (torch, slower walk), QC rejects bad frames, report states reduced confidence.
- **Unusual vehicles** (trucks, buses, motorcycles) are weaker for scale and crush. Report "cannot tell" rather than guess.
- **Legal:** the product does not assign legal fault. It drafts evidence with confidence for a human to confirm.
- **Licences:** research-only datasets are fine for the hackathon but must be replaced for a commercial product.
- **Privacy:** plates and faces are blurred before storage; no personal data in the test sets; staged and parked-car captures only.

# 11. Build scope: what to build now, what to show as roadmap

**MVP for the final (must run end to end on the jury's input)**
1. Upload page (phone), frame sampling, QC, plate/face blur.
2. VGGT reconstruction + vehicle segmentation + oriented 3D boxes.
3. Metric scale from known car or toy-model dimensions.
4. Top-view diagram with positions and headings; impact zone from damage masks (2D damage model + lifting).
5. Rules engine with 4-5 clauses that the toy mat can show: 105 (yield to the right), 104 (secondary road), 99/107 (left turn), 64/70 (solid line), 73 (rear-end following distance).
6. Report page + 3D point-cloud or splat viewer + PDF export.
7. Test-set table: T1 + T2 (ours) and a small T3 (20 CISS cases) for PDOF.

**Roadmap (show, do not claim as built)**
- Full crush-energy speed estimation calibrated on CISS + NCAP.
- Video path with signal-state detection (clauses 41, 99, 103).
- Simulator replay check (CARLA/BeamNG).
- Insurer data partnership; Uzbek / Russian voice guidance during capture.

**Team split (3 people)**
- Person 1: reconstruction (M2, M4) + 3D viewer.
- Person 2: vehicles + damage + road markings (M3, M5, M6) + datasets.
- Person 3: physics-lite + rules engine + report, backend, deploy, deck.

# 12. README checklist (required by the rules)

- How to run (one command), environment, GPU requirement.
- Architecture figure and module list.
- **Data sources and licences:** every dataset in 5.1, the synthetic generation method (5.2), our own captures and how they were made (5.3).
- **Models and licences:** every model in 3.2.
- Reused code from the elimination round, declared.
- Test-set results (5.4) and known failure cases.
- Deployment: where it runs, from which commit; GitHub Actions log.

# 13. Sources

- WIUT Hackathon 2026, Final task, Computer Vision track (hackathon.wiut.uz, team dashboard)
- Road Traffic Rules, Resolution No. 172 of 12.04.2022: https://lex.uz/docs/-5953883
- Amendment to clause 78 (60 km/h), Resolution No. 140 of 03.04.2023: https://lex.uz/uz/docs/-6422892
- VGGT: https://github.com/facebookresearch/vggt
- VGGT-1B-Commercial weights: https://huggingface.co/facebook/VGGT-1B-Commercial
- CrashSplat, 2D to 3D vehicle damage segmentation in Gaussian Splatting (2025): https://arxiv.org/abs/2509.23947
- TRACE, reconstruction of accidents in CARLA (2026): https://arxiv.org/abs/2604.22068
- TRACE / TrackMate, accident reports to BeamNG and MetaDrive (2025): https://arxiv.org/abs/2502.02025
- CrashAgent, crash scenario generation (2025): https://arxiv.org/abs/2505.18341
- AccidentSim (2025): https://arxiv.org/abs/2503.20654
- AC3R, automatically reconstructing car crashes from police reports (BeamNG): https://www.beamng.tech/research/
- Comprehensive Forensic Tool for Crime Scene and Traffic Accident 3D Reconstruction (MDPI Algorithms): https://doi.org/10.3390/a18110707
- Accuracy of 3D Gaussian Splatting for virtual crime scene reconstruction (Frontiers, 2026): https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1755361
- NHTSA Crash Viewer (CISS): https://crashviewer.nhtsa.dot.gov/crashviewer
- NHTSA CISS programme: https://www.nhtsa.gov/crash-data-systems/crash-investigation-sampling-system
- CISS search tool: https://github.com/adshelton/ciss-tool
- NCAP data download helper: https://github.com/ZJAllen/NCAPDownload
- Watson et al. 2022, rear impact reconstructions vs EDR in CISS (SAE 2022-01-5069)
- VehiDE dataset: https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection
- CarDD dataset: https://arxiv.org/abs/2211.00945 and https://cardd-ustc.github.io
- Nexar Dashcam Collision Prediction dataset: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction and https://arxiv.org/abs/2503.03848
- PC-Crash pricing: https://pc-crash.com/buy-now/
- Virtual CRASH: https://www.vcrashusa.com/vc6
- gsplat: https://github.com/nerfstudio-project/gsplat
- SAM 2: https://github.com/facebookresearch/sam2
- CARLA: https://carla.org
- BeamNG.tech research: https://beamng.tech/research/
- OpenStreetMap: https://www.openstreetmap.org (ODbL)

> Prepared 10 October 2026. Licence statements were checked against public pages on that date. Verify each licence and clause number again before submission. Compute and cost figures are estimates until measured.
