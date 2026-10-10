# Due-diligence verification, batch 11 (city-roads segment): 10 ideas for the WIUT CV hackathon (Tashkent, 10-11 Oct 2026)

> Read-me. Written 2026-10-10 by the verifier for ideas city-7, city-8, city-9, city-10, city-11, city-12, city-15, city-17, city-18, city-19.
> Tools: WebSearch (shared 200-call session quota; it was exhausted for most of the run: about 10 of roughly 35 attempts went through, so most planned Russian/Uzbek and second-pass novelty searches did NOT run) and WebFetch of primary pages (vendor pages, GitHub READMEs, open-data API endpoints, news articles; two PDFs were saved by the tool and read locally with pdftotext). Two Semantic Scholar lookups were made by DOI (metadata for a known paper, not a search). I did not use any other search route.
> Tags: **[SNIPPET]** = taken from a search-result summary, page body not read; **[CALC]** = my arithmetic on cited inputs; **[INFERENCE]** = my judgment; **[BACKGROUND]** = general domain knowledge I did not re-verify in this run; **[TEAMMATE]** = taken from the brief's saturation map, Uzbekistan fact sheet, hackathon intel or tech radar, not re-fetched by me.
> Conversion: about 11,800 UZS per USD ([TEAMMATE] market sheet). Judging weights ([TEAMMATE] hackathon_intel.md): Innovation and Originality 25%, Technical Complexity 25%, Impact and Practicality 25%, Presentation 15%, Business Potential 10%; only about 8.5 scheduled on-campus hacking hours; 7+3 minute pitches; the brief demands night/rain/local-vehicle evidence.
> Absence of a search hit is weak evidence: most "nothing found" statements below rest on one query.

## Verdict table

| id | idea | novelty | feasibility | demo | business | moat | local fit | verdict |
|---|---|---|---|---|---|---|---|---|
| city-7 | SightTriangle | 4 | 3 | 4 | 2 | 2 | 3 | sharpen |
| city-8 | SchoolGate | 3 | 4 | 3 | 2 | 2 | 4 | sharpen |
| city-9 | DustVis | 3 | 3 | 3 | 2 | 2 | 4 | sharpen |
| city-10 | SwerveMap | 4 | 3 | 3 | 2 | 3 | 3 | keep |
| city-11 | GhostWorks | 3 | 3 | 3 | 2 | 2 | 3 | sharpen |
| city-12 | WetLedger | 4 | 2 | 2 | 1 | 1 | 2 | kill |
| city-15 | OverpassIgnored | 3 | 4 | 3 | 2 | 2 | 4 | sharpen |
| city-17 | Euro4Gate | 2 | 3 | 5 | 1 | 1 | 2 | kill |
| city-18 | StopCount | 3 | 2 | 2 | 2 | 2 | 4 | sharpen |
| city-19 | GlareClock | 3 | 4 | 5 | 1 | 1 | 2 | kill |

Cross-cutting findings are at the end of the file.

---

## city-7 SightTriangle: intersection visibility audit from one phone pan

### Takeaway
No phone-based sight-triangle audit turned up in one extended search; the nearest work is mobile-LiDAR sight-distance analysis in papers and a two-person rod-and-tape manual method, so novelty is real. The weak points are physical (far-field monocular metric depth is unvalidated, and stopping sight distances for 40-60 km/h streets are already 35-75 m) and commercial ($15 per audit against a two-rod manual survey, vague payer). Verdict: sharpen.

### Cited Findings
#### Queries run
- EN extended: "intersection sight distance measurement smartphone app LiDAR or monocular depth sight triangle obstruction detection" (the earlier phrasing "intersection sight distance obstruction measurement smartphone app depth estimation" also ran). Not run (quota): Russian ("треугольник видимости киоск Ташкент"), Uzbek, a pedestrian-crossing "daylighting" probe, Uzbek building-code probe.
#### Findings
- The search "found no smartphone app built specifically for intersection sight distance obstruction measurement"; results were mobile-LiDAR frameworks, an Iowa manual-survey handbook and unrelated smartphone-depth papers (a TrueDepth/LiDAR validation used a door at 1-3 m, far below intersection ranges) [SNIPPET].
- Mobile-LiDAR intersection sight-distance paper (Sustainability 2021, 29 citations per Semantic Scholar): automated framework finds roadside obstacles from LiDAR point clouds, simulates the driver's field of view, and links limited sight distance to higher collision risk — [MDPI](https://www.mdpi.com/2071-1050/13/16/9259); metadata via [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/su13169259) (MDPI page itself returned 403). A second paper, "Virtual assessment of sight distance limitations using LiDAR technology: Automated obstruction detection and classification" (Automation in Construction, 2021) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0926580521000303) [SNIPPET].
- Manual practice (CTRE/Iowa handbook hosted by NACTO, 2001 AASHTO numbers): driver eye height 3.5 ft, approaching-vehicle height 4.25 ft; two people with a 3.5 ft sighting rod and a 4.25 ft target rod walk until the target is visible and measure the distance; compare with minimum stopping sight distance: 115 ft at 25 mph, 140 ft at 30, 195 ft at 40, 245 ft at 50, 285 ft at 55; obstructions inside the sight triangle "should be removed or modified" — [Iowa sight distance handbook (PDF)](https://nacto.org/wp-content/uploads/sight_distance_study_Iowa.pdf) (text extracted locally).
- Depth Anything 3 README: DA3METRIC-LARGE (0.35B, Apache-2.0) for metric depth with "metric_depth = focal * net_output / 300"; DA3-SMALL (0.08B) and DA3-BASE are Apache-2.0 "any-view" models; Large/Giant/Nested are CC BY-NC 4.0; README gives no scale-accuracy figures, no timing, no official WebGPU support — [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3). The DA3 abstract does not claim metric depth — [arXiv 2511.10647](https://arxiv.org/abs/2511.10647).
- MoGe-2: metric-scale point maps, checkpoints 326M/104M/35M params, "60ms latency per image (A100 or RTX3090, FP16, ViT-L)", code MIT; no numeric accuracy or range limits on the README — [MoGe README](https://github.com/microsoft/MoGe). The abstract claims "precise metric scale" without numbers — [arXiv 2507.02546](https://arxiv.org/abs/2507.02546).
- RF-DETR (Apache-2.0 for N/S/M/L): Small 53.0 AP at 3.5 ms, Medium 54.7 AP at 4.4 ms (T4, TensorRT FP16, batch 1) — [RF-DETR README](https://github.com/roboflow/rf-detr). COCO has no kiosk class [BACKGROUND], so a kiosk label needs fine-tuning or an open-vocabulary detector (OWLv2 and Grounding DINO are Apache-2.0 per [TEAMMATE] tech radar).
- Uzbek standard plate size is 520 mm x 110 mm (a scale cue only when a car is in frame) — [Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan).
- Local: AI cameras at 10 Tashkent crossings fine drivers for not yielding to pedestrians; a blogger reported 1,700 fines in 3 days (Spot did not verify the number; the Road Safety Service press secretary confirmed the programme), 30 more intersections announced, no vendor named, "everyone who did not brake once a pedestrian appears" is fined, and the article says nothing about visibility or obstructions at the crossings — [Spot, 13 Jul 2025](https://www.spot.uz/ru/2025/07/13/cameras-ai/). Failure to yield to pedestrians was 13.6% of Q1 2026 crashes [TEAMMATE, Kun.uz 13 Apr 2026]. Tashkent's deputy hokim cites 3,946 adaptive traffic lights at 271 intersections — [Spot, 29 Jul 2026](https://www.spot.uz/ru/2026/07/29/green-corridor/).
- OSAGO prices are fixed by Cabinet Resolution 458 (from 1 Jan 2026) [TEAMMATE], so an insurer has no tariff lever to "price black spots" in the compulsory line.
- Cost [CALC]: 0.5 s x 10-20 frames = 5-10 s on a T4 at $0.000164/s (Modal price, [TEAMMATE] tech radar) = $0.0008-0.0016 per audit.
- Market [CALC]: 271 intersections x $15 = about $4k; even 5,000 junctions x $15 = $75k one-off.
### Inferences
- [CALC] Road-plane geometry does not fix the far field: at 1.1 m phone height a point 40 m away sits 1.575 degrees below the horizon; a 0.2 degree pitch error moves it to 35.5-45.8 m (-11% / +15%), similar to monocular-depth error. Near-field blockers (kiosk, parked car, wall within about 15 m) are measured far better, and their grazing sight lines fix the visible distance along the crossing street by trigonometry, so the product should report blockers plus pass/fail at the stopping distance for the posted speed (35-75 m for 40-80 km/h by the Iowa table [CALC from ft]).
- Departure sight triangles for stop-controlled minor roads are longer (about 83-104 m for a 7.5 s gap at 40-50 km/h [CALC; gap value is AASHTO practice, BACKGROUND]) and are out of reach for a phone.
- At $15 the tool competes with two interns, a tape and two sticks; value must come from the documented removal list and report, not from measurement.
- [INFERENCE] The pedestrian-yield cameras fine any driver who has not braked when a pedestrian appears and the coverage never mentions whether the pedestrian was visible; an obstruction audit of the highest-fine crossings ("was this crossing even visible?") is a fresh, locally grounded angle and a natural buyer conversation with the Road Safety Service.
- DA3-Small (the proposed WebGPU preview) is not the metric model; the metric one is 0.35B (about 1.3 GB per [TEAMMATE]) and belongs on a server.
### Gaps
- No Uzbek/Russian code search (the Soviet-era "visibility triangle" rule and its Uzbek successor were not located; docs.cntd.ru refused connections); no evidence beyond anecdote that kiosks/parked cars block Tashkent corners; no outdoor 20-40 m accuracy numbers for DA3-metric or MoGe-2; no Tashkent price anchor for road-safety-audit work.

---

## city-8 SchoolGate: ten-minute phone audit of school zones

### Takeaway
Nobody was found doing near-miss audits at pick-up/drop-off as a startup, but the space is not empty: SenSen sells pole cameras for school-zone parking enforcement, Miovision returns a near-miss safety study 14 days after a few days of cameras, and DataFromSky sells survey analysis of recorded footage. The phone-prop audit is an incremental twist, the payer (school or parent committee at $20) is not credible, and low oblique phone views make child-versus-car time-to-collision noisy. Verdict: sharpen.

### Cited Findings
#### Queries run
- EN standard: "school zone safety audit AI video analytics app pickup dropoff traffic conflicts startup". Two other EN phrasings failed on quota. Not run: Russian/Uzbek probes, "surrogate safety school" probe.
#### Findings
- The search "didn't find any startup focused on conflict detection, such as near-miss or pedestrian-vehicle conflict analysis at pickup and drop-off"; results were SenSen, a TRID item "Utilising Artificial Intelligence to improve safety at school crossings" and a 2023 Florida aerial-imagery school-zone inventory study [SNIPPET] — [TRID](https://trid.trb.org/View/2509203); [TechXplore](https://techxplore.com/news/2023-11-artificial-intelligence-road-safety-issues.amp).
- SenSen School Zone Monitoring: pole-insertable cameras detect double parking, overstays, bus-lane violations and blocked crosswalks, log plates and feed council citations; claims "Reduce congestion by 40% during peak pick-up times" with no method; the case study (ParkSmart at Buderim Mountain State School) was run by the school's parents-and-citizens association with Sunshine Coast Council; no pricing — [SenSen](https://sensen.ai/home/school-zone-monitoring/).
- Miovision Safety Studies: "deploy cameras for a few days", study back "within 14 days"; metrics speed, impact angle, user type, temporal separation; 50 intersections in five Canadian cities, 119 low-cost countermeasures, "85% average reduction in risk" (vendor); no price, no school example — [Miovision](https://miovision.com/blog/from-reactive-to-proactive-reaching-vision-zero-with-ai-powered-near-miss-safety-studies/). Note this contradicts the idea's claim that Miovision sells only installed sensors.
- DataFromSky: TrafficDrone, TrafficCamera, TrafficEmbedded and TrafficSurvey ("post-recording analysis of footage from drones or fixed cameras"); homepage does not mention conflicts/TTC/PET or schools; no price — [DataFromSky](https://www.datafromsky.com/). Vivacity lists "near miss incidents" among installed-sensor outputs; no school-streets text on the homepage — [Vivacity](https://www.vivacitylabs.com/).
- Tools: roboflow/trackers (SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte; Apache-2.0) — [trackers](https://github.com/roboflow/trackers); RF-DETR as above. UA-DETRAC: 10 hours, more than 140 thousand frames, 8,250 vehicles, 1.21M boxes, 24 locations in Beijing/Tianjin; the host page states no licence — [Albany lab page](https://www.albany.edu/cnse/research/computer-vision-machine-learning-lab). It contains no children and no school scenes.
- Local: 2025 child-involved crashes -171 (-9.6%) and child deaths -36 (-12.8%) versus 2024, with SBDD crediting "strengthened safety measures near schools and in mahallas" — [Anhor, 15 Jan 2026](https://anhor.uz/news/v-uzbekistane-51/). A presidential decree of 20 Nov 2023 orders photo/video enforcement near roadside schools, crossings of 4+ lanes and crash hotspots (speeding, ignoring pedestrians, jaywalking); no count, agency, vendor or deadline in the article — [Spot, 20 Nov 2023](https://www.spot.uz/ru/2023/11/20/schools-cameras/). Universities moved to 08:00 and the Ministry of Transport reports 44,000 fewer vehicles at 93 locations around universities, with no method stated and no mention of schools — [Spot, 4 Sep 2026](https://www.spot.uz/ru/2026/09/04/university-transit/).
- Cost [CALC]: 3-5 min on a T4 = 180-300 s x $0.000164 = $0.03-0.05 per audit.
### Inferences
- The pitch line "city already retimes university starts (44,000 fewer vehicles)" is about universities and an unmeasured claim; do not present it as a school-zone fact.
- Enforcement cameras near schools are a state programme; an audit product has to be the thing that decides where they go and whether they worked (before/after), not a competitor.
- A prop phone sits at 1.5-2 m height: parked cars occlude children and ground-plane error is large; robust outputs are stopped-in-lane minutes, U-turns, blocked-crosswalk minutes and pedestrians-on-carriageway counts, not child TTC.
- Filming 6 gates with consent inside 8.5 hacking hours is not realistic; 1-2 pre-filmed gates are.
### Gaps
- Number of schools in Tashkent not retrieved; rules for filming minors in Uzbekistan not checked; no Uzbek price anchor for traffic studies; UA-DETRAC licence text not found (treat as research-only).

---

## city-9 DustVis: webcams as dust and fog meters

### Takeaway
Camera-based visibility estimation is established research (roadside CCTV since the 2000s, Korean test-road system, Chinese expressway deep-learning work) and no commercial vendor surfaced, so novelty is moderate. The published state of the art on real highway images is about 82% on five visibility classes, so "visibility in metres" over-promises. A free real-data validation source exists (Finnish weather cameras with co-located visibility sensors). Verdict: sharpen to classes, validated on real pairs.

### Cited Findings
#### Queries run
- EN standard: "visibility estimation from roadside camera images fog detection highway CCTV landmark contrast commercial system". Two other EN formulations failed on quota. Not run: Russian/Uzbek, "dust storm" probes, vendor-price probe.
#### Findings
- The search found academic research and "none of the results identified a commercial product" [SNIPPET]. Hautiere et al., "Automatic fog detection and estimation of visibility distance through use of an onboard camera", Machine Vision and Applications 2006, 333 citations — [DOI](https://doi.org/10.1007/s00138-005-0011-1), [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s00138-005-0011-1). A roadside-CCTV system built to the French highway-meteorology standard NF P 99-320 — [IIETA](https://iieta.org/journals/ts/paper/10.3166/TS.28.575-603) [SNIPPET]. A Korean CCTV-based Road Visibility Measurement System reportedly beat the fog sensor on stability and correctness on a test road — [KoreaScience](https://koreascience.kr/article/JAKO201310076757377.do) [SNIPPET; page unreachable on two tries].
- State of the art on real data: Vis-MFN (Sensors, Dec 2023) used about 30 cameras near traffic weather stations on the Jiangsu section of the Beijing-Shanghai Expressway, more than 18,000 images (15,000 train, 3,000 test), five classes (0-50, 50-100, 100-200, 200-500, 500+ m); accuracy 81.76% (STCN-Net 81.10%, AlexNet 69.21%); labels from weather-station observations adjusted by practitioners; dataset not public; no MAE reported — [PMC10747611](https://pmc.ncbi.nlm.nih.gov/articles/PMC10747611/).
- Free ground truth: Digitraffic (Finland) serves over 470 weather-camera stations updated about every 10 minutes and over 350 road weather stations at 1-minute updates; the sensor list includes NÄKYVYYS_M (PWD visibility in m), NÄKYVYYS_KM and NÄKYVYYS_DSC_1/2; each camera station carries `nearestWeatherStationId` (example C14516 -> 14027, `collectionInterval` 600 s); history endpoints cover the last 24 hours; the pages I read state no licence and ask for a `Digitraffic-User` header — [Digitraffic road traffic](https://www.digitraffic.fi/en/road-traffic/); [sensor list](https://tie.digitraffic.fi/api/weather/v1/sensors); [camera station](https://tie.digitraffic.fi/api/weathercam/v1/stations/C14516); [usage rules](https://www.digitraffic.fi/en/support/instructions/).
- Local: on the night of 8-9 Jan 2026 fog covered 11 of Uzbekistan's 14 regions with visibility of 50-100 m in Tashkent, Bukhara, Karshi, Termez, Namangan and Andijan; 26 flights were diverted (the headline says 22, the body and table say 26); forecasters urged road caution; no highway-specific reporting — [Spot, 9 Jan 2026](https://www.spot.uz/ru/2026/01/09/fog-flights/).
- Decree UP-229 (24 Nov 2025) sets up a smog rapid-response system with control over vehicle emissions and dust rules for construction — [gov.uz](https://gov.uz/en/miit/news/view/105885); vehicles contribute about 16% of pollution per Kun.uz — [Kun.uz, 22 Dec 2025](https://kun.uz/en/news/2025/12/22/beyond-tashkent-air-quality-is-a-national-problem-are-policies-keeping-up?hl=en-GB).
- Tashkent Safe City runs 42,750 intelligent cameras under the Tashkent MVD protection service — [Kun.uz, 27 Jan 2026](https://kun.uz/ru/news/2026/01/27/prezident-oznakomilsya-s-deyatelnostyu-slujby-oxrany-tashkenta).
- Spot tag pages for dust storms returned 404; no Karakalpakstan dust-road evidence was retrieved. The Vaisala PWD22 page was 404, so no sensor price anchor.
- Market [CALC]: 100 cameras x $20 x 12 = $24k per year. Compute is negligible (one frame per minute is 43,800 frames per camera-month).
### Inferences
- The slider demo on a self-hazed clear frame is circular (the estimator inverts the model that generated the haze); a credible demo replays a real Finnish fog event next to the PWD sensor trace and shows the class and trend.
- Reporting 4-5 classes with calibrated confidence matches published practice; metre estimates would not be defensible under questioning.
- Night/dust: the idea's "lit landmarks and headlight halos" is research-grade; dust colour and saturation are unvalidated. State the daylight limit.
- A haze index from existing city cameras that also tracks the smog decree is a stronger local payer story than road closure advice, but it leaves the road-safety track; mention as an option only.
### Gaps
- No Uzbek road-agency fog-closure procedure or camera inventory; Digitraffic licence text not retrieved; Korean RVMS details not read; no price for forward-scatter sensors.

---

## city-10 SwerveMap: let drivers' dodging reveal road hazards

### Takeaway
Using the crowd of passing vehicles as the label for hazards is original: one extended search found only accelerometer/wheel-speed patents and pothole-avoiding-trajectory work, nothing that infers hazards from lateral dodges on fixed cameras. Risks are decoy swerves, calibration, ground truth and a thin margin per camera-month. Best originality and technical-complexity profile in this batch. Verdict: keep, with a precision test.

### Cited Findings
#### Queries run
- EN extended: "detect road hazards from vehicle evasive swerving trajectories camera pothole obstacle". Three further formulations (CCTV lateral deviation; crowdsourced evasive manoeuvres; hazard-from-swerve paper) failed on quota. Not run: Russian/Uzbek.
#### Findings
- The search "did not find a source that learns hazards specifically from evasive swerve trajectories"; closest are IMU/wheel-speed pothole patents and a pothole-avoiding-trajectory patent — [US 10754344](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10754344); [US 12043251](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12043251) [SNIPPET].
- Market warning from the saturation map: Michelin's Better Roads is discontinued by end of 2025 and phone-on-windshield road AI (Xweather RoadAI, Vialytics) exists — [TEAMMATE; Michelin page](https://mobilityintelligence.michelin.com/us/road-assessments/); [Xweather RoadAI](https://xweather.com/roadai).
- Camera access: Tashkent's 42,750 Safe City cameras are run by the MVD protection service; the January 2026 plan is to integrate intelligent transport systems and parking, vendors unnamed — [Kun.uz](https://kun.uz/ru/news/2026/01/27/prezident-oznakomilsya-s-deyatelnostyu-slujby-oxrany-tashkenta). The 2019 Safe City model shares fine revenue with the contractor until the investment is repaid [TEAMMATE, SNIPPET] — [Kun.uz 2019](https://wap.kun.uz/en/news/2019/08/21/safe-city-project-implementation-in-tashkent-to-be-completed-in-2020).
- Data/tools: RDD2022 47,420 images, 55,000+ damages, CC BY 4.0 [TEAMMATE]; elimination clips are 4K at about 140 Mbit/s with a CPU-decode bottleneck, noon-to-dusk lighting, and reuse rights for a public demo are not stated [TEAMMATE hackathon_intel]; RF-DETR-S 3.5 ms (T4 TRT FP16) and roboflow/trackers as above.
- Cost [CALC]: a T4 running 24/7 at $0.59/h is about $431 per month; at 10-20 streams that is $22-43 per camera-month, against the proposed $25. Detection (3.5 ms) is not the limit; video decode is.
- Market [CALC]: 500 cameras x $25 x 12 = $150k per year.
### Inferences
- "No labels" is not strictly true (RDD2022 names causes), and lane merges, bus stops, parked cars and speed bumps produce the same kinks; restrict to straight mid-block segments with per-lane baselines and treat false positives as a feature (illegal parking hot-spots) in the pitch.
- Lateral offsets of 0.3-1 m need calibration and enough pixels; far-field tracks will be too noisy.
- Honest margin needs an L4/edge box or lower frame rate; at on-demand T4 prices the stated price just covers compute.
### Gaps
- Second-formulation novelty searches did not run; no evidence on swerve frequency at real Tashkent defects; no camera-access terms; no night/rain results.

---

## city-11 GhostWorks: idle lane-closure ledger for road contractors

### Takeaway
The closest prior art is an FHWA/NYU project (WorkZoneX) that already detects work zones and shows durations with workers present on 900+ NYC cameras, and its own advisory board asked for the very next step this idea sells (compare detected activity with permitted work periods). Novelty is therefore moderate. Capture demands (8-hour time-lapses at five sites) are not realistic in the hacking window, and blaming contractors for "idle" needs context. Verdict: sharpen.

### Cited Findings
#### Queries run
- EN standard: "FHWA work zone camera AI detect active work zone workers equipment New York City". Two others (lane rental/contractor penalty; construction idle-time startups) failed on quota. Not run: Russian/Uzbek.
#### Findings
- WorkZoneX (NYU C2SMART for FHWA, final report 3/1/22-5/31/23, dated May 2023): 900+ NYC traffic cameras; mAP 74.1% across classes, 98.4% scene-identification accuracy, up to 89.52% size-estimation accuracy; dashboard of work-zone durations and "durations with the presence of construction workers"; stakeholders wanted to "improve street construction permit enforcement"; the advisory board "suggested comparing detected 'active' work zones with the permitted construction time periods", not yet implemented; limitation: the model confuses construction workers with traffic-enforcement agents in similar vests; area estimation fails with natural boundaries; cost to run: individual computer $2,175 one-time + $2,860/yr, local server $3,025 + $3,160/yr, cloud $1,375 + $4,885/yr — [report PDF](https://rosap.ntl.bts.gov/view/dot/72518/dot_72518_DS1.pdf) (text extracted locally).
- Other: an NDSU field system for worker/equipment near-misses and boundary intrusions — [NDSU](https://www.ndsu.edu/news/2026-01-30-ndsu-researchers-working-improve-highway-work-zone-safety); a ClearObject ClearVision work-zone sales sheet — [ClearObject](https://www.clearobject.com/wp-content/uploads/2025/10/WorkZone-Safety-1pgr_ClearVision-for-SLED.pdf) [SNIPPET]. The search found "no evidence that NYC has operationally deployed an AI work-zone camera system" [SNIPPET].
- Tashkent: Chinese-financed repair programme of 16 streets/36 km in 2026 ($36.5M), 9 streets/33 km in 2027, 15 sections/50 km in 2028 within a $271.1M Bank of China credit — [TEAMMATE, Spot 6 Dec 2025](https://www.spot.uz/ru/2025/12/06/tashkent-roads/). Decree UP-229 requires construction sites to prevent dust, water and secure exit areas and tightens control of construction-material transport — [gov.uz](https://gov.uz/en/miit/news/view/105885). Headline only: Karakalpakstan procurement violations of 32 bn soum around road-quality studies (30 Sep 2026) — [Spot roads tag](https://www.spot.uz/ru/tag/%D0%B4%D0%BE%D1%80%D0%BE%D0%B3%D0%B8/).
- Cost [CALC]: 7 min = 420 s x $0.000164 = $0.069 per zone-day. Market [CALC]: 20 zones x $100 x 12 = $24k per year, versus about $4.9k per year to run WorkZoneX city-wide on existing cameras.
### Inferences
- The "no accountability ledger found" claim should be softened: the prior art states it as next step. What is new is a local implementation plus a contractor-level score.
- Legitimate idle periods (curing, utility waits, weather, night closures) mean a raw idle score will be disputed; log reason codes and show confidence.
- Five 8-hour captures cannot be made during an 8.5 hour hackathon; use 1-2 pre-captured sites plus public time-lapses.
- Decree-driven construction dust/exit-wash compliance has a named enforcer (the ecology inspectorate) and fits the saturation map's white space; it is a better wedge than idle blame.
### Gaps
- No evidence of any lane-rental or contractor-penalty mechanism in Uzbekistan; open-vocabulary detector performance on Tashkent vests untested; no Russian query.

---

## city-12 WetLedger: leak, wash or rain from road wetness

### Takeaway
No street-level wet-pavement leak detector was found, but only because utilities use better signals (in-pipe, acoustic, thermal, satellite). A wet patch seen from a taxi is dominated by deliberate or harmless water (washing, watering, irrigation, condensate, rain), true main leaks are often invisible, and there is no taxi video feed. Verdict: kill.

### Cited Findings
#### Queries run
- EN standard: "detect water main leak from street-level video wet pavement patches vehicle-mounted camera AI utility". Not run: Russian/Uzbek, wet-surface dataset probe.
#### Findings
- "I didn't find any source describing a vehicle-mounted camera that spots water main leaks from wet pavement." Alternatives found: Electro Scan (electrical resistance testing plus in-pipe CCTV and AI), Sydney Water/UTS acoustic sensors with AI, United Utilities/Sensat UAV thermal mapping, Maynilad satellite imagery plus AI [SNIPPET] — [Trenchless Technology](https://trenchlesstechnology.com/ert-cctv-and-ai-redefine-leak-detection-in-pressurized-water-mains/); [Construction Management](https://constructionmanagement.co.uk/sensat-algorithm-predicts-water-leaks/); [Malaya](https://malaya.com.ph/business/corporate/maynilad-taps-satellite-imagery-to-detect-pipe-leaks).
- Decree UP-229 explicitly requires watering at construction sites and dust suppression, i.e. deliberate wetting is policy — [gov.uz](https://gov.uz/en/miit/news/view/105885).
- ACDC rain frames exist for segmentation tests but their licence is unverified [TEAMMATE].
### Inferences
- False-positive sources in a hot Tashkent summer (irrigation spill, street washing, car washes, air-conditioner drip, rain) are my reasoning, not sourced; persistence across days does not separate them from a slow leak.
- GPS error of 5-10 m breaks cross-pass patch matching without visual place recognition.
- A "street-washing verified" variant already has simpler tools (truck GPS) and sits outside the automotive track.
### Gaps
- No Tashkent water-loss or complaint data; no Uzbek utility contact; no wet-surface dataset checked.

---

## city-15 OverpassIgnored: prove which pedestrian bridges nobody uses

### Takeaway
The measurement is not new (several manual-plus-video studies report overpass and underpass use rates, with wildly different results by site), but a cheap diagnosis of why a crossing fails (detour factor, fence gaps) with fix quantities is a useful twist, and Tashkent is piloting crossing treatments with no evaluation plan. Simplest pipeline of the batch. Verdict: sharpen to a crossing-treatment effectiveness meter.

### Cited Findings
#### Queries run
- EN standard: "pedestrian overpass footbridge usage rate video observation pedestrians avoid crossing at grade study detour". Not run: Russian/Uzbek, commercial-vendor probe.
#### Findings
- Kathmandu (Journal of Advanced College of Engineering and Management, vol. 12, 12 May 2026): 47,783 crossings at five sites (four overhead bridges, one underpass) over 15 days by manual and video observation; underpass 97.92%, overhead bridges 93.67%, overall 83.20%; median barriers and number of steps named as key determinants — [NepJOL](https://nepjol.info/index.php/JACEM/article/view/93949).
- Pakistan (Civil Engineering Journal 2023): 80,017 crossings at four sites over four weeks; 71.83% did not use facilities; underpass 62.5% versus bridge 11.62% [SNIPPET; the paper host reset the connection on four tries] — [paper](https://civilejournal.org/index.php/cej/article/download/4061/pdf/10882). Ankara: escalator-equipped overpass 62.9%, the highest of five (Rasanen et al., cited in a CETRA 2026 paper) [SNIPPET] — [CETRA](https://cetra.grad.hr/archive/cetra2026papers/1914.pdf). Also Quezon City 2025 (video, counts, surveys) and Kampala 2025 dissertations [SNIPPET].
- The idea's claim "none found pairing bridge and at-grade counts" is contradicted: the Pakistan and Kathmandu studies report use versus non-use, which requires counting at-grade crossers; they do not quantify detour factor [SNIPPET].
- Local: pedestrians in 2,689 of 5,895 crashes Jan-Sep 2025 (about 46% [CALC]) [TEAMMATE, Kun.uz 3 Nov 2025]. Tashkent ЦОДД installed an RRFB beacon system at an unregulated crossing on Osiyo Street (30 Jun 2026) to "assess practical effectiveness under Tashkent conditions"; the article gives no before/after plan or counts and quotes FHWA research that driver yielding rises from 18-30% to 98% with RRFB — [Spot](https://www.spot.uz/ru/2026/06/30/rrfb-approach/). AI cameras for non-yielding drivers at 10 crossings (a blogger's figure of 1,700 fines in 3 days, not verified by Spot; 30 more intersections announced) — [Spot](https://www.spot.uz/ru/2025/07/13/cameras-ai/).
- Cost [CALC]: 10 min on a T4 = 600 s x $0.000164 = $0.098 per 2-hour clip. Market [CALC]: 100 sites x $150 = $15k.
### Inferences
- Opposite outcomes for the same structure type (about 11.6% bridge use in Pakistan versus 93.7% in Kathmandu) prove that site-specific evidence is needed and make a sharp pitch line.
- Same fixed-camera pipeline as the elimination task; differentiate with the diagnosis output and the before/after treatment meter.
- Capturing 4 crossings x 2 hours is not realistic in 8.5 hacking hours; pre-record 1-2 sites.
### Gaps
- No Tashkent overpass inventory, cost or complaint data; Pakistan paper not opened; no Uzbek/Russian queries; night capture quality untested.

---

## city-17 Euro4Gate: visual vintage triage for smog checkpoints

### Takeaway
The premise (a pre-2010 / non-Euro-4 vehicle ban to enforce) is unverified: it appears as a Ministry of Ecology proposal in Jan 2024, one English Kun.uz article ties it to the Nov 2025 decree, and the government's own summary of that decree lists no vehicle-age ban. Even if it exists, plate recognition plus the registry gives the exact year, appearance cannot reveal emission class for long-running Chevrolet bodies, and make/model/generation recognition is a sold commodity. Verdict: kill.

### Cited Findings
#### Queries run
- RU standard: "указ 24 ноября 2025 смог Ташкент запрет автомобили выпущенные до 2010 года Евро-4 А-80" (returned only 2024 proposals and a summary stating the Nov 2025 decree is not confirmed). Not run: EN vendor probe for vehicle-age recognition (covered by a Sighthound fetch), Uzbek.
#### Findings
- Spot, 12 Jan 2024: the Ministry of Ecology "proposes to ban fuel below Euro-4, including A-80" and "a ban on vehicles produced before 2010 may also be introduced"; everything listed is a proposal; no emission checks or enforcement mechanism described — [Spot](https://www.spot.uz/ru/2024/01/12/air-actions/); also [Kolesa.kz, 16 Jan 2024](https://kolesa.kz/content/news/v-tashkente-hotyat-zapretit-ekspluatatsiyu-mashin-starshe-2010-goda-vypuska/), which (citing Gazeta.uz) calls the pre-2010 ban the "most radical" option in a ministry work plan, i.e. a proposal, not an adopted measure.
- Kun.uz (English, 22 Dec 2025): the 24 Nov 2025 decree included "banning low-quality A-80 fuel and vehicles manufactured before 2010 that do not meet Euro-4 standards" — [Kun.uz](https://kun.uz/en/news/2025/12/22/beyond-tashkent-air-quality-is-a-national-problem-are-policies-keeping-up?hl=en-GB). The gov.uz summary of Decree UP-229 (24 Nov 2025) lists "control over vehicle emissions", "combating illegal emissions" and tighter control of construction-material transport, with no vehicle-age ban, no truck rule and no checkpoints; it links no decree text — [gov.uz](https://gov.uz/en/miit/news/view/105885). Conflict unresolved; the decree text on lex.uz was not found.
- Sighthound's vehicle recognition returns "the make, model, color, and generation of any vehicle sold from 1991 onwards"; no accuracy or price on the page — [Sighthound](https://www.sighthound.com/). Plate Recognizer lists make/model/colour at $45 per camera per month [TEAMMATE]. Make-model-year classification is a classroom task (Stanford Cars, CompCars) [BACKGROUND].
- Tashkent's Safe City cameras include face and plate recognition at city entry and exit points — [Kun.uz, 27 Jan 2026](https://kun.uz/ru/news/2026/01/27/prezident-oznakomilsya-s-deyatelnostyu-slujby-oxrany-tashkenta). ANPR plus registry lookup is the standard route [TEAMMATE].
- Data plan: Avtoelon robots.txt blocks parameterised URLs (with listed exceptions), account/ajax paths, and names no photo paths — [robots.txt](https://avtoelon.uz/robots.txt); OLX robots.txt blocks `*/search/`, `*/list/next/`, `*/i2/*` and has no scraping statement — [robots.txt](https://www.olx.uz/robots.txt); site terms of service not read. Listing supply: Avtoelon 29,497 car listings (Chevrolet 19,923), OLX.uz 67,961 [TEAMMATE LIVE counters].
- Fleet: Chevrolet 83.2% of 2025 new sales, Cobalt 30.1%, Cobalt plus Damas 55.6% [TEAMMATE, UNOFFICIAL].
- Cost [CALC]: 0.2 s x $0.000164 = $0.00003 per vehicle. Market [CALC]: 20 checkpoints x $50 x 12 = $12k per year.
### Inferences
- Emission class is set by engine and calibration, not by body shape; models sold for years with a near-identical body (for example the Nexia and Damas lines [BACKGROUND, not verified here]) make a 2010 boundary visually undetectable, and seller-stated years add label noise.
- Registry lookup beats a visual guess wherever the plate reads; the residual niche (foreign plates, unreadable plates) is small.
- A year-from-photo demo is instant for judges but is the stock make/model/year project, which hurts originality scoring.
### Gaps
- Decree text not retrieved; Uzbek registry fields (year, Euro class) unknown; Sighthound coverage of Uzbek-market models unknown; site terms of service unread.

---

## city-18 StopCount: independent proof signal retiming cut stops

### Takeaway
Tashkent's traffic centre is rolling out "green waves" and publishing travel-time gains on Telegram with no method, so an independent measurement has a real hook. But Google's Project Green Light already measures stops saved for free in 100+ cities, ATSPM covers arrivals/splits from controller logs, and the idea's physical design (two cameras, per-vehicle stop counts, plate-free re-ID, before/after clips the team does not have) does not hold up. The claimant is also the operator, which weakens the payer story. Verdict: sharpen.

### Cited Findings
#### Queries run
- EN standard: "Project Green Light Google stops reduction intersections emissions". Not run (quota): EN probes for camera-based arrival-on-green/stop-rate products and INRIX/StreetLight comparison; Russian/Uzbek.
#### Findings
- Project Green Light: "now available in over 100 cities worldwide"; early numbers "potential for up to 30% reduction in stops" and 10% GHG (modelled with a US DOE emissions model, single vehicle type); an impact-analysis report is posted to the city's dashboard a few weeks after a change; no extra hardware; free "during the early research phase"; named cities Haifa, Kolkata, Hamburg, Boston; Uzbekistan is not named on the page — [Google Research](https://sites.research.google/gr/greenlight/).
- UDOT ATSPM (Apache-2.0): metrics include vehicle volumes, arrivals on red, pedestrian delay and split failures from high-resolution signal-controller logs (ASC3, Peek, Siemens, Trafficware decoders); the page does not list arrivals on green or a coordination diagram — [ATSPM](https://github.com/udotdevelopment/ATSPM). Miovision: study from a few days of cameras — [Miovision](https://miovision.com/blog/from-reactive-to-proactive-reaching-vision-zero-with-ai-powered-near-miss-safety-studies/).
- Tashkent claims, all from the ЦОДД traffic centre via its Telegram channel, none with a method: Buyuk Ipak Yuli green wave (29 Jul 2026): 13 intersections, travel toward the centre -29% (8.5 to 6 min), reverse -20% (10 to 8 min), no stop counts, no financing named — [Spot](https://www.spot.uz/ru/2026/07/29/green-corridor/). Small Ring Road 5 km (20 Sep 2026): 12 signal objects, 11 min 30 s to 9 min 40 s, "fewer forced stops" without a figure — [Spot](https://www.spot.uz/ru/2026/09/20/new-green-corridor/). Fitrat/Geydar Aliyev (11 Aug 2026): green for the busiest direction nearly doubled (phases of 35 s and 28 s before), "queues shorter", no numbers — [Spot](https://www.spot.uz/ru/2026/08/11/traffic-lights/). Amir Temur square wave (22 Aug 2026): 5 signals with fixed offsets, no numbers for this project, no financing named, no verification; the same article repeats that a 1.7 km Small Ring Road section went from about 5 min to 2 min 45 s (-45%) with no source beyond Spot's own reporting — [Spot](https://www.spot.uz/ru/2026/08/22/amir-timur-square/). Citywide: 3,946 adaptive traffic lights at 271 intersections.
- The idea's "World Bank-backed green-wave corridors" is not supported: the three corridor articles name no funder, and the World Bank's $200M transport loan (board approval reported 23-24 Mar 2026) finances 91 km of the M41 highway in Surkhandarya, about 180 bridges and drainage structures, road-safety measures and rail/strategy reforms; it does not mention Tashkent signals, ITS or green waves, and describes no results-based verification — [Spot](https://www.spot.uz/ru/2026/03/24/wb-transport/).
- Re-ID context: Chevrolet 83.2% and Cobalt plus Damas 55.6% of 2025 new sales [TEAMMATE, UNOFFICIAL].
- Compute: idea's own estimate $3-15 per corridor-day; T4 at $0.59/h means 5-25 GPU-hours at that range [CALC]; not independently validated.
- Market [CALC]: 20 corridors x $500 = $10k.
### Inferences
- A floating-car run (phone GPS) already gives travel time and stops per run without any CV; CV adds arrival-on-green share and queue at each stop line, which needs a camera per approach, not two corridor-end cameras.
- Per-vehicle appearance re-ID is ambiguous in a Cobalt/Damas-heavy fleet (and white-dominated colour mix is likely but unsourced); platoon-level travel-time distributions are the safer link.
- Waves are already live, so "before" footage cannot be shot now; baselines would have to come from the opposite direction, off-peak plans or ЦОДД's own pre-change data.
- Selling "independent proof" to ЦОДД is adversarial; the credible buyers are the hokimiyat leadership, a lender's verification agent or journalists.
### Gaps
- No evidence of any lender verification requirement for Tashkent signals; Tashkent controller logging capability unknown; no camera-based arrival-on-green vendor check; no INRIX/StreetLight comparison.

---

## city-19 GlareClock: sun-glare calendar for signals and signs

### Takeaway
The sun-position half is a free commodity (SunCalc-type tools), the phantom/glare problem is old and treated with cheap hardware (visors, lamellae), and the CV "washout score" has no ground truth. Phone compass error alone turns a date calendar into guesses. Demo is instant, but problem size, payer and moat are close to zero. Verdict: kill.

### Cited Findings
#### Queries run
- EN standard: "traffic signal sun glare sun phantom visibility washed out signal head analysis tool sun position azimuth crashes". Not run: Russian/Uzbek, signal-visibility standard probe.
#### Findings
- The search found "no sun-position or solar-geometry analysis for a particular intersection". SWOV documented a 1985 study of disability glare and the "Sun Phantom" caused by sunlight on an internal reflector; fixes are visors, an anti-phantom cross and lamella gratings; Econolite sells 8 and 12 inch visors — [SWOV](https://swov.nl/en/publicatie/analysis-effects-sun-phantom-conspicuity-directional-green-arrow-traffic-signals) [SNIPPET]; [Econolite](https://www.econolite.com/support-page/technical-documentation/signals/8-12-traffic-signal-visors) [SNIPPET]; [visor history](https://kbrhorse.net/signals/visors01.html) [SNIPPET].
- SunCalc shows sun altitude, azimuth and the sun's path for a date and place and appears free; no mobile compass or AR mode mentioned on the page — [SunCalc](https://www.suncalc.org/). The PhotoPills page returned 403, so its AR sun-path feature is not verified.
- Tashkent has 3,946 adaptive traffic lights — [Spot](https://www.spot.uz/ru/2026/07/29/green-corridor/). Market [CALC]: 3,946 x $10 = about $39k one-time.
- Heading sensitivity [CALC]: near the equinox at 41 degrees N the sunrise azimuth moves about 0.5 degrees per day (1/cos(41.3 degrees) x 0.40 degrees per day), so a 10 degree compass error shifts the predicted date by roughly 19 days; typical phone heading accuracy is about 5-15 degrees [BACKGROUND].
### Inferences
- The phantom effect described by SWOV depends on an internal lamp reflector; modern LED heads largely lack one, so the original problem may be much smaller today (unverified reasoning).
- "Washout" from phone exposures under direct sun cannot be mapped to driver perception without luminance measurements; two annotators' labels are subjective.
- A fix order costs a visor; no agency will pay $10 per head for a calendar.
### Gaps
- No crash data on sun glare at signals in Uzbekistan; PhotoPills/Sun Seeker AR features not verified; no traffic-light dataset licence check (LISA is non-commercial per [TEAMMATE]).

---

## Cross-cutting findings

- **Who pays.** Uzbek city CV has been bought through fine-revenue sharing (Safe City, 2019, contractor repaid from fines [TEAMMATE, SNIPPET]) and foreign credit (Bank of China $271.1M for repairs [TEAMMATE]). Audit and measurement products (city-7, 8, 9, 11, 12, 15, 18, 19) generate no fines, so $10-$500 price points to the khokimiyat are unproven. Plausible buyers: audit consultancies on World Bank/ADB projects, donors, contractors who need proof, or a pilot sponsored by the traffic centre.
- **Unverified official numbers are the local opening.** ЦОДД publishes travel-time gains on Telegram without method (city-18); the Ministry of Transport reports 44,000 fewer vehicles without method (city-8 context); the RRFB pilot has no evaluation plan (city-15). "Independent measurement" is a theme judges can verify live, but the claimants are the operators.
- **Stock pipeline warning.** [TEAMMATE hackathon_intel] notes reviewers have seen "YOLO + tracker + rules on a fixed camera + live-upload demo" many times; ideas 8, 10, 11, 15 and 18 share it, so the output artefact (report, ranked hot-spots, evidence pack) and the validation story must carry the originality score.
- **Time and constraint realism.** About 8.5 on-campus hacking hours: every capture plan (40 junctions, 6 school gates, five 8-hour time-lapses, four 2-hour crossings) must shrink to 1-2 pre-recorded sites. The brief demands night, rain and local-vehicle evidence; daylight-only ideas (7, 8, 19) need an explicit limit and at least one adverse-condition test.
- **Suggested order for this batch:** city-10 (keep), then city-15 and city-7 (sharpened), then city-8, city-9, city-11, city-18; kill city-12, city-17, city-19.
