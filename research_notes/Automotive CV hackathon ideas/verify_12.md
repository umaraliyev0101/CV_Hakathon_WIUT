# Verification batch 12: ten city-roads and EV/CNG CV ideas (WIUT Hackathon 26, Tashkent), adversarial due diligence

Method note (not a findings section). Date 2026-10-10. Scores are skeptical defaults on a 1-5 scale. FX about 11,800 UZS per USD (teammate fact sheet `uzbekistan_market.md`). Judging weights (teammate `hackathon_intel.md`): Innovation 25%, Technical Complexity 25%, Impact and Practicality 25%, Presentation 15%, Business 10%.

Search budget. The shared 200-call WebSearch quota was almost permanently exhausted. About 75 attempts were made; 12 ran:
1. EN "sidewalk width measurement AI street-level imagery obstruction detection"
2. EN "pedestrians forced into roadway by parked cars on sidewalk computer vision detection"
3. RU "заправка метаном пассажиры должны выйти из автомобиля правила АГНКС Узбекистан"
4. RU "нейросеть пожарный проезд двор парковка блокирует проезд спецтехники камера"
5. EN "phone-based EV underbody scan battery pack damage detection app dealer service"
6. EN "parking revenue control video analytics attendants cash lot vehicle counting audit"
7. EN "EV charging site audit computer vision photos nameplate checklist inspection startup"
8. EN "storm drain clog detection computer vision"
9. EN "home EV charger safety inspection photo AI extension cord fire risk"
10. EN "petrol station camera AI detects smoking customers phone use Hikvision Dahua forecourt"
11. EN "AI camera detects vehicles blocking fire lane emergency access road residential compound alert"
12. EN "AI detects cars parked on sidewalk forcing pedestrians into street computer vision city enforcement"

Refused by quota (so NOT run, and the novelty scores for these are lower confidence): natural-language traffic video search (4 variants), Chinese EV parts identification (3), other gas-station smoking/phone/pump-lock AI variants (7), a RU sidewalk-parking neural-net query, a Chinese-language fire-lane query, plus first variants of the storm-drain, EV-charging-inspection and parking-attendant queries. Everything else comes from direct WebFetch of primary pages, `gh search repos` / GitHub topic pages (GitHub API rate limit hit after about 25 calls), local `pdftotext` of the UrbanVGGT PDF, and teammate notes in this folder (`tech_radar.md`, `hackathon_intel.md`, `saturation_map.md`, `uzbekistan_market.md`, `verify_02/07.md`). Tags: (S) = claim seen only in a search-result summary, page unreadable (403, empty or connection refused); "teammate" = taken from a sibling notes file and not re-verified by me; [CALC] = my arithmetic; [INFERENCE] = my judgment. Absence of a search hit is weak evidence.

Fetches that failed: mdpi.com paper (403), occompt.com audit PDF (403), mass.gov audit PDFs (403), savannahga.gov (connection refused), mydigitalpublication.com vendor article (empty), Hikvision gas-station pages (empty, two URL variants), Dahua gas-station page (first URL guess: category names only; a second URL worked), USPTO patent PDF 12202521 (scanned image, no text).

---

## city-20 ForcedOnRoad: blocked sidewalks pushing pedestrians into traffic

### Takeaway
No product that counts "pedestrian forced into the carriageway" events was found, and monocular sidewalk-width measurement is an active 2026 research topic, but its published error (0.25 m MAE) is too coarse to rank blocks around a 1.2 m threshold without error bands. The payer is municipal, and one rush-hour pass over 20 blocks gives few events per block. Verdict sharpen. Scores: novelty 4, feasibility 3, demo 3, business 2, moat 2, local fit 4.

### Cited Findings
- UrbanVGGT (Tan and Zhang, arXiv 2603.22531, submitted 23 Mar 2026, v2 19 Aug 2026) estimates metric sidewalk width from one street-view image with segmentation, feed-forward 3D reconstruction, ground-plane fitting and camera-height scale calibration; on a Washington D.C. benchmark of about 300 images MAE is 0.252 m, 49.9% of estimates are within 0.25 m and 95.5% within 0.50 m; the authors say cross-city validation is still needed before planning use — [arXiv abs](https://arxiv.org/abs/2603.22531)
- Same paper (read locally from the PDF): without camera-height calibration MAE is 1.571 m; the optimum assumed camera height is 2.5 m (vehicle mount) and it assumes a fixed mount height; native-metric backbones are worse (UniDepthV2 0.289 m, MapAnything 0.463 m, DepthPro 0.768 m, Depth-Anything-V2-m 1.712 m; MapAnything with calibration 0.334 m); listed failure modes are occlusion by parked vehicles, very narrow sidewalks (small errors become large relative errors) and complex curb geometry; the 3-city SV-SideWidth prototype was not validated against ground truth — [arXiv PDF](https://arxiv.org/pdf/2603.22531)
- A 2025 MDPI study used Segment Anything to segment sidewalks, obstructions and barriers in 61 street images and notes that an earlier encroachment model still needed manual measurement (S, page 403) — [MDPI](https://www.mdpi.com/2076-3417/15/10/5261); a Vienna thesis reached 64% agreement with a reference dataset, errors concentrated in occlusion and intersections (S) — [TU Wien](https://repositum.tuwien.at/handle/20.500.12708/229239); an Amsterdam LiDAR study estimates obstacle-free width (S) — [arXiv 2211.04108](https://export.arxiv.org/pdf/2211.04108)
- The only "forced into the road" hits are patents and a dissertation on the vehicle side: a patent application on predicting a pedestrian stepping into the drivable space when a sidewalk is blocked by parked vehicles or garbage cans, and a University of Malta dissertation on pedestrians hidden between parked cars (both S, search summary; the USPTO PDF is a scanned image) — [USPTO 12202521](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12202521); [Malta](https://www.um.edu.mt/library/oar/handle/123456789/123527)
- Search 12 ("AI detects cars parked on sidewalk forcing pedestrians into street computer vision city enforcement") found general illegal-parking video analytics but nothing aimed at pedestrians forced into the street: an NTU/AI Singapore mobile-enforcement prototype (readiness level 7), Korea's MOVI edge-AI system (about 98% detection; incidents of 5 minutes or longer in peak hours fell up to 99% in the test area, drivers are nudged rather than fined), Santa Monica's camera-equipped vehicles flagging bike-lane blockages with officer review, Vehant's "No Parking" product and a Kragujevac fixed-camera paper — [AI Singapore](https://aisingapore.org/tech-offers/real-time-detection-of-illegal-parking-offences-using-video-analytics/); [Korea Herald](https://biz.heraldcorp.com/article/10893810); [Santa Monica](https://letsdatascience.com/news/santa-monica-uses-ai-to-enforce-bike-lanes-26356329); [Vehant](https://www.vehant.com/products/no-parking/)
- Project Sidewalk is volunteer labelling, about 40 city sites in 8 countries, Chicago site 194,086 labels and 228,396 validations; machine labels are part of its pipeline but accuracy and width measurement are not described — [Project Sidewalk Chicago](https://sidewalk-chicago.cs.washington.edu/)
- Open-source/open-data adjacency is GIS-based: `meliharvey/sidewalkwidths-nyc` (130 stars, NYC widths from sidewalk outlines), Berlin and Amsterdam width repos, and a 2-star ResNet-50 sidewalk-obstruction classifier; GitHub topics `sidewalk-detection` and `parking-violation` have 0 repos — [sidewalkwidths-nyc](https://github.com/meliharvey/sidewalkwidths-nyc); [DIP-Research obstruction](https://github.com/sydsborn/DIP-Research--Sidewalk-Obstruction-Detection); [topic sidewalk-detection](https://github.com/topics/sidewalk-detection)
- Local pedestrian exposure: pedestrians were involved in 2,689 of 5,895 crashes in Jan-Sep 2025 (about 46%, my division) — [Kun.uz](https://kun.uz/en/news/2025/11/03/over-1300-people-killed-in-road-accidents-across-uzbekistan-since-january)
- Tashkent has about 193 vehicles per 1,000 residents, roughly double the national average (article dated 20 Jan 2026) — [Times of Central Asia](https://timesca.com/tashkent-choked-by-smog-what-2025-revealed-and-how-residents-are-responding/)
- State enforcement: approved list of camera-recordable offences includes "breaching stopping and parking rules" but does not name sidewalks, crossings or fire lanes; fines within one month — [Anhor](https://anhor.uz/news/traffic-cameras/). Paid street parking is checked by inspectors with tablets and windscreen notices, and the operator (Streetpark Systems) is developing photo/video systems — [Kun.uz, 2 Jun 2025](https://kun.uz/ru/news/2025/06/02/v-tashkente-zarabotali-platnyye-ulichnyye-parkovki-chto-dumayut-ob-etom-jiteli)
- Model facts (teammate): MoGe-2 MIT weights, 60 ms/image on A100; SAM 3 848M params under the SAM License with gated checkpoints — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md); [SAM 3 repo](https://github.com/facebookresearch/sam3)
- Compute price anchor: Modal T4 $0.000164/s (about $0.59/h) — [Modal pricing](https://modal.com/pricing)

### Inferences
- [CALC] 1 km at one frame per 5 m is 200 frames; at 0.4 s per frame on a T4 that is 80 s, about $0.013 per km; at 1 s per frame about $0.033 per km. The idea's "about 1 US cent per km" holds within about 3x. Walking labour is the real cost: 1 km at 4 km/h is 15 min, about $0.9 at the 7.09M UZS/month average wage (teammate).
- [INFERENCE] The published error (MAE 0.25 m, only half within 0.25 m, calibrated to a 2.5 m vehicle mount) means a block whose true free width is 1.0-1.4 m can land on either side of a 1.2 m threshold; a handheld phone has no fixed camera height, so the team needs a reference marker or a pole mount and must report per-block error bands.
- [INFERENCE] One rush-hour pass per block yields very few step-into-lane events (assumption, no local data), so "events x speed limit" rankings would be dominated by sampling noise; repeated passes or fixed cameras are needed.
- [INFERENCE] Buyers: the paid-parking concessionaire and the hokimiyat. Both are single-buyer procurement; $60 per street-km is small. The state already fines camera-detected parking breaches, so the product is a prioritisation tool, not an enforcement tool.

### Gaps
- RU query on sidewalk-parking neural nets refused by quota; no Chinese-language probe for sidewalk-occupancy analytics.
- No Tashkent data on how often pedestrians are forced onto the road; no sidewalk-parking fine or minimum sidewalk width rule found.
- MoGe-2 accuracy for sidewalk width not benchmarked anywhere I found.

---

## evcng-10 Ruxsat: charger-site permit audit from video

### Takeaway
No product that audits EV-charger sites from a walk video with nameplate OCR was found, but the March 2026 Tashkent disconnections were about missing documents, power contracts and grid load, which a 90-second video cannot show, and no published permit checklist (extinguisher, signage, bollards) exists in the draft rules I could read. The checkable parts that match the rules are nameplate kW, the metrological verification mark and a geotagged site registry. Verdict sharpen. Scores: novelty 3, feasibility 4, demo 3, business 2, moat 2, local fit 4.

### Cited Findings
- March 2026 Tashkent inventory: reviews to put charger operations in order, assess connections and measure grid load; 12 working groups from 3 Mar; over 300 chargers inventoried by 6 Mar; 101 stations temporarily disconnected for lacking required documents; 62 owners had no electricity supply contracts (overlap with the 101 not stated); about 1,300 fast chargers above 20 kW in the city; the concern was overload from connections made without preliminary calculations or technical conditions; the article mentions no fire extinguishers, signage, breakers or bollards — [Spot, 6 Mar 2026](https://www.spot.uz/ru/2026/03/06/chargers-review/)
- Draft resolution (public consultation until 17 May 2026): 4,000 stations nationwide by end-2026; every station must pass a technical audit every two years; all charging equipment must pass mandatory metrological inspection; operating unverified or uncertified charging points is prohibited; the article mentions no fire safety, signage or extinguisher requirement and does not say who inspects — [Kun.uz, 5 May 2026](https://kun.uz/en/news/2026/05/05/uzbekistan-to-expand-ev-charging-network-to-4000-stations-by-end-of-2026)
- EV fleet 122.7k on 1 Jul 2026, Tashkent city 88,496 (about 72%) — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4)
- Search for charger-site audits by photo with nameplate OCR found no such product; adjacent items: SpotGenius camera AI with Zevtron watches stalls in operation (misuse, compliance), a Pionix ANPR demo, Self Inspection checklists (EV cable-in-trunk step), Inspektlabs "any physical asset" positioning from a directory snippet — [Mann Report](https://www.mannpublications.com/mannreport/2025/09/03/zevtron-partners-with-spotgenius-to-monitor-ev-charging-spaces-with-vision-ai-technology/); [Techcrunch Self Inspection](https://techcrunch.com/?p=2961386) (S)
- ChargerHelp: AI on charger telemetry plus field technicians; page mentions no photo or video verification — [ChargerHelp](https://www.chargerhelp.com/). Charge Saint: drivers submit guided photo evidence of faults/ICEing, "AI-enhanced" verification, owner dashboard; no site audits — [Charge Saint](https://terrapinn-cdn.com/exhibition/solar-storage-live/exhibitor-charge-saint.stm)
- Inspektlabs page lists vehicle verticals only (insurance, fleet, rental, remarketing, repair), per-inspection pricing plus integration fee, white-label forms/questionnaires — [Inspektlabs](https://www.inspektlabs.com/). OpenSpace sells construction walk-capture (360, phone) with progress verification; no compliance checklist workflow described — [OpenSpace](https://www.openspace.ai/)
- Qmerit (home EV charger installs, part of Schneider Electric): 15,000+ electricians, "AI-powered project form" taking a panel photo; the page describes no installation-photo QA — [Qmerit](https://www.qmerit.com/)
- Model cost (teammate arithmetic from list prices): Claude Haiku 5.5 at $0.10/$0.50 per M tokens, about $0.00031 per 1000x1000 photo; a 12-photo inspection about $0.004 (Haiku 5.5) to about $0.014 (Gemini 3.5 Flash-Lite) — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md); [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- PP-OCRv6 (PaddleOCR 3.7.0, 2026-06-11, Apache-2.0, English and Chinese tags) and Qwen-Image-Edit-2511 (Apache-2.0) are real, licensable components (teammate) — [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

### Inferences
- [INFERENCE] The pitch line "avoids disconnections like March 2026's 101 chargers" is mostly unsupported: the stated causes were paperwork and grid load. A nameplate-kW read plus a photographed meter/feeder supports only the "declared versus installed capacity" part.
- [INFERENCE] The permit checklist is the product, and no authority has published or accepted one. Without a hokimiyat or operator sign-off the output is a self-audit with no legal weight.
- [CALC] Market: if the 4,000-station target is met and each station is audited every two years, about 2,000 audits a year x 200k UZS (idea's price, about $17) is about $34k a year; compute cost per audit is under 2 cents, so margin is high but the pool is small.
- [INFERENCE] Staged "poster wall" demo shows the VLM checklist works, not that it works on real sites; filming 15 real chargers (the idea's plan) is needed to make the claim honest.

### Gaps
- Actual regulator checklist and inspecting body unknown (two Uzbek articles read, neither lists physical-safety items); PKM-732 and PP-269 decree texts not read.
- No Chinese/Russian-language probe for charger-site inspection tools; first EN query variants hit the quota.
- Metrological verification mark format on Uzbek chargers unknown.

---

## evcng-12 Nazorat: parking cash-leak radar from CCTV

### Takeaway
Auditors already recommend (and sometimes perform) manual vehicle counts against attendant receipts, and ANPR vendors sell dwell logs and over-time alerts, so the concept is established; the plate-free re-ID twist is technically the hardest way to get the number and is unnecessary, because expected revenue follows from an occupancy-time integral. Cheap ANPR ($35 per camera-month) is the obvious rival. The leakage premise itself is supported by audits. Verdict sharpen. Scores: novelty 2, feasibility 3, demo 2, business 3, moat 2, local fit 3.

### Cited Findings
- Audit evidence of attendant leakage (US, search-result summaries; the PDFs returned 403): an Orange County (Florida) Convention Center audit counted cars and compared with point-of-sale records and found no parking charge recorded for 17% (9 of 54) of observed vehicles on the first occasion and 33% (70 of 212) on the second; a Massachusetts state audit of MBTA said it could not confirm contractor controls over more than $22M in cash and recommended a system to reconcile vehicles entering/exiting with collections; a Savannah review used video footage, deposit logs and cash counts (S) — [Orange County Comptroller](https://www.occompt.com/DocumentCenter/View/629/Report-No-465---Audit-of-the-Orange-County-Convention-Center-Parking-Revenue-PDF); [Mass. Auditor](https://www.mass.gov/doc/massachusetts-bay-transportation-authority-controls-over-parking-lot-revenue/download); [Savannah](https://savannahga.gov/DocumentCenter/View/15051)
- ParkPow (Plate Recognizer): vehicle visit logs and per-vehicle history, dwell time, alerts for unauthorized vehicles and over-time stays, open APIs to "your payment system"; the page does not describe revenue reconciliation — [ParkPow](https://parkpow.com/)
- Plate Recognizer list prices: Stream $35 per camera-month, Stream + MMC $45 (adds make, model, colour, direction, dwell time); Snapshot 2,500 free lookups/month, $50 for 50k; gate controller $10 per gate-month; "trained in over 90 countries", Uzbekistan not named — [Plate Recognizer pricing](https://platerecognizer.com/pricing/)
- Parquery: any-camera occupancy, "find offenders", qualitative revenue claims, plate-blurring privacy filter, more than 80 projects in 30+ countries, quote-based pricing; the page does not describe payment matching — [Parquery](https://www.parquery.com/)
- Tashkent paid street parking: operator Streetpark Systems (Poytaxt Parking app), five zones at 12,000/10,000/8,000/6,000/5,000 UZS per hour, first 5 minutes free, inspectors check payment on tablets and leave windscreen notices, the operator is developing mobile and stationary photo/video systems, an attendant said most drivers leave without paying; page does not say whether cash is accepted — [Kun.uz, 2 Jun 2025](https://kun.uz/ru/news/2025/06/02/v-tashkente-zarabotali-platnyye-ulichnyye-parkovki-chto-dumayut-ob-etom-jiteli)
- Vehicle re-ID: GitHub topic `vehicle-re-identification` has 4 repos; benchmark sets are VeRi-776 and VehicleID (licences not shown on the page) — [GitHub topic](https://github.com/topics/vehicle-re-identification)
- Fleet homogeneity (teammate, UNOFFICIAL): Chevrolet 83.2% of 2025 new sales, Cobalt 30.1% — [BestSellingCarsBlog](https://bestsellingcarsblog.com/2026/02/uzbekistan-full-year-2025-chevrolet-cobalt-at-30-1-share/)
- Compute is trivial: RF-DETR Nano 2.3 ms on T4 TensorRT FP16, Apache-2.0 (teammate) — [RF-DETR](https://github.com/roboflow/rf-detr)

### Inferences
- [INFERENCE] Revenue over a shift needs only the occupancy-time integral (Little's law: average occupancy x time x tariff for a flat hourly tariff), not per-car pairing. Per-car re-ID across two cameras for same-model, same-colour cars (Cobalt, Damas, Nexia dominate) is the weakest part and most of the pairs would be dropped as ambiguous.
- [CALC] Rival economics: Plate Recognizer Stream for 2 cameras is $70 a month against the idea's $34 lot-month fee; an ANPR rival gives exact pairing. The idea's "$4 per camera-month" counts only the $250 box over 36 months and omits cameras, enclosure, power and installation.
- [CALC] Break-even: at 10% recovered leakage a $34 monthly fee needs about $340 of monthly lot revenue, about 4M UZS, i.e. roughly 13 paid two-hour stays a day at 5,000 UZS/h; so if leakage exists, price is not the obstacle. The audits above show 17-33% unrecorded vehicles in two US cases, so 10% is not extreme, but there is no Uzbek data.
- [INFERENCE] In Tashkent the visible leakage is driver non-payment on app-paid curbside parking (the operator's problem and the operator is building its own cameras), not attendant skimming; cash-and-attendant lots are the idea's own assumption.
- [INFERENCE] The demo is a simulator with declared skim rates; it shows the dashboard, not that real skimming is detected.

### Gaps
- Number and revenue of attendant-run cash lots in Tashkent unknown; no owner interview; first EN query variants refused by quota.
- Vendor article on "independent counting for revenue verification" could not be read (empty fetch); vendor not identified.
- VeRi-776 / VehicleID licences not verified.

---

## wild-11 RuleCam: Uzbek-language traffic rules without retraining

### Takeaway
Natural-language video search and VLM alert verification are now free blueprints (NVIDIA VSS, Apache-2.0) and commercial features (Spot AI Iris, Camio), and the city-traffic-centre buyer is state-run and tied to existing Chinese/Kazakh camera stacks. The Uzbek-language rule box is a thin twist. It does continue the elimination task, which the brief lists as a direction. Verdict sharpen to one customer and one decision. Scores: novelty 2, feasibility 3, demo 4, business 2, moat 2, local fit 4.

### Cited Findings
- NVIDIA Video Search and Summarization blueprint: natural-language search over streams or archives (alpha), alert verification (perception alerts verified by a VLM to cut false positives), real-time VLM alerts; "Smart City" is one listed application; Apache-2.0 repo; models Cosmos3 Nano Reasoner and Nemotron 3.5 Lightning 30B A3B; hosting NIMs locally needs an NVIDIA AI Enterprise developer licence; the page does not mention custom event rules — [NVIDIA VSS](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
- Spot AI: Iris lets you "ask any camera a question in plain language"; works with existing IP cameras; sectors include government and car washes, no traffic sector; no price shown — [Spot AI](https://www.spot.ai/). Camio: cloud smart search and personalised alerts on existing cameras, policy-compliance framing, pay-for-use, no price — [Camio](https://www.camio.com/)
- Hikvision "large model" checkpoint cameras claim 75% fewer false alarms on seatbelt/phone detection (vendor, teammate) — [AAP release](https://www.aap.com.au/aapreleases/cision20250730ae40557)
- Elimination task (teammate): 14 fixed classes, fixed camera, open VLMs (Qwen-VL, InternVL) named as allowed, extra-credit hints about operator dashboards and "anything a city traffic centre would want" — [Elimination PDF mirror](https://raw.githubusercontent.com/bobomurodovd44/traffic-vision/master/WIUT%20Hackathon%20_%20CV%20Track%20Elimination%20Task.pdf)
- SAM 3: text or exemplar prompts segment all instances of a phrase, 848M parameters, tracker for video, SAM License, gated checkpoints — [SAM 3](https://github.com/facebookresearch/sam3)
- Qwen3 lists "Northern Uzbek" among 119 languages — [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- Modal pricing: L4 $0.000222/s (about $0.80/h) — [Modal pricing](https://modal.com/pricing)
- State side: 399,269 administrative warnings 4 Feb-10 Jul 2026 (seat belt, phone, speeding, parking, jaywalking; detection method not stated), penalty points since 1 Apr 2026 — [Anhor](https://anhor.uz/news/v-uzbekistane-pochti-400-tysyach-voditelej-poluchili-preduprezhdeniya-za-narusheniya-pdd/); national camera/radar list of 1,831 points and a Chinese (Maxvision) national plate system (teammate) — [Anhor](https://anhor.uz/news/radar-2/); [Hackmag](https://hackmag.com/news/uz-license-plates)
- Dataset licensing flag (teammate): BDD100K toolkit is BSD-3 but the data terms could not be verified; UA-DETRAC and DoTA terms not checked — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)

### Inferences
- [CALC] Cost claim: at $0.80/h an L4 second costs $0.00022, so 1.5 s per candidate is $0.00033 and the idea's $10 per camera-month equals about 30,000 candidates a month (about 1,000 a day). A loose rule such as "marshrutka stops in right lane" on a busy junction can exceed that, so cost per camera depends on rule looseness and is unbounded unless capped.
- [INFERENCE] Famous: an open-vocabulary "describe the event" feature is the 2026 default in video analytics (NVIDIA, Spot AI, Camio). A traffic-centre buyer is the least startup-friendly customer in Uzbekistan (closed Maxvision/Huawei/Sergek networks); ports, yards and car markets are small.
- [INFERENCE] Risk on the demo: a 4B VLM verifying "stops in right lane" will hallucinate; precision/recall must be reported on the 20 hand-labelled rules or the claim "no retraining" is untestable.
- [INFERENCE] The strongest argument is brief-alignment (continues the elimination task, local-language rules). The weakest is moat.

### Gaps
- Natural-language traffic-video search queries (5 variants) refused by quota; Nexar, Hayden and Twelve Labs not checked.
- No source for "MVD is extending AI cameras by year-end"; the state's AI-camera rollout is evidenced only via teammate notes.
- Not tested: Qwen3-VL-4B's Uzbek instruction following for rule parsing.

---

## matrix-4 Lane Clear: fire-truck access audit for crowded courtyards

### Takeaway
The conventional tool is manual CAD swept-path software (AutoTURN), and no phone-walkthrough passability auditor was found in EN or RU. The adjacent commercial product is Dahua's fixed-camera "fire lane parking detection" (dwell-time alarm on a watched lane, 99% vendor-claimed car detection; no width or passability), and Russia's AI parking enforcement works on road snapshots, not courtyards. The risk is metric accuracy from a phone video (published sidewalk-width error with MapAnything is 0.33-0.46 m), which is tolerable for flagging pinch points against a 3.5 m default if a reference scale is used. The demo is vivid and the pain is real in Soviet-era micro-districts. Verdict keep. Scores: novelty 4, feasibility 3, demo 4, business 2, moat 2, local fit 4.

### Cited Findings
- AutoTURN (Transoft): swept-path analysis in CAD (Pro) or browser (Online); libraries of over 900 manufacturer and 1,800 standard vehicles, a "CITY - EMERGENCY" library listed in a 2024 update; IntelliPath auto-generates drive paths from available space; no AI and no phone/video input; quote-only pricing — [AutoTURN](https://www.transoftsolutions.com/autoturn)
- Dahua "fire lane parking detection" (search 11): AI video analysis watches fire lanes around the clock and alarms only when a vehicle stays past a set time (dwell time), vendor-claimed 99% detection, 108 dB audio alarm, aimed at apartments, factories, industrial parks and schools; the page names a fixed IR bullet camera, does not describe how the lane is defined, and says nothing about lane width or whether a fire truck can pass; no price — [Dahua](https://www.dahuasecurity.com/solutions/smb-solutions/fire-lane-parking-detection)
- RU search for a neural network that detects cars blocking fire lanes in courtyards found no ready system; closest: Moscow Oblast AI scanning hourly "Safe Region" camera snapshots for parking in prohibited roadside zones since Aug 2025, a car over 30 minutes goes to the transport ministry; over 900 systematic cases found and 526 vehicles evacuated; no vendor named — [ComNews, 2 Mar 2026](https://www.comnews.ru/content/244034/2026-03-02/2026-w10/1011/neyroset-vyyavlyaet-fakty-parkovki-zapreschennykh-zonakh-vdol-avtodorog-podmoskovya)
- SpotVision (Belarus): courtyard camera app showing free/occupied spaces, zones on Google Maps; a public dataset gave about 30% accuracy because images were at human height, 2,500 custom images brought it to about 95%; no fire-lane feature — [dev.by](https://devby.io/news/SpotVision)
- GitHub: no fire-lane or fire-access computer-vision repositories (`gh search repos "fire lane"` returns unrelated items; topic `parking-violation` has 0 repos) — [topic](https://github.com/topics/parking-violation)
- MapAnything: metric 3D reconstruction from images (demo accepts video), outputs a metric scaling factor, Apache-2.0 code and `facebook/map-anything-apache` weights for commercial use, up to 1000 views in profiling — [MapAnything](https://github.com/facebookresearch/map-anything). VGGT-1B-Commercial is gated; original VGGT is non-commercial (teammate) — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)
- MapAnything error on a metric task: 0.463 m MAE native and 0.334 m with camera-height calibration in the UrbanVGGT sidewalk-width comparison — [arXiv PDF](https://arxiv.org/pdf/2603.22531)
- Tashkent has about 193 vehicles per 1,000 residents — [Times of Central Asia](https://timesca.com/tashkent-choked-by-smog-what-2025-revealed-and-how-residents-are-responding/). Spot's fire tag page shows no headline about fire crews blocked by parked cars — [Spot](https://www.spot.uz/ru/tag/пожар/)
- The emergency-access "3.5 m default" is the idea's own number and was not verified against ShNQ.

### Inferences
- [CALC] A 0.33-0.46 m error is 9-13% of a 3.5 m clear width; at pinch points near 3.0-4.0 m the verdict is uncertain, so the output should be a band ("likely blocked / likely clear / measure on site") and the capture should include a reference object of known size (A4 sheet, car length) to cut scale error.
- [INFERENCE] The idea's "scaled by known car lengths" works only where a known model is in frame (Cobalt, Damas are common but length varies by trim); a printed reference card is more reliable and demo-friendly.
- [INFERENCE] "AI finds a car in the fire lane" is already a shipped feature (Dahua), so the defensible novelty is the phone walkthrough plus swept-path passability on unmarked routes, before an incident; the pitch must state the Dahua difference up front (fixed camera and marked zone and dwell alarm vs. one-pass geometric audit).
- [INFERENCE] Payer: $20-40 per courtyard is small; the credible buyers are the fire service and city programmes, the insurer link is untested (KASKO book fell 47% in 2025, teammate). The pitch should lead on impact (25% weight), not revenue.

### Gaps
- Chinese-language, Kazakh and Uzbek-language probes for fire-lane analytics not run (Dahua found via English); fire-truck dimensions and ShNQ width rule not sourced; the Shymkent rescuer audits are the idea author's claim, not re-checked.
- No Uzbek incident data on fire crews delayed by parked cars.
- MoGe-2 / VGGT-Commercial metric accuracy on courtyard video not benchmarked.

---

## wow-12 Rain-Ready: pre-storm drain clog dispatch list

### Takeaway
Municipal phone-on-windshield products (Xweather RoadAI, Vialytics) already inventory drains and manholes, storm-drain detection has a published paper, a public dataset and hobby repos, and a Roboflow set even has partial-block classes, but I found no product that ranks inlets by measured clog severity against a rain forecast. The Tashkent flood evidence (10 May 2026, 23 addresses) was not re-verified, theme fit with "automotive or road safety" is indirect, and the free Open-Meteo API is non-commercial only. Verdict sharpen. Scores: novelty 3, feasibility 4, demo 4, business 2, moat 2, local fit 3.

### Cited Findings
- Xweather RoadAI (Vaisala): standard smartphone mounted on any vehicle's windshield; lists storm drains, maintenance holes and drainage structures; debris and flooded areas are "special objects" recorded through driver annotations; named customers City of Montreal, Lauderdale County (TN), City of Milton Keynes; validators UK DfT (PAS 2161), Danish Road Directorate, Cerema; no prices — [Xweather RoadAI](https://xweather.com/roadai)
- Vialytics: photographs the road every 10 ft with GPS, detects cracking, patches, potholes, signs, manholes, drains, markings; 1,000+ communities in 7 countries; results in a dashboard within hours — [Vialytics](https://www.vialytics.com/)
- Research and datasets (search 8): a 2020 Sensors paper evaluated RetinaNet for storm-drain and manhole detection in street-level images, 297 images (166 manholes, 142 storm drains) from Campo Grande, Brazil, AP50 about 84-89% for storm drains (92% overall on the easier split), dataset public; it does not address blockage — [PMC7472039](https://pmc.ncbi.nlm.nih.gov/articles/PMC7472039/). A 2024 YOLOv5s-based "YOLO-SDD" storm-drain status detector for street view (S) — [Smart City Gnosys](https://smartcity.efri.uniri.hr/article.php?id=62119). A Roboflow Universe dataset `storm_drain_od` whose class names include grid_open and grid_block_30/60/90 (S, class names only; page 403, licence and size unknown) — [Roboflow](https://universe.roboflow.com/storm-drain/storm_drain_od-oql60). An early-stage "Drain Sentinel" competition entry for blocked-drain detection in Lagos (forum post June 2025, no data or accuracy) — [Axelera community](https://community.axelera.ai/project-challenge-recognize-react-27/drain-blockage-detection-flood-response-system-for-lagos-nigeria-226)
- Open-source: `Arn099xx/StormDrainBlockageDetector` (deep-learning storm-drain blockage detection from images, real plus synthetic data, Richmond BC; 0 stars) — [GitHub](https://github.com/Arn099xx/StormDrainBlockageDetector); `sfbrigade/adopt-a-drain` (citizens adopt drains, 45 stars, not CV) — [GitHub](https://github.com/sfbrigade/adopt-a-drain); GitHub topic `storm-drain` has 0 repos — [topic](https://github.com/topics/storm-drain); urban-waterlogging benchmark LSM-Adapter (ECCV 2024, 28 stars) is flood-scene detection — [GitHub](https://github.com/zhang-chenxu/LSM-Adapter)
- Open-Meteo: "The free API is for non-commercial use"; commercial use needs a paid plan (Standard 1M calls/month); free tier 300,000 calls/month — [Open-Meteo pricing](https://open-meteo.com/en/pricing). MET Norway forecast API is open data under CC BY 4.0 with attribution, identifying User-Agent, caching and rate-limit rules, no SLA — [MET Norway terms](https://api.met.no/doc/TermsOfService)
- No street-level flood or storm-drain datasets verified (teammate) — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)
- Modal T4 $0.59/h — [Modal pricing](https://modal.com/pricing)
- A "ливень" tag page on Spot returned 404; the 10 May 2026 flood claim (23 addresses, 8 districts, 20 mm/day capacity) is the idea author's and was not located.

### Inferences
- [INFERENCE] The accurate novelty claim is "clog severity per grate plus forecast-ranked crew list", not "drain detection". RoadAI's owner (Vaisala/Xweather) also sells forecasts, so the combination is a near feature-add for them.
- [CALC] Cost: 1 km at 20 km/h is 3 min = 2,700 frames at 15 fps; RF-DETR Nano at 2.3 ms is about 6 s of T4, about $0.001; SAM 3 on grate crops only; so "$0.02 per km" is plausible.
- [INFERENCE] Tashkent drainage is partly open aryks and roadside ditches (idea's own hook); the number of grates is unknown. If most flooding is capacity-driven, clog ranking has limited effect; the team should back-test the ranker on the May 2026 flood addresses.
- [INFERENCE] Theme: drains are city infrastructure; the road-safety link (flooded underpasses, hydroplaning) must be argued, and judges may see it as off-track for an automotive/road-safety track.
- Licensing: the forecast leg must use a commercial Open-Meteo plan or MET Norway data in the cost model.

### Gaps
- One storm-drain search ran (search 8); four other variants were refused; Chinese/Indian municipal drain-AI products and Roboflow dataset contents not checked.
- No local grate counts, flood frequency, or drainage-authority budget; 10 May 2026 event not located.

---

## insurer-14 PlugSafe: home EV charging fire-risk check from photos

### Takeaway
Installer-side tools (Qmerit's panel-photo AI form) exist, but one successful search found no AI or photo-based check of home EV charging setups; what exists is consistent safety guidance that extension cords must not be used (automaker manuals, UL engineers). The main problems are that a photo cannot see earthing, RCD behaviour or wire gauge inside walls, so a "pass" over-claims; the money is tiny ($1 per pass); and "pass" liability falls on the issuer. It is the most phone-testable demo in the batch. Verdict sharpen. Scores: novelty 4, feasibility 3, demo 4, business 2, moat 2, local fit 3.

### Cited Findings
- Search 9 ("home EV charger safety inspection photo AI extension cord fire risk") returned only safety guidance and nothing on AI or photo-based checks: automakers warn against extension cords (Kia's owner manual prohibits them; a Chevrolet Bolt manual says "Do not use extension cords, multi-outlet power strips, splitters, grounding adaptors, surge protectors, or similar devices"); the article cites overheating, melted plugs and receptacles, and more risk when unattended overnight, with no statistics — [KBB](https://www.kbb.com/car-advice/no-extension-cord-ev-charging); a Design News piece quoting UL engineers says an extension cord's current capacity falls with length (S, page 403) — [Design News](https://www.designnews.com/automotive-engineering/no-extension-cords-best-home-ev-charging-practices); a fire district advises checking cables for cuts, cracks or exposed wires before each charge (S) — [Marysville Fire District](https://www.marysvillefiredistrict.org/electric-vehicle-ev-charging-safety)
- Qmerit (Schneider Electric): installer network (15,000+ electricians), "AI-powered project form" with panel-photo upload for upgrade assessment; no safety check beyond permitting and inspection — [Qmerit](https://www.qmerit.com/)
- Tashkent regulator concern about chargers connected without load calculations risking residential grid overload (March 2026) — [Spot](https://www.spot.uz/ru/2026/03/06/chargers-review/)
- EV fleet 122.7k, Tashkent city 88,496 — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4); 2025 EV sales about 40,700 (+35%, CERR via Daryo, S) — [Daryo](https://daryo.uz/en/nqDKEOIvg); EV loans 12% (domestic) and 16% (foreign), planned residential limits of 22 kW in apartment buildings and 7 kW in houses (teammate) — [Spot, 10 Jan 2026](https://www.spot.uz/ru/2026/01/10/ev-plans/); [Spot, 3 May 2025](https://www.spot.uz/ru/2025/05/03/ev-charging-stations/)
- Components (teammate): D-FINE-N (Apache-2.0, 15 MB, browser build `onnx-community/dfine_n_coco-ONNX`), PP-OCRv6, Qwen3.5 small VLMs, Claude Haiku 5.5 about $0.0003 per photo — [D-FINE](https://github.com/Peterande/D-FINE); [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)
- GitHub searches for extension-cord / cable-hazard / socket detection returned nothing before the API rate limit (so no open-source clone seen).

### Inferences
- [INFERENCE] Detection of cords, adapters and charger bricks is easy; scorch marks need data; reading "3x2.5 mm2" off a black cable jacket (embossed or small print, curved) is the least reliable claim and is unverified; breaker ratings on a clean panel photo are readable.
- [INFERENCE] The things that decide home-charging safety in old housing stock (protective earth, RCD, wire material and gauge behind the wall, socket contact quality) are not photographable; a "pass" would be a false assurance. Output should be "visible risks found / not assessable", never a pass.
- [CALC] Pool: 40.7k EV sales x $1 = about $41k a year at 100% attach; the banks (MKBANK is a hackathon partner) and insurers have no stated requirement for it.
- [INFERENCE] The sellable version is a dealer-handover qualifier that turns the panel and socket photos into a wallbox + RCD + cable quote, so the dealer pays from the wallbox sale margin.

### Gaps
- Home-charging fire statistics for Uzbekistan not found; EV insurer practice unverified; only one product search ran (search 9, nothing found) and one more was refused; Uzbek electrical standards for EV sockets not checked.

---

## insurer-16 PackLedger: underbody impact ledger for EV battery warranty

### Takeaway
Drive-over underbody scanners with VIN history already exist (Vumo, UVeye), but a phone-grade, VIN-linked change ledger for EV battery trays was not found; a search for phone-based pack-damage detection returned only OBD battery-health apps. The make-or-break is capture on a dark, dirty underbody and the "scrape age from appearance" claim, which no source supports; baseline-versus-now diffing is the defensible core. Verdict sharpen. Scores: novelty 3, feasibility 2, demo 3, business 2, moat 3, local fit 4. (Business is 2, not higher, because no evidence of battery-warranty dispute frequency, BYD exclusions or dealer willingness to pay was found.)

### Cited Findings
- Vumo UB Scanner: drive-over system, patent-pending, captures hundreds of frames per second and stitches the underbody; detects corrosion and oil leaks, estimates EV battery condition, frame-damage pre-qualification; report, VIN/plate recognition and a database with historical VIN search; customers auctions and dealerships; sales-quote pricing — [Vumo](https://vumo.ai/ub-scanner/)
- UVeye: scans exterior, undercarriage and tyres, detects leaks, rust and underbody damage; 1,000+ systems; GM, Volvo, Toyota, Amazon, Hertz, JLR; landing page mentions no EV-battery module or VIN history; no price — [UVeye](https://uveye.com/); $191M raised Jan 2025, $380.5M total (teammate) — [Pulse 2.0](https://pulse2.com/uveye-ai-based-vehicle-inspection-technology-company-raises-191-million/amp/)
- Search for phone-based EV underbody/pack damage detection returned only OBD battery-health apps (BatScan, TUV Nord battery certificate, evDash, Ev Analyzer); none detects physical pack-housing damage from a phone — [BatScan, IVA list](https://www.iva.se/en/what-iva-does/distinctions/iva-list-2026/batscan-battery-health-reports-for-used-evs/)
- Three hobby repos for "vehicle underbody inspection" exist with 0 stars — `gh search repos "underbody inspection"`; teammate note on a pit-camera underbody idea reaches the same conclusion that per-VIN change detection is more honest than absolute defect detection — [verify_05.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/verify_05.md)
- BYD Jizzakh built 20.2k EVs in 2025 (5x), Yuan Up local sales 368 (2024) to 6,800 (2025), 2026 plan 40,000; the article does not mention a service network or battery warranty — [Spot, 17 Mar 2026](https://www.spot.uz/en/2026/03/17/byd-plans/); BYD dealer headlines (BYD Sergeli expansion for sales and service, 21 Mar 2026; BYD Mirobod; BYD Astana Motors Fergana) — [Spot BYD tag](https://www.spot.uz/ru/tag/byd/)
- Components (teammate): anomalib v2.7.0 (Apache-2.0; PatchCore, RAD; trains on normal-only images), SAM 3, RF-DETR-Seg — [anomalib](https://github.com/open-edge-platform/anomalib); [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)
- No BYD warranty terms for Uzbekistan were found, so the claim that external battery-tray impact voids warranty is the idea's assumption.

### Inferences
- [INFERENCE] A baseline from the same car at delivery makes anomaly detection a registration-and-diff problem; mud, water and lighting changes between visits create most false positives, so human confirmation of flagged cells is required.
- [INFERENCE] "Fresh versus old" from brightness or oxidation cues needs dated pairs the team cannot collect in 2-3 days; the dated-visit diff gives "new since last visit" without appearance-based ageing.
- [CALC] Pool: roughly 20k locally built plus about 38k imported EVs a half-year (teammate) suggests tens of thousands of new EVs a year; at $2 per check and two or three checks per car the pool is hundreds of thousands of dollars at full adoption, but dispute frequency and dealer willingness are unknown.
- [INFERENCE] The demo ("film any dark underside with a torch") is the riskiest in the batch; pre-recorded clips plus 5-10 real cars would be needed.

### Gaps
- Warranty exclusions and dispute rates unknown; no Uzbek insurer EV claim data; phone-on-pole capture quality untested; no search for Chinese EV underbody inspection products.

---

## service-19 Hanzi-Fix: Chinese EV part label to local part

### Takeaway
Photo translation and visual product lookup are commodity (Google Lens translates 100+ languages), industrial part identification from photo or label exists (Partium), and a teammate already found Chinese VIN decoders and judged a nameplate decoder weak. The only distinctive asset is an importer stock graph the team cannot build in 2-3 days, at $15 per workshop-month. Verdict kill. Scores: novelty 2, feasibility 4, demo 4, business 2, moat 2, local fit 4.

### Cited Findings
- Google Lens: "instantly translate menus, signs, handwriting, or anything else in over 100 languages" and identify products by camera; parts not mentioned; shopping results not available in China — [Google Lens](https://search.google/ways-to-search/lens/)
- Partium: identifies parts "from a photo, a label or a few words", photo/scan/OCR inputs, supplier comparison and alternatives; industrial maintenance/MRO customers (Deutsche Bahn, Home Depot, Walmart); automotive/EV not mentioned — [Partium](https://partium.io/)
- Teammate: free Chinese VIN decoders exist; Chinese-VIN decoding ("Chinese Vehicle History Report") and the BYD Atto 3 pack variants make nameplate-to-spec a data problem, not a vision problem — [verify_02.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/verify_02.md); [Product Hunt](https://www.producthunt.com/p/chinese-vehicle-history-report/chinese-vehicle-history-report)
- GitHub topic `chinese-ocr`: 51 repos including `shibing624/imgocr` (small ONNX model from PaddleOCR v5); PP-OCRv6 has English and Chinese tags (teammate) — [GitHub topic](https://github.com/topics/chinese-ocr)
- EV imports H1 2026: 37,874 units, $447.7M, 2.6x H1 2025; China 37,560 (99.2%); the article says nothing about repair, service or parts — [Spot, 14 Aug 2026](https://www.spot.uz/ru/2026/08/14/ev-import-rise/)
- Official BYD dealers and service are expanding (BYD Sergeli, BYD Mirobod, BYD Astana Motors Fergana) — [Spot BYD tag](https://www.spot.uz/ru/tag/byd/); OLX.uz carries about 67k spare-parts listings (live counter, teammate) — [OLX](https://www.olx.uz/transport/)
- Qwen3 includes Northern Uzbek among 119 languages (relevant for Uzbek output) — [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)

### Inferences
- [INFERENCE] A judge photographing a Chinese label gets what their phone already gives; the extra value (part-number regex, voltage class, HV warning, local stock) is a lookup table and a partnership, not computer vision.
- [CALC] Price: $15 per workshop-month would need hundreds of paying shops for a few thousand dollars a month; parts importers at $0.10 per lead depend on conversion data the team will not have.
- [INFERENCE] Salvage only as one screen inside a broader EV-workshop assistant (dashboard-fault reader plus HV warning plus part cross-reference), and only if a team member already has shop access.

### Gaps
- Product searches (3 variants) refused by quota; Taobao/1688 image search not checked; no data on independent EV workshop counts or on how they source parts; no BYD part-number format source.

---

## evcng-2 PumpGuard: cabin-empty interlock for methane pumps

### Takeaway
Local relevance is high (explosions, 464 suspended stations, new Cabinet measures), but the premise, a "passengers must be out of the car" rule, was not found in any Uzbek or Russian source (one RU search, four page reads), the documented failure causes are cylinder age/overfill/wrong fuel rather than cabin occupancy, and relay integration with certified dispensers is a hard regulatory and engineering step. Smoking and phone-call detection at petrol-station refuelling zones is already sold as alert-only by Dahua (thermal + AI, early-warning alarm) and Thundercomm, and vision-triggered machine stops exist in industrial EHS (Intenseye), so only the cabin-occupancy rule and the pump hold are new. Verdict sharpen toward the cylinder-registry check the state is actually building. Scores: novelty 3, feasibility 3, demo 3, business 2, moat 2, local fit 4.

### Cited Findings
- RU search for a rule that passengers must leave the car during methane filling in Uzbekistan found no source; results were about restriction schedules. Page reads: Spot "bus-cng" (Dec 2025) has no safety rule on passengers; the 2022 Syrdarya report does not say who was inside the bus and only has a reader comment asking whether passengers are made to get off; Russian Wikipedia's CNG-station article has no such rule; the 28 Jun 2026 Cabinet-measures article says nothing about passengers, smoking, phones or cameras — [Spot bus-cng](https://www.spot.uz/ru/2025/12/25/bus-cng/); [Podrobno 2022](https://podrobno.uz/cat/proisshestviya/v-avtobuse-v-syrdare-vzorvalsya-gazovyy-ballon-shest-chelovek-pogibli-15-postradali-/); [RU Wikipedia](https://ru.wikipedia.org/wiki/Автомобильная_газонаполнительная_компрессорная_станция); [Spot, 28 Jun 2026](https://www.spot.uz/ru/2026/06/28/gas-station-safety/)
- Cabinet measures of 18 Jun 2026: Interior and Digital ministries must link Eco-Himoya vehicle gas-cylinder inspection data to U-Gas so station operators can verify cylinder status; U-Gas mandatory (1,459 of 1,553 registered facilities fully connected); from 1 Sep equipment with excess pressure, expired life or unknown manufacture date restricted; 6,196 facilities inspected in 2025 with 46,737 violations; 464 stations temporarily suspended — [Spot, 28 Jun 2026](https://www.spot.uz/ru/2026/06/28/gas-station-safety/)
- UGaz (Uzinfocom): automatically checks arriving vehicles for valid technical inspection, warns the operator if the cylinder permit is missing, flags incompatible fuel (methane vs propane), monitors dispensers and the staff serving them; no cameras mentioned — [Spot, 24 Sep 2025](https://www.spot.uz/ru/2025/09/24/ugaz/)
- Incidents: 18 Aug 2022 Syrdarya bus cylinder exploded during methane filling, 6 killed and 15 injured — [Podrobno](https://podrobno.uz/cat/proisshestviya/v-avtobuse-v-syrdare-vzorvalsya-gazovyy-ballon-shest-chelovek-pogibli-15-postradali-/); 18 Sep 2025 Fergana propane cylinder filled with methane by mistake, no injuries; 5 Sep 2026 Andijan Spark cylinder exploded at a CNG station, no injuries (teammate) — [Podrobno](https://podrobno.uz/cat/proisshestviya/oshibka-voditelya-privela-k-vzryvu-na-avtozapravke-v-fergane/); [Podrobno](https://podrobno.uz/cat/proisshestviya/na-agnks-v-andizhane-vzorvalsya-gazovyy-ballon-avtomobilya-spark/)
- Prior art (search 10, "petrol station camera AI detects smoking customers phone use Hikvision Dahua forecourt"): Dahua gas-station solution uses thermal cameras plus AI to "detect and identify high risk behaviors such as smoking or making phone calls" in the refuelling area with an early-warning alarm, ANPR with a vehicle blocklist, thermal monitoring in the EV charging zone and certified explosion-proof cameras for the unloading area; the page does not mention dispenser or relay linkage; its Jul 2023 leaflet lists ordinary thermal and bullet cameras for the refuelling zone — [Dahua](https://www.dahuasecurity.com/in/solutions/enterprise-solutions/energy/gas-station); [Dahua leaflet](https://material.dahuasecurity.com/uploads/soft/20230817/Leaflet_Dahua-SMB-Gas-Station-solution_V1.0_EN_202307(2P).pdf). Thundercomm "Smart Petrol Station": edge box analytics for smoking, phone calls, smoke/flame, uniforms, extinguisher placement, unloading workflow; claims no on-site supervisor needed; no dispenser linkage or price stated — [Thundercomm](https://www.thundercomm.com/solution/smart-petrol-station/). Hikvision forecourt alarms (alert only) are a teammate snippet; my fetches of two Hikvision gas-station URLs returned empty — [Hikvision](https://www.hikvision.com/fr/solutions/solutions-by-scenario/gas-stations). Intenseye (industrial vision EHS) markets a "Software Hard Stop" that stops machinery on a high-risk event (0.8 s at a building-materials plant) and PLC/DI-DO interlocks; fuel retail is not among its listed sectors — [Intenseye](https://www.intenseye.com/)
- Axis explosion-protected cameras are certified for Zone 1/Division 1 and Zone 2/Division 2 (NEC, ATEX, IECEx); the page gives no prices and does not mention petrol stations — [Axis](https://www.axis.com/products/explosion-protected-cameras). Dahua's reference design puts ordinary thermal/bullet cameras in the refuelling zone and reserves explosion-proof units for the unloading area (see above), so Ex-rated cameras at the dispenser are not shown to be mandatory by vendor material.
- GitHub topic `smoking-detection` has 9 repos (none about fuel stations) — [GitHub topic](https://github.com/topics/smoking-detection)
- Edge hardware (teammate): Jetson Orin Nano Super 67 INT8 TOPS, 7-25 W, no USD price on NVIDIA's page — [NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)

### Inferences
- [INFERENCE] The passenger-exit rule is plausible (common CNG practice) but unverified for Uzbekistan; if no rule text exists, the cabin-empty interlock has no regulatory basis and the SOP must come from a station operator.
- [INFERENCE] Through-windscreen occupant detection fails on tinted or small Damas/Labo windows, glare and rear seats; the fallback "exit-event counting" cannot know how many people were inside.
- [INFERENCE] The documented explosion drivers (old cylinders, overfilling beyond 80%, wrong fuel; teammate) and the new Cabinet measure point to a cylinder-plate check. A pivot: OCR of the cylinder passport/stamp and vehicle plate checked against U-Gas/Eco-Himoya, holding the pump on expired or unknown test date, with an evidence packet.
- [INFERENCE] Business: $21 per pump-month (250k UZS) must cover Jetson and cameras; if cameras must be Ex-rated for the dispenser zone the BOM can exceed the fee (Ex-rated price not sourced); stations may also be unable to alter certified dispensers without approval.
- [INFERENCE] Phone-at-pump alerts rest on a precaution rather than documented incidents; do not lead with them.

### Gaps
- Hikvision forecourt analytics unverified (pages unreadable); only one gas-station AI search ran (7 other variants refused); whether Dahua/Thundercomm deployments exist in Uzbekistan unknown; no CNG attendant-versus-self-service data; station counts only via proxies; relay/authorisation protocols of Uzbek CNG dispenser brands unknown.
