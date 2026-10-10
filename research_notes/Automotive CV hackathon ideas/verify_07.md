# Adversarial due diligence, batch 07: ten fleet-logistics / road-safety / CNG ideas (Tashkent CV hackathon, track "Computer Vision")

Compiled 2026-10-10. Ideas checked: fleet-5 LoadGate, fleet-8 Plomba Twin, fleet-10 StackSafe, fleet-11 ScaleGuard, fleet-12 CubeProof, fleet-15 TruckProof, safety-8 RideCoach, safety-13 DarkRoad, safety-14 RiderAudit, evcng-1 Navbat.

Method and limits. Untagged = I fetched and read the page. **(S)** = taken from a search-result summary or a page that returned 403/empty/redirect, lower confidence. The session's shared WebSearch quota (200 calls, shared by all agents) ran out repeatedly, so about 12 planned queries were refused (listed under Gaps of each idea) and I filled the holes with direct page fetches of primary sources. UZS conversions use about 11,800 UZS per USD (rate from the teammate fact sheet). Teammate files used as context: saturation_map.md and uzbekistan_market.md in this folder.

---

## fleet-5 LoadGate: pre-departure cargo securing and overhang check

### Takeaway
The securing-check half already exists as a Fraunhofer IML pilot (phone photo in, correct/incorrect out, log to TMS), so the real novelty is only the metric overhang/envelope measurement. Payer evidence in Uzbekistan is thin; the most credible wedge is a construction-site exit gate under the November 2025 smog decree rather than a $10/truck/month haulier subscription.

### Cited Findings
- Fraunhofer IML "CargoSecuring Assist": smartphone/tablet (or fixed-camera) photo of the cargo area, AI identifies load beams, straps and form-fitting, returns an immediate correct/incorrect signal and an automated load-securing log into the TMS; offered as a pilot, no pricing or accuracy published, no mention of overhang, dimensions or 3D. Page cites 70% of inspected vehicles with defects and 16,000 accidents a year from inadequate load securing (scope not stated). — [Fraunhofer IML](https://www.iml.fraunhofer.de/en/fields_of_activity/material-flow-systems/software_engineering/checking-load-securing-by-photo-cargosecuring-assist.html)
- Osnabrück prototype: photos of loaded trucks classified safe / unsafe / unusable by a two-stage ANN; abstract cites 11,371 police truck checks in 2020 with 9.6% (1,091) load-safety violations (country not stated on the abstract page; accuracy and dataset size not on the abstract). — [arXiv 2306.03795](https://arxiv.org/abs/2306.03795)
- Roboflow sells a bundle-strap inspection template (counts straps per bundle vs spec, checks spacing) for steel, pipe and lumber, not general truckloads (S). — [Roboflow](https://roboflow.com/ai/bundle-strap-inspection)
- Nothing found that measures cargo overhang or envelope in metres from phone video. Russian query found only a BNTU Excel/VBA lashing calculator and a sensor-based monitoring paper, no photo AI. — [BNTU repository](https://rep.bntu.by/items/5b5f7084-fec6-4a30-bea5-13a8439c0380) (S)
- Smog decree UP-229 (24 Nov 2025), government summary: "Construction sites are required to prevent dust emissions, carry out watering, and secure exit areas" and "control over the transportation of construction materials is being tightened". The page does not state tarp or wheel-wash rules or penalties. — [gov.uz](https://gov.uz/en/miit/news/view/105885)
- Model facts: SAM 3 has 848M parameters, checkpoints are gated on Hugging Face, text-prompt open-vocabulary segmentation, SAM 3.1 released 27 Mar 2026 — [GitHub](https://github.com/facebookresearch/sam3); SAM License grants a non-exclusive royalty-free licence, bars ITAR/military end uses, does not explicitly address hosted SaaS — [LICENSE](https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE). MapAnything default weights are CC-BY-NC 4.0; `facebook/map-anything-apache` is Apache 2.0 and recommended for commercial use; intrinsics optional; outputs a metric scaling factor — [GitHub](https://github.com/facebookresearch/map-anything).
- Price check: $10/vehicle/month = about 118,000 UZS = about 1.7% of the 7.09M UZS average monthly wage [CALC from teammate sheet]. Individually owned trucks number 330.3k (1 Apr 2026), mostly small owner-operators (teammate sheet, uzbekistan_market.md).
- Queries run: EN extended "truck load securing AI smartphone photo inspection overhang strap detection"; RU "проверка крепления груза грузовик нейросеть фото приложение".

### Inferences
- Novelty is 3, not higher: the same workflow (phone photo, rulebook, TMS log) is a named Fraunhofer pilot; only the metric overhang/height/width envelope is open.
- The 3D pipeline (SAM 3 + MapAnything + LLM rulebook) is overbuilt for the demo; a side+rear photo pair plus known truck dimensions or a printed marker gives overhang and height with simple geometry.
- Strap sufficiency (EN 12195-1 lashing capacity) cannot be judged from a photo, only strap presence/placement.

### Gaps
- No Uzbek rule text for overhang/width/height or oversize permits retrieved (query on ПДД Uzbekistan overhang was refused by the exhausted quota).
- No Uzbek data on load-securing fines, cargo-claim frequency or insurer appetite.
- No evidence that the smog decree requires load covering or wheel washing.

---

## fleet-8 Plomba Twin: same-seal proof for cargo

### Takeaway
Seal-number reading and seal-photo evidence are commodity (AllRead, CargoSnap, Vector); generic-object "fingerprinting" from a phone photo is a commercial technology (Alitheon) and a seal-specific phone method is patented (Authentic Vision), so the idea is a re-application, not a blank space. Regulated transit is moving to state GPS e-seals (EAEU), which shrinks the addressable gap to domestic/fuel/non-EAEU legs.

### Cited Findings
- Vector photographs the seal at pickup and records seal integrity at each checkpoint; "blind seal verification" has the checker read and enter the seal number without seeing the stored one. The page does not claim AI matching of seal photos; no pricing or funding. — [Vector](https://www.withvector.com/solutions/fraud-prevention/)
- Alitheon FeaturePrint identifies one specific object from a single photo using natural surface micro-features, "a mobile phone or an ordinary industrial camera is sufficient", vendor claims greater than 99.9% accuracy and zero false positives, API for ERP/supply chain; use cases listed are aerospace parts, electronics, pharma, luxury, OEM parts; seals/cargo not specifically addressed; no pricing. — [Alitheon FAQ](https://www.alitheon.com/faq)
- Authentic Vision patent US 10891726 "Seal tamper detection" (priority 2015-12-11, granted 2021-01-12): smartphone image of a seal's designed breaking point is compared with a stored reference; each seal carries an identifier (serial number) used to look up the reference; integrity state can be logged to detect replicated seals; main use case is spirits-bottle refill protection. — [Google Patents](https://patents.google.com/patent/US10891726)
- CargoSnap: operators scan or photograph the seal, number validated against the expected value, photo, timestamp, operator and location recorded, "mismatches or broken seals trigger exceptions"; no origin-vs-destination image comparison described. — [CargoSnap](https://www.cargosnap.com/resources/blog/best-practice-for-seal-verification-at-unloading)
- AllRead (Barcelona): mobile-first seal reading after fixed cameras proved too costly; 97% seal detection and 92% code reading in a controlled fixed-camera project; port/terminal customers; no tamper detection stated. — [AllRead](https://www.allread.ai/?p=29276)
- Santos Brasil tablet OCR seal app (2019): read time 7 s to 1.43 s, accuracy 83%. — [Datamar](https://datamarnews.com/noticias/santos-brasil-launches-application-for-reading-container-seals)
- TamperLens (open source, 1 star): DINOv2 embeddings plus ORB/RANSAC/SSIM heatmap for packaging and seal tamper, no benchmarks. — [GitHub](https://github.com/hasnaintanoli/TamperLens)
- EAEU navigation seals: reusable GLONASS/GPS devices that also monitor cargo-compartment integrity, one seal for the whole route; new control procedure in force 11 Feb 2026, stage-by-stage. — [economist.kg](https://economist.kg/transport/2026/02/25/tsifrovaia-plomba-protiv-sieroi-loghistiki-kak-novyie-pravila-tranzita-izmieniat-rynok-ghruzopierievozok-ieaes/)
- More than 70,000 navigation-seal uses since Feb 2026; from 31 Jul 2026 use expanded to all road transport and more rail categories; EEC and Uzbekistan Customs Committee plan data-exchange treaty talks for Sep-Oct 2026; Uzbekistan-EAEU trade up 25% to over $20B in 2025. — [TKS.ru, 3 Aug 2026](https://www.tks.ru/news/nearby/2026/08/03/0009/eaes-rasshiryaet-tamozhennoe-partnerstvo-s-uzbekistanom-cherez-obmen-dannyimi-i-tranzit/)
- Uzbek customs item (28 Nov 2024) concerns an electronic system replacing paper certificates for vehicles carrying goods under customs seals; it does not mention electronic seals. — [Anhor](https://anhor.uz/news/jelektronnaja-sistema/)
- Model facts: RF-DETR Nano detection 2.3 ms on T4 TensorRT FP16, Apache-2.0 — [RF-DETR](https://github.com/roboflow/rf-detr); XFeat Apache-2.0, real-time sparse matching on CPU for VGA, but README warns its homography demo suits planar scenes — [XFeat](https://github.com/verlab/accelerated_features).
- Queries run: EN extended "cargo seal verification photo AI tamper detection smartphone same seal"; RU "проверка пломбы по фото нейросеть подмена пломбы"; EN "container seal number OCR photo app AI seal verification gate logistics startup"; RU "электронные пломбы транзит Узбекистан таможня навигационная пломба внедрение".

### Inferences
- Novelty 3: the exact "same physical generic seal, phone-only, ledger across hops" product was not found, but its two halves are sold (object fingerprinting, seal photo evidence) and one seal-specific phone method is patented.
- Honest weak point: a lookalike seal that copies the serial can still match on glyph edges, so discrimination must come from micro-texture or crush marks, which need macro-distance, controlled-light photos; pick-and-reseal of the identical object is not detectable by identity matching.
- Cheap to test: 40 seals x 12 photos gives a real ROC in a day.

### Gaps
- No Uzbek data on seal-swap theft or tanker short-delivery losses; no evidence that fuel-flow sensors would not already cover tanker fraud (not checked).
- Uzbek electronic-seal rules not found (EN query on this was refused by the quota; the RU query succeeded only for EAEU neighbours).
- Alitheon's real-world accuracy on cheap moulded plastic or metal seals is a vendor claim only.

---

## fleet-10 StackSafe: rollover and crosswind score for tall loads

### Takeaway
The only close product (Road-Aware) needs the CoG category typed in by hand, so photo-derived CoG is genuinely unclaimed, but the output reduces to a static-stability formula on a measured height, validation data does not exist, and I found no payer or local incident evidence. Best treated as a feature of a bigger pre-departure product, not a standalone.

### Cited Findings
- Road-Aware Safety Systems app (2019): computes maximum safe curve speed from the truck's rollover threshold and safe descent speed; fleets enter tractor type, trailer configuration, load weight and a high/medium/low centre-of-gravity category; 10-tractor trial over eight road segments; vendor cites a mine-site rollover where the calculated threshold was 29-33 mph and GPS showed 34 mph; no measured rollover reduction reported; pricing not given. — [TruckingInfo](https://www.truckinginfo.com/335010/app-helps-drivers-through-curvy-descending-slopes)
- Patents estimate CoG, lateral rollover limit or cargo weight from vehicle sensors, not photos: US 8483942 (S) — [FreePatentsOnline](https://www.freepatentsonline.com/8483942.html); US 11772946 (S) — [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11772946). Search summary also notes one trucking article gives a rule of thumb (CoG height vs half track width) and that no photo-based load-height tool turned up (S). — [Truck News](https://www.trucknews.com/health-safety/the-science-behind-truck-rollovers-and-how-to-prevent-them/1003200255/)
- Depth Anything 3 licences: DA3METRIC-LARGE (0.35B) and DA3MONO-LARGE Apache-2.0; DA3-BASE/SMALL any-view Apache-2.0; DA3-LARGE, GIANT and Nested variants CC BY-NC 4.0 (so a multi-view "orbit" with the strong models is non-commercial). — [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- MoGe-2 ViT-L weights are MIT (HF card), code MIT; README claims 60 ms per image on A100/RTX3090 FP16 ViT-L. — [HF](https://huggingface.co/Ruicheng/moge-2-vitl); [GitHub](https://github.com/microsoft/MoGe)
- Local evidence: a Russian news query for truck rollovers with cotton/hay/wind in Uzbekistan returned only unrelated rollovers (Samarkand-Bukhara km 108, Kamchik pass) with causes not linked to load height or wind. — [Podrobno](https://podrobno.uz/cat/proisshestviya/na-108-m-kilometre-dorogi-samarkand-bukhara-proizoshlo-krupnoe-dtp-perevernulsya-gruzovik-/)
- Price check: $8/truck/month = about 94,400 UZS = about 1.3% of average monthly wage [CALC].
- Queries run: EN extended "load center of gravity estimation truck rollover risk loaded cargo height photo smartphone app"; RU "перевернулся грузовик хлопок сено ветер Узбекистан ДТП перегруз высота груза".

### Inferences
- Novelty 4 (nothing photo-based found) but the physics is the textbook static stability factor (track width / 2 x CoG height); the CV only supplies height, width and length. Dynamic rollover thresholds are lower than static ones, so any "safe speed" needs a conservative margin and a disclaimer.
- No payer evidence: Road-Aware shows a small US bulk-hauler market for rollover advisories, not an Uzbek one.

### Gaps
- No Uzbek rollover, load-height or crosswind-closure statistics found.
- No data on hay/cotton stack density variability or on a tilt-table calibration.
- No Russian/Uzbek-language market query for rollover advisory products succeeded beyond the one above.

---

## fleet-11 ScaleGuard: AI referee for weighbridge cheating

### Takeaway
The feature set (wheels on platform edge, extra person on the deck, abnormal stops, plate-plus-face swap check, weight-curve anomaly) is already advertised by Chinese unmanned-weighbridge vendors, and lifted-axle detection has a published CNN; the idea is a regional gap and a retrofit story, not a new category. Real cheat footage is the hard blocker for a 2-3 day build.

### Cited Findings
- Weitewosi/Vitworth "AI anti-cheating unmanned weighbridge": camera/AI vision flags wheels touching the platform edge (partial loading) with a voice prompt and recording, a driver getting off or interfering, abnormal stops; plate fraud via plate plus facial recognition and reservation matching; weighing-curve AI flags "jumping the scale" (claims 99.7% accuracy for abnormal waveforms); says it works with "hundreds of enterprises"; no price; lifted axles not mentioned. Vendor claims, unverified. — [wtwsscales.com](https://www.wtwsscales.com/info/the-ai-anti-cheating-and-system-103440717.html)
- Cerema (SETO project): YOLOv8s lifted-truck-axle detector from side-on cameras, 87% precision, 91.7% recall, 1.4 ms inference; notes weigh-in-motion struggles with lifted axles; does not address weighbridge fraud; dataset size not stated on the abstract page. — [arXiv 2506.11574](https://arxiv.org/abs/2506.11574)
- Matrix Comsec (India) weighbridge security: ANPR, multi-angle video of driver, cabin and cargo bed linked to scale tickets, barrier control; page does not claim position-on-platform or person-on-scale analytics; all efficacy numbers unsourced. — [Matrix Comsec](https://www.matrixcomsec.com/weighbridge-security-camera-system/)
- Hener Scale lists the five classic frauds (straddling one axle off the deck, bridging two scales, tare manipulation, electronic tampering, operator collusion) and recommends photo-eye position sensors that block the reading until the truck is fully on the deck, encrypted digital load cells and ANPR with barriers and an audit trail; it mentions no AI. — [Hener Scale](https://henerscale.com/the-silent-profit-killer-5-common-weighbridge-frauds-and-how-to-stop-them/)
- A second Chinese vendor page describes only retail-scale fraud and digital-injection detection, not weighbridge vision (so the "AI vision" claim rests on one vendor page). — [fatbodyscale.com](https://www.fatbodyscale.com/share/industrial-weighing-weight-capacity-platform-scale-digital-scale-printing-scale-weighing-solutions-weighing-technology-belt-scale-floor-scale-truck-scale-hanging-scale-crane-scale-electronic-scale-label-printing-scale-body-scale-kitchen-scale.html)
- RF-DETR-Seg Nano 3.4 ms on T4 (Apache-2.0) — [RF-DETR](https://github.com/roboflow/rf-detr). The Hailo model-zoo master branch targets Hailo-10/15 only; I could not confirm any DETR-family model on Hailo-8/8L, so the "RF-DETR on Pi 5 + Hailo at 20 fps" claim is unverified. — [Hailo model zoo](https://github.com/hailo-ai/hailo_model_zoo)
- Russian/CIS query found no vision-based weighbridge anti-fraud vendor; results were Mettler Toledo's fraud guide (hardware/sensor countermeasures) and an ASU-TP vendor with plate matching (S). — [Mettler Toledo](https://www.mt.com/kz/ru/home/supportive_content/guides/po/tandl/fraudprevention.html) (S); [Business Gazeta](https://www.business-gazeta.ru/article/517356) (S)
- Queries run: EN extended "weighbridge fraud detection camera AI lifted axle truck scale cheating"; RU "автомобильные весы мошенничество видеоаналитика нейросеть взвешивание".

### Inferences
- Novelty 2: an almost identical feature list is on sale in China; it would be new only as a Central Asian retrofit.
- Arithmetic [CALC with assumed inputs]: at 100 trucks/day x 25 t and an assumed $5-10/t, each 0.1% leakage is about $12-25/day, so the $80/month fee breaks even at roughly 0.015% leakage; payback is trivial if any real leakage exists, but proving it needs real footage.
- Cheaper barriers, light curtains and ANPR already stop the most common cheats (straddling, bridging); collusion with the operator, the hardest case, needs process change.

### Gaps
- No Uzbek weighbridge fraud cases, vendor landscape or price points found.
- Hailo-8/8L compatibility of RF-DETR unverified.
- Cerema dataset size and training details unreadable (PDF returned binary).

---

## fleet-12 CubeProof: bulk-load volume receipts for tippers

### Takeaway
A phone-photo bulk-load estimator is already marketed (Oxmaint, claims unverified) next to LOADSCAN's 700+ laser installs and phone-based Timbeter for logs, so "phone-only" is not unclaimed; the physical blocker is that a ground-level phone cannot see the heap inside a high-sided tipper, which is why every industrial and even the Oxmaint workflow uses a top-down or elevated view.

### Cited Findings
- Oxmaint "AI Vision Camera" (page links App Store/Google Play, API): driver takes a guided sequence at the loading site (top-down bed view, side profile, material ID), 60-90 s, returns weight and per-axle distribution, bridge-formula check; claims ±3-5% weight and 2-4% volume without methodology; sand, gravel, asphalt, soil, grain covered; no price, no named customers. — [Oxmaint](https://oxmaint.com/industries/fleet-management/ai-load-weight-estimation-computer-vision)
- LOADSCAN laser Load Volume Scanners: over 700 units installed in over 50 countries, trade approvals in NZ, Australia, USA, quarries/sandpits/construction; cloud portal, no phone app. — [Loadscan](https://www.loadscan.com/)
- Timbeter: phone-photo timber/log measurement incl. truck loads, 50 countries, first 10 measurements free, no accuracy figure, bulk materials not mentioned. — [Timbeter](https://timbeter.com/)
- red_mad_robot / Stoilensky GOK (NLMK) pilot: three fixed cameras plus a depth camera per wagon/truck point; pile peak height within about 10 cm; 72,000-lumen floodlights for night; trains under-delivered by 2-3% of the accounting norm; projected savings about 70M RUB/yr; not yet a deployed commercial product (2021). — [Habr](https://habr.com/ru/companies/redmadrobot/articles/571504/)
- Habr 2022 log-truck pipeline (YOLOv5 + OpenCV) uses a licence plate of known size as the scale ruler; no accuracy metrics. — [Habr](https://habr.com/ru/articles/661649/)
- Academic single-camera result: relative error 0.169 for construction-waste truck payload volume (S, ScienceDirect page returned 403). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0921344921006212); a 2026 Springer multi-view paper needs at least four views (S, paywalled redirect). — [Springer](https://link.springer.com/article/10.1007/s41693-026-00179-z)
- Smog decree: "control over the transportation of construction materials is being tightened"; no tarp requirement stated. — [gov.uz](https://gov.uz/en/miit/news/view/105885)
- Price check: $0.10/load = about 1,180 UZS.
- Queries run: EN extended "smartphone truck load volume estimation photo bulk material dump truck AI cubic metres"; RU "определение объема груза в кузове самосвала по фото нейросеть кубометры"; EN "smartphone app estimate truckload volume ... product" (refused, quota).

### Inferences
- Novelty 3: the nearest product is a small vendor with unverified claims; the novelty is the buyer-side receipt framing and CIS focus.
- Peer-reviewed single-camera error is about 17% (one data point), so a ±10% claim is credible only after testing against scale tickets with an elevated capture position.
- A buyer-side receipt only works if the buyer can mandate it at the gate; sellers will not accept a buyer's app as arbiter.

### Gaps
- No Uzbek evidence of short-load disputes in sand/gravel trade (RU query refused by quota).
- No independent accuracy data for Oxmaint.
- Springer multi-view paper abstract unreadable.

---

## fleet-15 TruckProof: challenge-video carrier verification

### Takeaway
Carrier-identity and fraud-screening tools (Overhaul, KYT, FleetGen, Highway) are established in the US, and ATI.SU already operates counterparty ratings across Russia/Kazakhstan/Uzbekistan; no product combining plate, tech passport, vehicle type, driver face and a live code in one challenge video was found. Technically the most buildable idea in the batch, and Uzbekistan's MyID makes the identity layer an integration rather than a build.

### Cited Findings
- Overhaul FraudWatch: on-site carrier verification with photo verification of vehicle and driver cross-checked against a risk profile; US data sources; no video, plate reading, liveness or challenge code on the page; "$100M in losses avoided" without methodology. — [Overhaul](https://over-haul.com/platform/intelligence/fraudwatch/)
- KYT (Know Your Trucker): broker sends a link, carrier photographs the driver's licence and takes a selfie, face liveness blocks static/pre-recorded images, "under 20 seconds"; no vehicle/plate check; sponsored article, Feb 2024. — [Freight Caviar](https://www.freightcaviar.com/stories/put-a-face-behind-every-shipment-tendered)
- FleetGen (ex Fenderr Freight): biometric driver verification plus authority/insurance/crash checks; method details, pricing and geography not given. — [FleetGen](https://www.fleetgen.ai/)
- Highway verifies carriers with licence plus live photo and is adding ML to detect AI-generated photos; FBI Dec 2024 PSA on generative-AI impersonation; ATA reports roughly 1,500% increase in strategic cargo theft since 2021; double brokering ranked top fraud concern in a Freight Caviar poll. — [FreightWaves](https://www.freightwaves.com/news/generative-ai-supercharging-fraud)
- ATI.SU (CIS freight exchange): 452,700 participants, loads from Russia, Kazakhstan, Uzbekistan, Kyrgyzstan, Armenia, UZ/KZ/RU language options, links to participant ratings, a counterparty-check guide and "Светофор+"; the page itself describes no document/driver verification programme. — [ATI.SU](https://ati.su/)
- MyID (Uzbekistan national face-biometric ID): liveness and face recognition, claims iBeta testing to ISO 30107-3, SDKs (mobile, web, server, cloud), more than 19 million users, more than 280 million authorizations, 31 commercial banks and 45 mobile operators connected; pricing not published. — [MyID](https://myid.uz/)
- fast-plate-ocr: MIT licence, global pretrained models, 0.32-0.68 ms per plate on RTX 3090, Uzbek formats not listed. — [GitHub](https://github.com/ankandrew/fast-plate-ocr)
- InsightFace pretrained models (buffalo_l etc.) are non-commercial research only; separate licensing needed; code is MIT. — [GitHub](https://github.com/deepinsight/insightface)
- Central Asia plates dataset (500 images, KZ/KG/UZ/TJ/TM, CC BY 4.0) — [Mendeley](https://data.mendeley.com/datasets/4c5z833m6m/2) (from teammate saturation map).
- Price check: $0.50 per verification = about 5,900 UZS.
- Queries run: EN extended "carrier identity verification video freight broker fraud AI vehicle plate driver face liveness"; RU/UZ-specific queries on Uzbek freight fraud and Telegram groups were refused by the quota.

### Inferences
- Novelty 3: each component exists; the triangulation in one challenge video with a CIS focus is the new part.
- Biggest hole: a thief driving a real truck with real documents passes a pure identity/vehicle check; the product must verify the owner/lessee-to-carrier link, not only person-document-vehicle consistency.
- MediaPipe gives landmarks for an active challenge (blink, turn) but is not a presentation-attack detector; use MyID or a commercial PAD and avoid InsightFace weights in a commercial claim.

### Gaps
- No Uzbek cargo-fraud statistics, no confirmation of how Telegram freight groups verify carriers today.
- Whether Uzbek vehicle registry data can be queried by plate for ground truth not checked.
- ATI.SU's actual verification features are not described on its landing page.

---

## safety-8 RideCoach: handlebar-phone coaching for scooter couriers

### Takeaway
Per-ride safety ratings for scooters exist (Superpedestrian, Lime, Voi, Drover) but on operator-owned hardware, and Chinese platforms now hand riders camera helmets; a courier-owned phone scoring wrong-way, sidewalk and red-light behaviour was not found. The detection targets do not match what Uzbek enforcement actually records, and the plan has no positive labels.

### Cited Findings
- Superpedestrian Pedestrian Defense (July 2021): detects sidewalk riding, wrong-way riding on one-way streets, swerving, hard braking; per-ride Safety Rating drives in-app/SMS training, ride discounts and lockouts; no outcome figures published. — [IoT Insider](https://www.iotinsider.com/transport/superpedestrian-unlocks-e-scooter-safety-breakthrough/)
- Drover AI PathPilot: sidewalk/bike-lane detection and speed control integrated into scooters, partners Voi, Spin, Bolt, Beam, Dott, Blue Systems, Felyx; no rider-phone or courier offering. — [Drover](https://drover.ai/)
- Lime Vision (on-board cameras, audible alert, slows the scooter), Segway Pilot Lite (AI box slows the vehicle, logs events) and a Voi UK computer-vision trial do sidewalk detection on operator fleets (all S, search summaries only). — [Traffic Technology Today](https://www.traffictechnologytoday.com/news/multimodal-systems/ai-enabled-computer-vision-for-e-scooters-unveiled-by-lime.html); [ZAG](https://zagdaily.com/tech/the-segway-pilot-lite-bolstering-e-scooter-safety-with-ai/); [TNW](https://thenextweb.com/news/the-ai-police-are-here-to-bust-you-for-bad-escooter-behavior)
- China 2026: JD's food-delivery unit launched a helmet with AI voice assistant and road-facing camera; Meituan's second-generation helmet promised up to 100,000 free units to riders in Suzhou and Beijing; Ele.me has its own; the article does not say they detect violations. — [TNW](https://thenextweb.com/news/china-ai-smart-helmets-delivery-riders-jd-meituan-price-war)
- Uzbekistan Jan-Sep 2026 (SBDD): 563 moped/scooter crashes, 89 killed, 488 injured; 4,312 SBDD-recorded violations vs about 300 in all of 2025. Yandex Uzbekistan: 23,785 two-wheeler violations in 2026, about 4.7 bn UZS fines, only 8% (1,890) missing documents, 92% reflectors/lights/tint/passengers; mopeds 3.8% of 1,286 fatal crashes. The article gives no breakdown for helmets, wrong-way, sidewalk or red-light, names no platform responsibility rule, no courier headcount and no platform quote. — [Spot, 6 Oct 2026](https://spot.uz/ru/2026/10/06/couriers)
- RF-DETR Nano 2.3 ms on T4 TensorRT (detection), Apache-2.0 — [RF-DETR](https://github.com/roboflow/rf-detr).
- Price check: 3,000 UZS per courier per month = about $0.25.
- Queries run: EN extended "delivery courier scooter smartphone handlebar AI safety score wrong-way sidewalk riding detection"; EN "Meituan Ele.me rider safety AI camera phone detect wrong-way riding delivery riders".

### Inferences
- Novelty 3: each behaviour detector exists somewhere; the courier-owned-phone-plus-bonus loop is not found but platforms could ship it themselves (they already put cameras on riders' heads in China).
- Business risk: the recorded violation mix is dominated by equipment/passenger offences, not the three behaviours the CV targets; platform demand is asserted, not evidenced.
- Honesty risk: with "no unsafe riding staged" there is no positive-event ground truth; a canned laptop clip demonstrates overlays, not accuracy.

### Gaps
- No Uzbek platform (Yandex Eats, Uzum Tezkor, Express24) statement on courier safety scoring; no courier headcount.
- No evidence on phone thermal/battery limits for continuous camera inference while the courier app runs.
- No Uzbek helmet rule or fine found (teammate sheet).

---

## safety-13 DarkRoad: unlit slow road-user alert for night drives

### Takeaway
Night animal detection exists as research and thermal prototypes, and large fleet dashcams (Netradyne etc.) market night video, but nothing found alerts on carts, unlit tractors or livestock with a safe-speed advisory. Night phone video is the weakest regime for open-vocabulary auto-labelling, and I found no Uzbek night-crash data to size the problem.

### Cited Findings
- YOLO-based night animal detection on highways: precision 0.923, recall 0.773 in normal night conditions; highways only, no carts (S). — [ResearchGate](https://www.researchgate.net/publication/389333841_Enhancing_Road_Safety_Detection_of_Animals_on_Highways_During_Night)
- Colorado high-school thermal AI dashcam prototype for deer (S). — [Outdoor Life](https://www.outdoorlife.com/conservation/ai-thermal-wildlife-vehicle-collisions/)
- Netradyne: Driveri night dashcam "uses active illumination"; page quotes NHTSA that night is about 25% of driving but 49% of passenger-vehicle fatalities, and FMCSA that 35% of fatal large-truck crashes happen at night (US figures, as quoted by the vendor); no mention of animals, carts or unlit objects. — [Netradyne](https://www.netradyne.com/blog/the-importance-of-having-an-ai-powered-dash-cam-for-night-driving)
- South African dashcam analysis of a night crash into a cow: 87 km/h at impact, driver apparently noticed the cow about 3 s before impact (timestamps internally inconsistent), no headlight-range figures. — [Arrive Alive](https://www.arrivealive.mobi/collision-and-dash-cam-analysis-of-crash-into-cow-at-night)
- Uzbekistan: Q1 2026 crash causes were speeding 32.9%, ignoring signals 22.4%, not yielding to pedestrians 13.6%, falling asleep 4.2%; no night, cart, tractor or livestock share was found (teammate sheet, uzbekistan_market.md, citing Kun.uz 13 Apr 2026).
- RF-DETR Small 3.5 ms on T4, Apache-2.0 — [RF-DETR](https://github.com/roboflow/rf-detr).
- Price check: 60,000 UZS per vehicle per month = about $5.
- Queries run: EN extended "dashcam night unlit hazard detection livestock animals carts rural roads AI warning"; RU "ДТП ночью трактор без света гужевая повозка животные на дороге Узбекистан" (refused by quota).

### Inferences
- Novelty 4 because the specific class set and the speed-advisory framing were not found, but it is a re-training exercise on a known detection problem.
- Physical limit [my engineering judgment, unsourced]: low-beam illumination of dark objects is shorter than stopping distance at highway speeds, so alerts help mainly at lower speeds or with IR/thermal sensing; claims of "metres to impact" from monocular depth at night are unvalidated.
- Daytime and dusk detection of the same hazards on the same routes avoids the night-labelling bottleneck.

### Gaps
- No Uzbek night-crash share, no cart/tractor/livestock crash data.
- No evidence fleets would pay for rural hazard maps.
- Quantitative ADAS night-pedestrian performance (AAA-type tests) could not be retrieved.

---

## safety-14 RiderAudit: passenger-phone safety scores for staff shuttles

### Takeaway
The rider-sourced audit concept is new, but the core signals (driver phone use and seat belt) are not observable from a passenger seat and a 10-second clip at boarding cannot see driving behaviour, so the CV does not carry the product. Passenger rating forms and telematics already cover most of the tender value.

### Cited Findings
- NUS Internal Shuttle Bus: ordinary passengers score the Bus Captain through an e-form ("Mystery Rider"), and telematics safety scores improved 33% in 2020-21 (S, page body returned empty on two fetches). — [NUS](https://uci.nus.edu.sg/improving-customer-service-on-the-isb)
- UKM (Malaysia) prototype collects passenger perceptions of express-bus drivers in real time to let operators monitor drivers; research report, not deployed (S). — [UKM](https://ftsm.ukm.my/v5/file/research/technicalreport/LP-CAIT-FTSM-2021-042.pdf)
- In-cab DMS from Streamax and Cipia/Harman requires installed hardware (teammate saturation map, 2B).
- InsightFace pretrained face models are non-commercial research only, relevant if face embeddings are used. — [GitHub](https://github.com/deepinsight/insightface)
- Price check: 100,000 UZS per bus per month = about $8.5.
- Queries run: EN standard "passenger-reported bus driver behaviour app phone camera safety score contractor staff shuttle".

### Inferences
- Novelty 4 for the exact mechanism, but a plain rating form (NUS) already captures passenger judgement without CV.
- Feasibility is low for physical reasons: headrests hide hands and phone, mirror reflections are small and glare-prone, and boarding happens while stationary.
- Filming a driver raises personal-data issues; I did not verify Uzbek personal-data law specifics.

### Gaps
- No Uzbek staff-shuttle fleet sizes, contract terms or tender practice found.
- No data on the cost of cheaper non-CV monitoring in Uzbekistan (GPS speed, ratings).
- NUS programme details unverified (page empty).

---

## evcng-1 Navbat: live CNG queue and open/closed radar

### Takeaway
No driver-facing CNG station status or queue product was found in Uzbekistan: the state UGaz system watches stations but exposes nothing to drivers, and closure news travels via the Energy Ministry Telegram channel and Facebook groups. The problem is acute and structural (3 restriction waves in winter 2025-26, new station permits not issued), but the 3-second stall-detection design cannot work and winter queue data cannot be filmed in October.

### Cited Findings
- UGaz: online accounting and control system for CNG stations by Uzinfocom, announced by the Ministry of Energy (article of 24 Sep 2025); real-time monitoring, checks arriving vehicles for valid technical inspection and cylinder-installation permits, flags incompatible fuel; integrated with Tax Committee and Interior Ministry data; the article mentions no public app, station status or queue feature. — [Spot](https://www.spot.uz/ru/2025/09/24/ugaz/)
- Winter 2025-26: nationwide limits from 11 Dec 2025 (6M m3/day deficit); 8 Jan 2026 stations 10:00-16:00; on 25 Dec only 11 of 33 CNG stations serving Tashkent public transport remained open; the ministry announced changes on its Telegram channel and users shared closures in Facebook groups; no queue lengths in the article. — [Spot, 8 Jan 2026](https://www.spot.uz/ru/2026/01/08/gas-stations/)
- Restrictions reimposed 10 Mar 2026 (cold, humid air mass, low pipeline pressure), normal operation from 17 Mar (Tashkent) and 19 Mar (Fergana Valley); limits imposed three times in the 2025-26 season; no queue or throughput data. — [Spot, 17 Mar](https://www.spot.uz/ru/2026/03/17/methane-return/); [Spot, 18 Mar](https://www.spot.uz/ru/2026/03/18/methane-ferghana/)
- Deputy PM Khodzhaev (Karakalpakstan meeting, "9 Feb", year not printed): permits for new methane stations are not being issued: "Even if you build a station, the invested funds may be frozen. There will be no gas for it." — [Anhor](https://anhor.uz/news/stroitelstvo/)
- Queue evidence is old: "150 cars" at a Kokand station on 22 December and "kilometre-long queues" are from an Anhor article dated 24 December 2018, not the 2025-26 season. — [Anhor, 2018](https://anhor.uz/news/poholodaniesozdalo-ogromnie-ocheredi-nagazovih-avtozapravkah/)
- Spot's CNG tag page shows 2025-26 headlines on UGaz and operating hours but none on a status app, map, bot or online queue. — [Spot tag](https://www.spot.uz/ru/tag/%D0%B0%D0%B3%D0%BD%D0%BA%D1%81)
- Generic queue analytics: open-source YOLOv8 + DeepSORT queue monitor estimating wait as queue length x average service time (people, not vehicles) — [GitHub](https://github.com/Piyush-Sambyall/Wait-Time-Detection); GPS-based gas-station queue patent CN103198552A — [Google Patents](https://patents.google.com/patent/CN103198552A/en). Hikvision forecourt page could not be read (empty fetch).
- Model facts: RF-DETR-Seg Nano 3.4 ms on T4, Apache-2.0 — [RF-DETR](https://github.com/roboflow/rf-detr); MoGe-2 weights MIT, 60 ms ViT-L on A100 — [HF](https://huggingface.co/Ruicheng/moge-2-vitl).
- Market sizing inputs from the teammate sheet: petrol-gas is 33.9% of fuel-typed Avtoelon listings (not a fleet share); 576k registered self-employed taxi drivers, 361k via aggregators (KSE thesis); average wage 7.09M UZS (uzbekistan_market.md).
- Price check: 15,000 UZS per car per month = about $1.27.
- Queries run: EN extended "gas station queue detection computer vision app wait time drivers CNG fuel queue"; RU "очередь на метановой заправке приложение Узбекистан работает ли заправка онлайн"; RU "метан заправки Ташкент очередь телеграм канал мобильное приложение статус АГНКС" (no app found). Refused by quota: Yandex Maps/CNG-status query, Uzbek CNG-finder app query.

### Inferences
- Novelty 4 (nothing found), but Google/Yandex Maps live-busyness or a state open/closed feed could erode it; not checked.
- Design flaw [my estimate]: CNG queues advance in steps every few minutes (a fill takes minutes), so tracked-box displacement over 3 seconds cannot separate a moving queue from a stalled one; throughput must come from successive time-stamped counts or longer clips, or from a "hose connected at pump" cue.
- Data flaw: it is October, queues are short; winter-crisis footage must come from archived news video.
- Counting cars x about 6 m pitch is likely as accurate as metric-depth lane extent and far simpler.

### Gaps
- Could not check Google/Yandex Maps or Telegram channels for existing station-status services.
- No queue-length or waiting-time data for 2025-26 found anywhere.
- No count of CNG stations (teammate sheet has none).

---

## Cross-cutting model/licence facts used above
- RF-DETR: Nano det 2.3 ms, Small 3.5 ms, Seg Nano 3.4 ms on T4 TensorRT FP16; Nano/Small/Medium/Large and all Seg sizes Apache-2.0, Atto/Femto/Pico/XL/2XL under PML 1.0 — [GitHub](https://github.com/roboflow/rf-detr)
- MapAnything: use `facebook/map-anything-apache` for commercial claims — [GitHub](https://github.com/facebookresearch/map-anything)
- Depth Anything 3: only Base, Small, Metric-Large and Mono-Large are Apache-2.0 — [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- SAM 3: gated checkpoints, SAM License — [GitHub](https://github.com/facebookresearch/sam3)
- XFeat: Apache-2.0 — [GitHub](https://github.com/verlab/accelerated_features)
