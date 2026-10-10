# Verification batch 13: ev-cng-energy and cross ideas (evcng-3, evcng-4, evcng-6, evcng-7, evcng-14, evcng-15, matrix-3, matrix-5, matrix-6, usedcar-7)

> Research date 2026-10-10 (finals Day 1). IMPORTANT LIMITATION: the shared WebSearch quota was exhausted almost at once. Of 19 search attempts only 5 returned results (marked "ran"); the other 14 were refused (marked "refused"). Everything else was verified by direct WebFetch of specific URLs (primary pages, laws, vendor pages, sibling notes in this folder). Novelty scores therefore rest on 5 live searches + fetched vendor pages + the teammate saturation map + my background knowledge (flagged). Absence of a competitor here is NOT proof it does not exist. Flags: (S) = search-result summary only, page not opened or 403; (sibling) = taken from another verifier's note in this folder; (CALC) = my arithmetic; (assumption) = unsourced input. FX: about 11,800 UZS per USD (teammate sheet).

## Score table (novelty / feasibility / demo / business / moat / local fit -> verdict)
| id | N | Fe | D | B | M | L | famous | verdict |
|---|---|---|---|---|---|---|---|---|
| evcng-3 BallonID | 4 | 2 | 3 | 2 | 2 | 5 | no | sharpen |
| evcng-4 BayWatch | 2 | 4 | 3 | 2 | 2 | 4 | yes (crowded: Cleverciti, SpotGenius/Zevtron, MAV BayiQ, Intelli-Park, Charge Saint) | kill |
| evcng-6 ChargeMD | 3 | 3 | 4 | 2 | 2 | 3 | no | sharpen |
| evcng-7 Tarozi | 3 | 2 | 2 | 1 | 1 | 3 | no | kill |
| evcng-14 Sim | 4 | 2 | 2 | 1 | 2 | 3 | no | kill |
| evcng-15 Joylash | 3 | 4 | 3 | 2 | 1 | 3 | no | sharpen |
| matrix-3 Plug-Ready | 3 | 3 | 4 | 3 | 2 | 4 | no | sharpen |
| matrix-5 Tranche Eye | 3 | 4 | 3 | 1 | 1 | 3 | no | kill |
| matrix-6 Stall Whisperer | 3 | 2 | 2 | 2 | 2 | 3 | no | kill |
| usedcar-7 PledgePulse | 3 | 3 | 4 | 3 | 3 | 4 | no | keep |

Cross-cutting findings
- CNG side: the state already owns the station-side channel. UGaz (UZINFOCOM, exclusive rights under Presidential Decree PF-35 of 28 Feb 2025) does real-time accounting, plate-based checks of technical inspection, cylinder-installation permit and fuel-type compatibility, and says its goal is to prevent illegal sale. Resolution 315 (18 Jun 2026) adds cylinder-inspection data integration by 1 Sep 2026. Any station-side CV product must integrate with, or sell around, that monopoly.
- EV side: the documented Tashkent charger "problems" (101 disconnected, "103 of 500 meet technical conditions") are paperwork and grid-connection compliance, NOT device faults. Three ideas (ChargeMD, Stall Whisperer, Tranche Eye) lean on those facts as if they were reliability evidence. They are not. No legal fine for ICE-ing exists yet (Tokbor is still lobbying).
- Market size in USD is small for every EV idea (1,399 stations on 1 May 2025). Judging weights (hackathon_intel.md): Innovation 25, Technical Complexity 25, Impact and Practicality 25, Presentation 15, Business 10; so small-TAM ideas can still place if innovation and technical depth are shown honestly.

---

## evcng-3: BallonID (flashlight-sweep reader for gas-cylinder stamps)

### Takeaway
Best local fit and most distinctive in the batch, but (a) the state is solving cylinder status by database (UGaz + Eco-Himoya) through an exclusive integrator, (b) photometric reconstruction from a handheld torch sweep on curved, rusty, painted shoulders is research-grade, (c) the web torch constraint is probably Android-only, (d) revenue per scan is tiny. Sharpen to a robust capture + rules tool sold to inspectors, not stations.

### Cited Findings
- Cabinet Resolution No. 315 of 18 Jun 2026: licensee must not fill vehicles that lack permit documents, have not passed the next technical inspection, or have unauthorised cylinders, and must not fill cylinders past service life or not passed the next test (Annex 6 s.2(b) sub-clause "z", amending licence Annex 25 item 9); MVD with the Digital Ministry must integrate the Committee's cylinder-inspection data with Eco-Himoya and U-Gas by 1 Sep 2026 (point 4(v)); the text does NOT say how a station verifies cylinder status, sets no stamp/marking requirement and no inspection interval for vehicle cylinders, and names no fine (U-Gas data = grounds for administrative measures, supply suspension for uncorrected station violations) — [Lex.uz](https://lex.uz/ru/docs/8280849)
- Spot 28 Jun 2026: MVD and Mintsifry "must integrate Eco-Himoya data ... station operators will be able to check the condition of gas-cylinder equipment"; no deadline in the article; the 1 Sep restriction is on station equipment (excess-pressure, expired life, unknown manufacture date) and the article does not name car cylinders; 464 stations suspended; stamps/markings not mentioned — [Spot](https://www.spot.uz/ru/2026/06/28/gas-station-safety/)
- UGaz (UZINFOCOM, 11 Sep 2025): operator workstation, licence-plate recognition that checks gas-equipment registration and technical-inspection status, accounting/online reporting, video surveillance, real-time recording at each station, equipment lent to stations, tariff 25 UZS per m3 from 24 Sep 2025, goals include curbing theft and unauthorised dispensing, exclusive rights per PF-35 of 28 Feb 2025 — [UZINFOCOM](https://uzinfocom.uz/news/378)
- Spot 24 Sep 2025: UGaz warns the operator on expired technical inspection, missing cylinder-installation permit and incompatible fuel (methane vs propane); integrated with Tax Committee, MVD and Energokontrol; cylinder serial numbers and test dates are not mentioned — [Spot](https://www.spot.uz/ru/2025/09/24/ugaz/)
- Metal cylinders carry stamp markings on the shoulder (manufacturer ID, standard, serial number, manufacture date, charging pressure, capacity, test-agency mark, date of each re-validation test); each retest date is stamped — [Wikipedia Gas cylinder](https://en.wikipedia.org/wiki/Gas_cylinder)
- Prior art on reading stamps: Chinese patent CN112950237A (publication 2021-06-11, Changzhou market-supervision service centre) reads the steel stamp with OCR equipment as the unique cylinder ID for blockchain traceability; no lighting or accuracy detail visible — [Patsnap](https://eureka.patsnap.com/patent-CN112950237A)
- Standard industry approach is electronic ID marking, not stamp reading: India's Gas Cylinders (Amendment) Rules 2022 make QR/RFID/barcode marking mandatory (oxygen from 20 Jul 2022; other non-flammable gases from 20 Jan 2023); Find My Cylinder reads QR only — [Find My Cylinder](https://findmycylinder.com/)
- Live search (ran): "gas cylinder stamp reading smartphone OCR CNG LPG cylinder inspection app" returned industrial OCR (Qualitas LPG case study), QR trackers, the CN patent; summary said no consumer smartphone app reads stamped test dates on CNG/LPG cylinders (S)
- Relief-reading prior art: Reflectance Transformation Imaging computes per-pixel normals from many photos with a STATIONARY camera and known/knowable light directions; page says nothing on handheld/torch variants or stamped metal — [Cultural Heritage Imaging](https://culturalheritageimaging.org/Technologies/RTI/)
- Anyline reads embossed/low-contrast tyre DOT text on-device with cloud processing; page gives no method, lighting or accuracy detail — [Anyline](https://anyline.com/products/tire-sidewall-scanning)
- Incidents: 18 Sep 2025 Fergana, a Labo's propane cylinder was filled with methane (driver "confused the fuel"), exploded, no injuries — [Podrobno](https://podrobno.uz/cat/proisshestviya/oshibka-voditelya-privela-k-vzryvu-na-avtozapravke-v-fergane/); an article cites 55 explosions, 41 killed, 23 injured and blames overfilling beyond 80% (period not stated; ban on non-SPG-1 cylinders from 1 Feb 2018) — [Podrobno 2018](https://podrobno.uz/cat/obchestvo/v-uzbekistane-vstupil-v-silu-zapret-na-staroe/)
- Models exist: PP-OCRv6 shipped in PaddleOCR 3.7.0 (2026-06-11) — [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); SAM 3 is 848M parameters, needs a CUDA GPU (CUDA 12.6+), SAM License, no speed numbers — [SAM 3](https://github.com/facebookresearch/sam3)
- MDN documents a `torch` image-track property but gives no compatibility data; I believe Chrome Android supports it and iOS Safari does not (background knowledge, unverified; sibling verify_03 reached the same view) — [MDN](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints)
- Petrol-gas is 9,965 of 29,403 fuel-typed Avtoelon car listings (33.9%, sibling live read) and used turnover was 52-57k per month in Sep-Nov 2025 (teammate sheet)

### Inferences
- The legal duty to refuse expired/untested cylinders is real, but the state's route is registry lookup by plate. Stamp reading is only needed when the record is missing, wrong, or the cylinder was swapped. That is a narrow but defensible "physical-to-registry bridge".
- CALC upper bound: ~625-690k used deals a year x 33.9% gas share x 20,000 UZS add-on = about 4.2-4.7 bn UZS (about $0.36-0.40M) if every gas-car deal bought it; realistic capture is a few percent. Station-side revenue (5k UZS per scan) is not independently sellable because UGaz is an exclusive state system.
- Honest demo is possible on a flat stamped plate; claims about real cylinders need a measured legibility rate on real shoulders (rust, paint, trunk access). "UNREADABLE" is the right default.

### Gaps
- Where the test/retest stamp sits on SPG-1 cylinders, retest interval in Uzbekistan, whether Eco-Himoya integration is live, what Eco-Himoya is. No real cylinder images found. Web searches for RU stamp-reading and photometric-stereo-on-phone were refused.

#### Queries run
ran: "gas cylinder stamp reading smartphone OCR CNG LPG cylinder inspection app". refused: "CNG cylinder hydrotest date stamp OCR app inspection"; RU "считывание маркировки газового баллона по фото нейросеть дата освидетельствования"; "smartphone flashlight photometric stereo read stamped embossed characters metal"; "handheld smartphone flash photometric stereo recover surface relief engraved text without calibration". Fetch failed: ecohimoya.uz (certificate mismatch), so what Eco-Himoya is remains unknown.

---

## evcng-4: BayWatch (LED-and-plug reader for blocked bays)

### Takeaway
Verdict kill. The "free / charging / finished-but-parked / blocked" product class is already sold by at least five vendors; the only real twist is reading the charger's own LED where no status feed exists, which is a feature, not a product. No legal fine exists yet, the nudge needs a plate-to-contact lookup that does not exist, and revenue per bay is a few dollars a month. Salvage: fold LED/plug-state reading for non-networked chargers into another EV idea.

### Cited Findings
- Cleverciti (Duesseldorf): occupancy sensing "married to the charging status reported by the EV charger itself"; alert when a vehicle sits in an EV space without charging for 10+ minutes (never started, or finished and stayed); 3 months of data gave an average 1.25 violations per charger per day (40 a month); post dated 5 Aug 2022 — [Cleverciti blog tag](https://www.cleverciti.com/en/resources/blog/tag/EV+charging)
- SpotGenius + Zevtron (3 Sep 2025): flags gas-powered cars in EV stalls and EVs that stay after completing a charge, plate validation through Zevtron's platform; the article mentions no plug or LED reading; no hardware, pricing or customers named — [Mann Report](https://www.mannpublications.com/mannreport/2025/09/03/zevtron-partners-with-spotgenius-to-monitor-ev-charging-spaces-with-vision-ai-technology/)
- Live search (ran) also surfaced MAV Systems BayiQ (ANPR camera for EV bay management, one unit covers several bays), PoGo Charge + Intelli-Park in Cwmbran (ANPR + parking charge notices for non-EVs, EVs get 90 minutes) and Charge Saint; page bodies for BayiQ and the Environment+Energy Leader piece returned 403 (S) — [Traffic Technology Today](https://www.traffictechnologytoday.com/news/machine-vision-alpr/mav-launches-bayiq-anpr-camera-for-ev-charging-bay-management.html); [Electrical Times](https://www.electricaltimes.co.uk/pogo-charge-partners-with-intelli-park-to-tackle-iceing-and-improve-ev-driver-experience)
- Charge Saint (S4EV, UK): drivers photograph blocked/broken chargers; evidence is timestamped, geolocated, "AI-enhanced" verified, sent to the site owner who can send a warden or issue a penalty notice by post; "eliminates hardware costs"; UK only — [Terrapinn exhibitor page](https://terrapinn-cdn.com/exhibition/solar-storage-live/exhibitor-charge-saint.stm)
- Uzbekistan: on 7 Jul 2025 a Tokbor representative proposed fining ICE cars in EV bays (Code of Administrative Liability amendment); decree PKM-732 (1 Nov 2024) asked MVD and hokimiyats for proposals within two months and Tokbor says it was not carried out; the Chamber head called the proposal appropriate — [Spot](https://www.spot.uz/ru/2025/07/08/ev-stations/)
- 1,399 charging stations on 1 May 2025 (+840 YoY), 820 (58.6%) in Tashkent, 92k users — [Spot](https://www.spot.uz/ru/2025/05/03/ev-charging-stations/)
- RF-DETR-Seg Nano: 40.3 COCO mask AP, 3.4 ms on a T4 (TensorRT FP16, 312 px), Apache-2.0 — [RF-DETR](https://github.com/roboflow/rf-detr); fast-plate-ocr is MIT, global model, README gives no country list (Uzbek formats unverified) — [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr); SAM 3 (848M params, CUDA) is too heavy for a Jetson-class box in real time.

### Inferences
- LED reading is a real but narrow wedge: it matters only for non-networked units. No evidence found on how many Tashkent chargers are non-networked.
- A Telegram nudge reaches only EV drivers already in an operator's app; ICE blockers cannot be contacted (no plate-to-owner lookup). "Fee evidence" has no legal consumer until the proposed fine exists.
- CALC: $5 per bay-month vs a Jetson Orin Nano Super (about $250, background knowledge) serving 4 bays = 12-month hardware payback before any labour; use existing CCTV/RTSP.

### Gaps
- Price per kWh at Tashkent DC chargers (needed to test the "30k UZS per 30-minute block" assumption), charger LED conventions per Chinese model, share of non-networked units.

#### Queries run
ran: "EV charging bay camera AI detects charging complete vehicle still parked idle fee ICE blocking". refused: "EV charging bay ICEing detection camera computer vision blocked charger"; "EV charging bay monitoring camera detect charging LED status plug idle vehicle overstay startup".

---

## evcng-6: ChargeMD (charger fault triage and uptime trust map)

### Takeaway
Photo reports of broken chargers exist (Charge Saint, PlugShare-style apps); backend reliability data exists (Paren). The pin-level "unsafe" verdict is the risky novelty: no real burnt-connector data, synthetic defects validate nothing, and the verdict carries liability. The error-code decoder and verified uptime feed are the defensible parts. The idea's local hook is misread.

### Cited Findings
- Charge Saint, see evcng-4 — [Terrapinn](https://terrapinn-cdn.com/exhibition/solar-storage-live/exhibitor-charge-saint.stm)
- Paren lists a "Reliability Index" and "Down Stations" metric from standardised real-time data; the page mentions no photos or computer vision; customers named: Suncoast Charging (operator), map makers/automakers, fleets, government — [Paren](https://paren.app/)
- PlugShare page body was not readable by fetch (title only); that it offers photo check-ins and failure reports is background knowledge, unverified here.
- "Only 103 of 500 Tashkent stations meet technical conditions" is about grid connection: the deputy minister says operators connecting as sub-consumers must go to regional grid companies, declare capacity; 11 stations were given time to fix paperwork; equipment faults or quality are not mentioned; the "about 20 offline" figure is one company's — [Spot 3 Jan 2025](https://www.spot.uz/ru/2025/01/03/charging-stations/)
- 101 Tashkent chargers temporarily disconnected in Mar 2026 for missing documents, 62 site owners had no electricity supply contract, the concern is grid overload, about 1,300 DC chargers above 20 kW; the article mentions no equipment faults — [Spot 6 Mar 2026](https://www.spot.uz/ru/2026/03/06/chargers-review/)
- anomalib v2.7.0 (Apache-2.0, 2026-10-06) ships MH-PatchCore and trains on normal-only images, but the MVTec AD benchmark is non-commercial (sibling tech_radar.md) — [anomalib](https://github.com/open-edge-platform/anomalib)
- FLUX.2 [klein] 4B: Apache-2.0, text-to-image plus multi-reference editing, about 13 GB VRAM, inpainting not explicitly listed — [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

### Inferences
- CALC: 100k UZS (about $8.5) per site-month x 1,399 stations = about $143k a year at 100% share; realistic 5-10% = $7-14k.
- Payer shift: EV taxi fleets (state target of 300k EV taxi drivers, a PLAN in the teammate sheet) lose money on failed charges; they are a better buyer than operators.

### Gaps
- No fault statistics for Tashkent chargers; no evidence EV drivers want a photo-triage app instead of operator apps; PlugShare features not re-verified.

#### Queries run
refused: "EV charger fault photo report AI broken charger app reliability". Fetched: Charge Saint, Paren, PlugShare (empty), Spot x2, anomalib via sibling.

---

## evcng-7: Tarozi (CCTV audit of methane sales)

### Takeaway
Kill. It duplicates the state's UGaz mandate, volume-from-dwell is physically noisy, and the demo would be a slider on a simulated ledger.

### Cited Findings
- UGaz already includes a video-surveillance subsystem, plate recognition, real-time per-station recording and online reporting, with stated goals to curb theft and unauthorised dispensing; exclusive rights to UZINFOCOM under PF-35 — [UZINFOCOM](https://uzinfocom.uz/news/378)
- Spot: UGaz accounts for gas in real time, is meant to prevent illegal sale, and "controls the work of dispensers and the personnel" — [Spot](https://www.spot.uz/ru/2025/09/24/ugaz/)
- Resolution 315 Annex 5 point 18: the integrator's monitoring system detects violations such as fills of non-inspected vehicles; supplier may suspend gas supply to a station after an uncorrected violation (Annex 5 points 19-26) — [Lex.uz](https://lex.uz/ru/docs/8280849)
- Hikvision's forecourt solution page returned an empty body (could not check for dwell/reconciliation features).

### Inferences
- A private chain has no reason to buy an audit of a system the state already runs with exclusive rights; a regulator would simply use UGaz.
- Fill volume depends on cylinder size, starting pressure and station pressure, so dwell x nominal flow is a weak volume proxy; with no real ledger the validation is circular.
- Salvageable idea (count served cars vs logged transactions) is exactly what UGaz's plate + accounting pairing can already do.

### Gaps
- Not verified whether private station chains want independent audits; feature details of FootfallCam/AllGoVision.

#### Queries run
refused: "fuel station CCTV video analytics reconcile pump sales fraud detection"; "fuel station video analytics match vehicles served to pump transactions ...".

---

## evcng-14: Sim (cable-fed EV census from night video)

### Takeaway
Kill. No prior art found for cable detection from street imagery, but only because it is physically weak: thin cables and charge-port flaps at night from a moving phone are not credible to detect at honest accuracy in 48 hours, there are privacy issues, and utilities already hold transformer-level load data.

### Cited Findings
- Live search (ran): no work found that detects a charging cable or an active session from street-level images. Closest: Swansea Street View driveway classification (87.2% and 89.3% accuracy) for home-charging potential [cronfa.swansea.ac.uk](https://cronfa.swansea.ac.uk/Record/cronfa56687); SEAI aerial-image model for home-charger feasibility [SEAI brief](https://www.seai.ie/sites/default/files/documents/research-project/RDD-00790-Policy-Brief-Prakash.pdf); A*STAR street-view roadside-charger siting [A*STAR](https://research.a-star.edu.sg/articles/highlights/street-smarts-for-roadside-chargers/); inlet detection only for close-range charging robots (S)
- 122.7k EVs registered on 1 Jul 2026; Tashkent city 88,496 (72.1%, CALC) — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4)
- Planned home-charging limits: up to 22 kW for apartment buildings and 7 kW for houses; the article presents it as a plan, not a confirmed rule — [Spot](https://www.spot.uz/ru/2025/05/03/ev-charging-stations/)
- Whether Uzbek EVs have distinguishable licence plates was not found (Spot EV tag page has no plate headline).

### Inferences
- Salvageable fragment (overnight EV-density by model badge for courtyard-charger siting) is a weaker variant of evcng-15 and loses the novelty.

### Gaps
- No EV-plate rule found; no data on how many Tashkent EV owners charge via extension cords; privacy law not checked.

#### Queries run
ran: "detect electric vehicles charging at home from street-level images charging cable computer vision residential". refused: "EV charging cable curbside detection street imagery computer vision home charging census".

---

## evcng-15: Joylash (charger siting from lot CCTV dwell)

### Takeaway
Feasible and demoable but thin: one-off studies for mall owners, no moat, and the actual bottleneck is grid connection and permits, not demand data.

### Cited Findings
- Paren sells charger utilisation, session behaviour and reliability data to operators for "location and pricing decisions" (backend data, no CCTV) — [Paren](https://paren.app/)
- Charger-siting analytics is an established category built on session, traffic and demographic data, not owner CCTV: Stable (models trained on millions of real charging sessions; GreenCore chose it for 10,000 charging plazas, 1 Mar 2022), FreeWire's site-analytics tool with a profitability calculator, HERE/Korem (footfall and time-spent factors), Driivz placement analytics; the live search found no camera or video-analytics approach to dwell for siting. Pages returned 403 (S) — [Business Wire](https://businesswire.com/news/home/20220301006296/en/GreenCore-Selects-Stables-EV-Charging-Site-Selection-Software-to-Build-10000-Charging-Plazas-Where-Drivers-Need-Them); [CSP Daily News](https://cspdailynews.com/technologyservices/freewire-launch-site-analytics-tool); [HERE Korem](https://here.com/learn/blog/korem-ev-charging-station-site-selection); [Driivz](https://driivz.com/blog/optimal-ev-charger-placement-data-driven-site-selection/)
- Cleverciti and Parquery sell any-camera occupancy and dwell analytics (teammate saturation map, not re-fetched).
- 1,399 stations at 1 May 2025 (+840 YoY) — [Spot](https://www.spot.uz/ru/2025/05/03/ev-charging-stations/); Mar 2026 crackdown on chargers connected without contracts or load calculations — [Spot](https://www.spot.uz/ru/2026/03/06/chargers-review/)
- Statutory support: 10% loans for station builders, land at half price (teammate sheet, Spot 10 Jan 2026)

### Inferences
- CALC: 3M UZS (about $254) one-off x a hundred malls/hotels = about $25k, one time. Lead-gen to charger distributors (success fee on an installed unit) is the better revenue line.
- Chinese-brand badge is a noisy EV proxy (Chery/Haval ICE models are common); the simulator must show low/base/high.

### Gaps
- No CCTV-based siting competitor confirmed absent; no distributor margin data.

#### Queries run
ran: "EV charger site selection parking lot camera video analytics dwell time utilization forecast payback".

---

## matrix-3: Plug-Ready (90-second phone survey for home EV chargers)

### Takeaway
Qmerit already has an AI panel-photo step in the US; the local twist (Uzbek panels and limits, dealer + bank flow) is real. The weak points are the $15-25 site-visit saving (assumption), cable-length accuracy from monocular depth, and that apartment-block charging is an approvals problem more than a measurement problem.

### Cited Findings
- Qmerit's project form lets customers "upload a panel photo to quickly assess whether upgrades may be needed", described as AI-powered with an electrician confirming; installs "start at $749"; the page does not state what the model reads or its accuracy — [Qmerit](https://www.qmerit.com/)
- Treehouse's site did not resolve (DNS), so it was not checked.
- MoGe-2 outputs metric-scale geometry from one image, but the README gives no metric accuracy numbers and MoGe-3 is now the default — [MoGe](https://github.com/microsoft/MoGe); Depth Anything 3 has DA3METRIC-LARGE (0.35B, Apache-2.0), no metric accuracy numbers — [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- EV market: 122.7k EVs, 72.1% Tashkent — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4); BYD plans 40,000 units in 2026 (teammate sheet, Spot 17 Mar 2026); 2025 passenger EV sales about 40.7k (Daryo via teammate sheet, S)
- Planned 22 kW (apartments) / 7 kW (houses) limits are plans — [Spot](https://www.spot.uz/ru/2025/05/03/ev-charging-stations/)
- byd.uz is a placeholder; no evidence found of what BYD Royal Motors offers for home chargers. Royal Motors (Toyota, BYD, Ducati) is a hackathon sponsor — [hackathon_intel.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/hackathon_intel.md)

### Inferences
- CALC (assumption: half of ~40k annual EV buyers need a survey): about 20k x $4 = about $80k a year ceiling. Uzbek electrician labour is cheaper than the US, so the $15-25 visit baseline is probably overstated; get one installer quote.
- Demo is strong: any phone, any socket, a bank card.

### Gaps
- Uzbek breaker/meter label conventions, installer prices, whether dealers bundle wallboxes, accuracy of cable-run estimates.

#### Queries run
refused: "AI electrical panel photo EV charger installation quote remote assessment breaker amperage reading".

---

## matrix-5: Tranche Eye (photo-verified milestones for charger-station loans)

### Takeaway
Kill. A few hundred milestones a year, generic lender-verification vendors already sell the workflow, the fraud scenario is hypothetical, and the cited local hook (101 chargers cut off) is about paperwork that photos cannot verify.

### Cited Findings
- Truepic lists business lending ("virtually confirm that the loan collateral exists and is in good condition"), equipment financing ("verifying the location and condition of equipment"), and construction/private lending where draws are funded faster after self-serve inspections; customers shown include Equifax, OnDeck, Dun and Bradstreet as logos — [Truepic](https://www.truepic.com/)
- 101 stations disconnected for missing documents, 62 owners without electricity contracts — [Spot](https://www.spot.uz/ru/2026/03/06/chargers-review/)
- Station growth +840 in the year to 1 May 2025 — [Spot](https://www.spot.uz/ru/2025/05/03/ev-charging-stations/)

### Inferences
- CALC (assumptions: 20% of new stations bank-financed, 3 milestones each, $10 each): about 500 milestones = about $5k a year. Replacing a "$50-100 site trip" is unsourced.

### Gaps
- No evidence of AC-for-DC substitution fraud; no bank interview; OpenSpace/Buildots not fetched.

#### Queries run
No dedicated search succeeded.

---

## matrix-6: Stall Whisperer (hidden charger faults from driver behaviour)

### Takeaway
Kill. Telemetry already exposes hidden faults at zero camera cost for networked units; video adds hardware and confounders; no fault labels exist; a toy-car demo proves nothing.

### Cited Findings
- Teammate reading of the Caltrans project: a usage-anomaly tool found hidden faults cutting uptime 6-38% (mean 16%) at 12 chargers, and backend monitoring misses mechanical damage and obstruction (S). My two fetches of the PDF failed (empty error, then connection refused), so these numbers are unverified by me — [Caltrans report](https://dot.ca.gov/-/media/dot-media/programs/research-innovation-system-information/documents/final-reports/task-id-4353-final-report-a11y.pdf)
- Paren already surfaces "Down Stations" from backend data — [Paren](https://paren.app/)
- OCPP is widely used by charger vendors and management platforms for centralised monitoring — [Wikipedia OCPP](https://en.wikipedia.org/wiki/Open_Charge_Point_Protocol); that OCPP status messages include Preparing/Faulted is background knowledge, not verified here.
- The "103 of 500" hook is grid-connection compliance — [Spot](https://www.spot.uz/ru/2025/01/03/charging-stations/)

### Inferences
- The only gap camera behaviour fills is non-networked units, which also lack a ticketing integration. Avoidance is confounded by stall position, habit and blocking.

### Gaps
- Caltrans details unverified; no Tashkent hub video or fault labels.

#### Queries run
No dedicated search succeeded.

---

## usedcar-7: PledgePulse (monthly video check-ins for collateral cars)

### Takeaway
Best of the batch for demo and payer clarity. Building blocks exist elsewhere (verified capture, 360-video asset verification, challenge-response capture), but I found no lender-facing monthly vehicle re-verification. Weak points: the pledge-monitoring pain is unverified in Uzbekistan and the "condition delta" is not honest from phone video.

### Cited Findings
- Truepic: verified capture in a mobile browser or app, "Location authentic", "Time and date accurate", "Device integrity"; lending use cases include collateral existence/condition and equipment location; page does not mention liveness, anti-spoof or screen-replay detection — [Truepic](https://www.truepic.com/)
- DoxAI Asset Verification (formerly Verimoto): lenders and brokers, 360-degree exterior video of an asset, geotagged photos, ownership/PPSR history, 4,000 brokers and lenders, 120,000 verifications; page does not mention periodic re-verification after disbursal, anti-spoofing or pricing — [DoxAI](https://doxai.co/product/asset-verification/business/) (found via live search, which also found vehicle-RC verification and face liveness, no vehicle check-in product)
- Ravin AI "Fleets and Leasing: monitor your assets' condition"; no lender use case, no anti-spoof description — [Ravin](https://www.ravin.ai/); Inspektlabs lists insurance, fleets, rental/leasing, remarketing, no lender use case — [Inspektlabs](https://www.inspektlabs.com/)
- Challenge-response capture already exists in insurance: Venta Capture uses a session-only spoken code and compares phone motion-sensor data with image motion; VAARHAFT SafeCam does photo-of-photo, virtual-camera and GPS-spoof checks; Truepic Vision Flex "starts at a 1,000 USD monthly minimum" (vendor-written comparison, secondhand) — [VentaVid](https://www.ventavid.com/blog/insurance-image-fraud-detection)
- Uzbek lending: auto loans to the population were 34.6 trn UZS in Jan-Nov 2023 (4.4x the same period of 2021), 19.6 trn in 2022; used passenger-car sales 1.16M in 2023 — [Anhor](https://anhor.uz/news/cerr/); active bank promotions: Ipoteka Bank BYD hybrid loans from 11.9% (5 Feb 2026), Apex Bank 0% Geely (17 Feb 2026), InfinBANK 0% Onix/Tracker (22 Oct 2025); the tag page shows nothing on collateral monitoring or GPS trackers — [Spot](https://www.spot.uz/ru/tag/автокредит/)
- A pledge registry for movable property exists; a 14 Oct 2025 headline proposes expanding pledgeable property and pledge accounts (not vehicle-specific) — [Spot pledge tag](https://www.spot.uz/ru/tag/залог/)
- Threat-model evidence is real but small: the Ministry of Economy and Finance (draft open until 5 Aug 2025) wants a separate "may take the car abroad" clause in car powers of attorney because proxies increasingly take cars abroad and sell them; in 2024 about 50 cars belonging to other people were taken out without return, 8 such cases in June 2025 alone; the article mentions no pledged or leased cars and no banks — [Spot 22 Jul 2025](https://www.spot.uz/ru/2025/07/22/cars-borders/)
- MKBANK, the Central Bank and Uzcard are hackathon partners; the roster suggests a financing hook may land (speculation in the source) — [hackathon_intel.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/hackathon_intel.md)
- Sibling verify_03: physical-liveness presented as explainable evidence, plus a published red-team table, is the defensible design; a general fake-detector claim is not.

### Inferences
- CALC: 6,000 UZS = about $0.51 per check; 12 checks a year = about $6 per loan-year versus Truepic's $12,000-a-year minimum. Per-loan cost is credible if a bank currently pays for physical pledge visits (unverified).
- About 50 cars exported under powers of attorney in 2024 is tiny against 4.9M cars, so "stopped asset loss" is a weak headline; early-warning on contact loss, usage change and missed check-ins is the stronger value.
- Wedge with strongest need: lessors who OWN the asset (rent-to-own, EV-taxi leasing programmes), because for retail loans the pledge registry and insurance already exist.
- Telegram "Photo" mode strips EXIF; use in-browser capture (sibling verify_03).

### Gaps
- No bank statement of pledge-monitoring cost or loss; no data on pledged cars sold by power of attorney; outstanding auto-loan count unknown. Live searches for vehicle collateral monitoring were mostly refused.

#### Queries run
ran: "lender asset verification video selfie with vehicle loan collateral AI India NBFC". refused: "vehicle collateral monitoring video check-in lender AI auto loan"; "auto loan collateral vehicle verification video lender remote inspection pledged car".

---

## Fetch log (successes unless noted)
lex.uz Resolution 315; Spot 28 Jun 2026; UZINFOCOM news 378 (twice); Spot 24 Sep 2025 UGaz; Podrobno Fergana; Podrobno 2018; Mann Report SpotGenius/Zevtron; SpotGenius home; Cleverciti blog tag; Charge Saint (Terrapinn); Paren; PlugShare (empty); Qmerit; Truepic; Inspektlabs; Ravin; DoxAI; VentaVid; Anyline; CHI RTI; CN112950237A; Find My Cylinder; Wikipedia Gas cylinder and OCPP; Spot x5 (7 Jul 2025, 6 Mar 2026, 3 May 2025, 3 Jan 2025, EV tag, auto-loan tag, pledge tag); stat.uz EV; Anhor CERR; GitHub PaddleOCR, rf-detr, sam3, fast-plate-ocr, MoGe, Depth-Anything-3; Hugging Face FLUX.2-klein-4B; MDN MediaTrackConstraints. Failed: Caltrans PDF (twice), Hikvision gas-station page (empty), stable.auto and treehouse.tech (DNS), byd.uz (placeholder), caniuse torch (no data), MAV BayiQ and Environment+Energy Leader (403).
