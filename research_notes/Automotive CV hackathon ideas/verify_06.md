# Verify batch 06: adversarial due diligence on 10 service-aftermarket and fleet-logistics CV ideas (WIUT Hackathon 26, Automotive CV track)

> Method note (not a findings section). Compiled 2026-10-10. The shared WebSearch budget (200 calls across all agents) was already used up when this batch started: all 12 WebSearch attempts (EN and RU, standard and extended, spread over the session) returned "budget used up", so the instruction "one WebSearch per idea" could not be met. Novelty checks instead used primary-source lookups: Apple App Store Search/Lookup API (US, RU, UZ storefronts), Google Play search pages (fetched with WebFetch), GitHub repo/code search (`gh`), arXiv API, Crossref API, Hugging Face Hub API, Google Patents pages, vendor pages and Uzbek news pages fetched directly, plus the teammates' notes in this folder (tags: (T) = taken from a teammate note and not re-fetched by me; everything else was fetched by me on 2026-10-10). OpenAlex and Semantic Scholar were rate-limited. ASTM, Fleetio, echerha.gov.ua, kgd.gov.kz, drishti.com, omnicomm.com (EN) and cargoruqsat.kz were blocked or unreachable. So "no hit" below means "no hit in the sources listed", not "does not exist". Scores and verdicts are in the structured output; this file holds the evidence. Inferences are my reasoning and are labelled.

## wild-2 TintCheck: phone tint meter with permit-ready certificate

### Takeaway
Phone VLT meters already ship in both app stores, and the Uzbek rule as reported is a paid online permit for any front-side tint with no VLT certificate step, so the payer logic and the novelty both fail. Verdict: kill.

### Cited Findings
- Google Play "Tint Meter: Window Tint" (HighApps Mobile, package am.highapps.tintmeter): listing text "Measure car window tint percentage (VLT) with your phone light sensor", 10K+ downloads (details page body was truncated, text came from the search-result extraction) — [Play search](https://play.google.com/store/search?q=tint%20meter%20window%20tint&c=apps&hl=en); [Play app id](https://play.google.com/store/apps/details?id=am.highapps.tintmeter)
- iOS "Window Tint Meter – VLT Meter" (CODIGITAL SOLUTIONS - FZCO, released 2026-01-02, 16 ratings): "measure the Visible Light Transmission of any window", uses the device's light sensors, subscriptions $1.99 weekly / $4.99 monthly / $9.99 yearly — [App Store](https://apps.apple.com/us/app/window-tint-meter-vlt-meter/id6757194952)
- iOS "Tint Meter" (Hayk Arustamyan, released 2026-09-29, 0 ratings): two readings with the front camera (closed window, then lowered window), returns a VLT percentage with a range, re-check for steady light; says "Tint Meter is an estimate, not a certified tint meter... Don't rely on it for legal, police, insurance or certification purposes" — [App Store](https://apps.apple.com/us/app/tint-meter/id6812510825)
- GitHub repo search "tint meter": only GTA5/FiveM game scripts and desktop UIs (e.g. nj-ramadhan/trb-dsk-pykivy-window-tint-meter-app, 0 stars); "window tint transmittance camera": no repos. Crossref "smartphone tint meter visible light transmission car windows accuracy": no relevant paper (a 2022 Russian law paper on administrative liability for car-window light transmission was the only on-topic hit) — [GitHub search](https://github.com/search?q=tint+meter&type=repositories); [Crossref hit](https://doi.org/10.7256/2306-9945.2022.2.37911)
- Uzbek rule: tinting front side windows needs a one-year permit, issued online through the Unified Portal of Interactive Public Services, cost 8 BRV (about 3M UZS at BRV 375k), 10% online discount; rear side and rear windows need no permit; windscreen strip up to 14 cm; the article gives no VLT thresholds, no measuring procedure and no certificate requirement beyond the permit — [Anhor](https://anhor.uz/news/tonirovka/)
- Spot (28 Mar 2025): permit fee cut from 45 to 8 BRV; rear-side tint free since 1 Jul 2022; no permit = 25 BRV (9.375M UZS), repeat 40 BRV; "current law doesn't separate permit absence from violating light-transmission norms"; MIA draft Art. 126-1 would fine breaking permit conditions 1 BRV; no VLT figures, no tint-meter mention — [Spot](https://www.spot.uz/ru/2025/03/28/window-tinting/); [Anhor, MIA draft](https://anhor.uz/news/mia-proposes/)
- BRV rises to 440,000 UZS and minimum wage to 1.36M UZS from 1 Sep 2026 (Spot, 23 Jun 2026), so 8 BRV = 3.52M UZS and 25 BRV = 11.0M UZS [CALC] — [Spot BRV tag](https://www.spot.uz/ru/tag/%D0%B1%D1%80%D0%B2/)
- Ordinary LCD screens contain polarizers (Wikipedia, LCD article); the article does not discuss polarized sunglasses — [Wikipedia](https://en.wikipedia.org/wiki/Liquid-crystal_display)
- HF datasets: `referencesource/state-window-tint-vlt-limits` (licence "other", a US state-limit table) and `zmy1234567890/tintedglass-rdt-synth`; no through-glass VLT photo set — [HF datasets API](https://huggingface.co/api/datasets?search=tint)

#### Queries run
App Store US: "window tint meter", "tint meter", "tint meter VLT camera"; App Store RU: "светопропускание тонировка" (0), "тонометр" (blood-pressure apps only); Play: "window tint meter vlt", "tint meter window tint"; gh repos: "window tint VLT", "tint meter", "window tint transmittance camera"; Crossref x2; OpenAlex (rate-limited). WebSearch x3 (budget used up).

### Inferences
- The measurement is a commodity (three apps, one with 10K+ installs, plus hardware meters the idea itself prices at USD 60-300), and the leading indie app disclaims legal use, so a "permit-ready certificate" has no legal weight unless the traffic police accept it, which no source suggests.
- In the sources read the permit is a payment-gated registration for any front-side film, not a VLT test, so the "legal / permit needed / too dark" taxonomy collapses to "is there film on a front window"; a certificate registry would have no consumer at the permit step. The 150,000 UZS/month shop fee (about USD 12.7 at about 11,800 UZS/USD) is below the price of buying a hardware meter once.
- Physics risks (my reasoning, not sourced): outer-surface reflection of ambient light adds veiling glare that inflates the reading of dark film, and LCD light plus polarized sunglasses makes the demo reading depend on lens rotation. Accuracy of +/-5 points is plausible only in shaded, controlled capture, which is too coarse for threshold decisions.

### Gaps
- Uzbek VLT thresholds (technical regulation), how police measure, and whether the permit portal asks for a VLT figure or film certificate were not found (the Cabinet decision and portal page were not retrieved).
- Google Play app internals (method, accuracy) not retrievable; hardware meter prices not re-checked.

## matrix-9 Orange Guard: EV high-voltage lockout verifier for service bays

### Takeaway
No vision product advertising EV high-voltage lockout/tagout was found among the main safety-vision vendors, so novelty is real, but the payer, the liability of a "gate" and the camera-unverifiable voltage-test step are weak points.

### Cited Findings
- Intenseye homepage: PPE, vehicle-pedestrian, zones, ergonomics, a "Heat & electrical hazard camera (Sentinel Thermal)", Sentinel Hard Stop machine stop; no LOTO, no EV or high-voltage terms; "one subscription per camera" — [Intenseye](https://www.intenseye.com/)
- Protex AI homepage: PPE, near misses, area control, vehicle control; no LOTO or EV text; logos include General Motors, Toyota, Tesla — [Protex](https://www.protex.ai/)
- Voxel homepage: PPE, ergonomics, area controls, vehicle safety; no LOTO or EV/high-voltage use case on the homepage (product pages not read) — [Voxel](https://www.voxelai.com/)
- viAct homepage: PPE, danger zone, area control, vehicle control, near-miss, "Permit to Work" software; no LOTO, no electrical-safety or EV-service text (only an "Automotive & EV" industry link) — [viAct](https://www.viact.ai/)
- Spot AI homepage: generic "SOP compliance" judged "against an SOP or goals you set" by an AI agent; no LOTO, EV bay or high-voltage text. This means a general video-agent vendor could approximate step checks with prompts, which lowers the moat of a bespoke state machine — [Spot AI](https://spot.ai/)
- Retrocausal: patented CV step-by-step verification ("Assembly Copilot tracks individual steps"), bolt-torque vision, battery-pack assembly among automotive cases; no LOTO, no EV service — [Retrocausal](https://www.retrocausal.ai/)
- arXiv API: `all:lockout AND all:vision` 0 results; `all:"high voltage" AND all:"electric vehicle" AND all:safety AND all:"computer vision"` 0 results. GitHub "lockout tagout": only management apps, no CV; "ev high voltage safety yolo": none. HF datasets "lockout tagout" and "insulated gloves": 0 results — [arXiv API](https://export.arxiv.org/api/query?search_query=all:lockout+AND+all:vision); [HF](https://huggingface.co/api/datasets?search=lockout%20tagout)
- LOTO procedure ends with demonstrating isolation is effective; EN 50110-1 step "verify that the installation is dead" — [Wikipedia LOTO](https://en.wikipedia.org/wiki/Lockout%E2%80%93tagout)
- 122.7 thousand EVs registered in Uzbekistan on 1 Jul 2026, Tashkent city 88,496 (Statistics Committee, 21 Sep 2026) — [stat.uz](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4)
- EV imports Jan-Jun 2026: 37,874 units, USD 447.7M, 2.6x the same period of 2025, 99.2% from China — [Spot](https://www.spot.uz/ru/2026/08/14/ev-import-rise/)
- byd.com/uz homepage lists eight models but no dealers, service points or HV-service terms, so the number of BYD service bays could not be sized — [BYD Uzbekistan](https://www.byd.com/uz)
- Models and licences (T): RF-DETR Nano 2.3 ms on a T4 (Apache-2.0), SAM 3 weights gated under the SAM License, SAM 2.1 Apache-2.0, `roboflow/trackers` Apache-2.0 — [tech_radar.md](tech_radar.md)

#### Queries run
Fetched vendor homepages (Intenseye, Protex, Voxel, Retrocausal; drishti.com did not resolve); arXiv x2; gh repos x3; HF datasets x2; App Store RU "электромобиль высоковольтная безопасность" (0). WebSearch (extended) used up.

### Inferences
- Nothing found in these sources does EV-specific lockout sequencing, but generic procedure-verification vendors (Retrocausal) could add it; "no hit" is limited by the search outage.
- The only step that truly protects the technician, verifying absence of voltage with a meter, cannot be confirmed from a bay camera beyond seeing that a meter was used; a green light that unlocks the work order risks false assurance and liability, which is a reason to use the product as an assessor or audit log rather than a gate.
- Market size is unknown: no count of BYD or other EV service bays was found. With USD 25 per bay-month, 50 EV bays nationwide would be about USD 1,250 per month [CALC with an assumed bay count].

### Gaps
- Number of authorised BYD/EV service points and bays; any OEM HV-lockout checklist for BYD models; whether Uzbek dealers have cameras in bays; Jetson Orin Nano Super street price (not on NVIDIA's page per teammate notes).

## matrix-11 Razborka Lens: teardown video to sellable parts catalogue

### Takeaway
Single-part ID apps are a commodity and CIS dismantler apps already ship donor-car digitisation, part-number photo recognition and multi-platform price export, so the idea must be narrowed to what those do not do.

### Cited Findings
- Robzar (released 2026-07-29), "inventory system for auto dismantlers and parts shops": barcode scanner, donor cars with fitment and photos, "Поступления — серийная оцифровка запчастей с разобранной машины, этапы и баллы оцифровки", "Распознавание номера детали по фотографии и подсказки номеров по ранее оцифрованным машинам", payback per donor car, showcase of machine kits — [App Store RU](https://apps.apple.com/ru/app/robzar/id6792543052)
- Baza iDriver (2025-09-29): smartphone CRM for dismantlers; photos from the phone, "автоматическая выгрузка прайсов на десятки площадок", QR sales, donor-kit payback — [App Store RU](https://apps.apple.com/ru/app/baza-idriver/id6752250808)
- Marketplaces in the region: AutoRazbor (autorazbor.kz), myrazbor.kz, "Авторазбор KG" (Kyrgyzstan), CARRO (4M used parts, KZ/BY/RU/UA), Drom Zapchasti (180M parts, upload price lists), Uzbek Avtoehtiyot app (2026-06-25) — [AutoRazbor](https://apps.apple.com/ru/app/autorazbor/id6762520773); [CARRO](https://apps.apple.com/uz/app/carro-%D0%BC%D0%B8%D0%BB%D0%BB%D0%B8%D0%BE%D0%BD%D1%8B-%D0%B1-%D1%83-%D0%B7%D0%B0%D0%BF%D1%87%D0%B0%D1%81%D1%82%D0%B5%D0%B9/id1454397872); [Drom Zapchasti](https://apps.apple.com/ru/app/%D0%B4%D1%80%D0%BE%D0%BC-%D0%B7%D0%B0%D0%BF%D1%87%D0%B0%D1%81%D1%82%D0%B8/id1082252059); [Avtoehtiyot](https://apps.apple.com/uz/app/avtoehtiyot/id6780532509)
- Consumer part-ID from a photo: MechanicML (VIN-aware fitment, price compare, "AI verified listings"), PartsIQ, Car Part Identifier (USD 4.99/month); Play has at least four more "Car Part Identifier" apps — [MechanicML](https://apps.apple.com/us/app/mechanicml-identify-car-parts/id6752443699); [PartsIQ](https://apps.apple.com/app/id6760045058); [Car Part Identifier](https://apps.apple.com/us/app/car-part-identifier/id6738489603); [Play search](https://play.google.com/store/search?q=car%20part%20identifier%20photo&c=apps&hl=en)
- US benchmark: Car-Part.com, "150 million parts from 3100 Auto Recyclers" — [App Store](https://apps.apple.com/us/app/car-part-com-auto-parts-market/id1024205586)
- OLX.uz shows 67,363 "Автозапчасти и аксессуары" ads (read 2026-10-10; teammate saw 67,430) — [OLX](https://www.olx.uz/transport/)
- Sergeli car market raid (31 May, year not printed): intermediaries sold parts worth about 100,000 UZS for 1,000,000 UZS; about 100 people identified — [Anhor](https://anhor.uz/ru/news/zloumyshlenniki-vstrechali-posetitelej)
- Cobalt 95,362, Damas 78,712, Tracker 27,921, Labo 25,129 sold Jan-Sep 2025; Chevrolet 83% of new sales — [Kun.uz](https://kun.uz/en/news/2025/10/11/uzbekistans-car-market-sees-growing-competition-as-chevrolet-share-drops)
- PP-OCRv6 "supports 50 languages... Chinese, English, Japanese, and 46 Latin-script languages" (no Cyrillic); PP-OCRv5 multilingual covers Cyrillic — [PaddleOCR README](https://github.com/PaddlePaddle/PaddleOCR)
- HF `DrBimmer/car-parts-and-damage-dataset` (MIT, polygon masks of parts and damage) is the closest open seed; no open Chevrolet/BYD part-catalogue images found — [HF](https://huggingface.co/api/datasets?search=car%20parts)
- SAM 3 and DINOv3 weights are gated under custom licences (T) — [tech_radar.md](tech_radar.md)

#### Queries run
App Store US: "car part identifier photo", "auto parts scanner identify part number photo", "dismantler auto recycler inventory"; RU: "авторазбор", "автозапчасти", "Авито" (sanity check), "распознать запчасть по фото" (0), "запчасти по фото" (0), "razborka avtozapchast" (0); UZ: "автозапчасти", "запчасти", "avtoehtiyot qismlar"; Play: "car part identifier photo"; gh repos "car parts recognition" (tiny repos); HF datasets "car parts", "vin"; fetched OLX, Anhor, Kun.uz.

### Inferences
- Robzar's feature list (July 2026) overlaps the core: donor digitisation, part-number recognition from a photo, fitment hints from earlier cars. What I did not find in any description is video-based teardown segmentation, condition grading, or Uzbek/Russian listing drafting; absence is limited by truncated store descriptions.
- Time-saving economics [CALC]: average wage 7.09M UZS (about USD 600) per month (T, Kun.uz 27 Jul 2026) is about USD 3.75 per hour at 160 hours, so 2 saved hours per donor is about USD 7.5; USD 25 per month breaks even on time savings alone at about 3.3 donor cars per month. Revenue uplift from better listings is unsourced.
- Stamped, rusty OEM numbers are the hardest OCR case and most body panels, lamps and bumpers carry no number, so the real differentiator is shape/fitment retrieval for number-less parts.

### Gaps
- Number of dismantlers and active Telegram channels in Uzbekistan; whether any CRM above operates in Uzbekistan; availability of Chevrolet/BYD parts catalogues with images (UzAuto site unreachable).

## matrix-14 Claim Pack: warranty evidence builder for dealer service desks

### Takeaway
OEMs, claim-photo vendors and OCR SDKs already cover each component (guided capture, VIN/odometer OCR, duplicate detection), the VIN check digit is not universal, and the dealer is not the party that controls the claim format. Weakest idea of the sharpen set.

### Cited Findings
- GM's dealer app CSMT (updated 2026-05-26): pre-authorise warrantable repairs, "use photos to improve efficiency of the Part Return Process", part traceability, field product reports, all modules accept images and video — [App Store](https://apps.apple.com/us/app/csmt-certified-service/id665252466)
- Inspektlabs: smartphone guidance system, mandatory sections and forms, fraud FAQ "identify digitally altered images, duplicate photos, poor quality submissions, and attempts to conceal prior damage"; per-inspection pricing plus one-time integration fee, 100 free inspections; no mention of warranty claims or dealer service — [Inspektlabs](https://www.inspektlabs.com/)
- Bdeo motor-claims page: "VIN, GPS, plate, odometer and screen detection. Five anti-fraud controls in one analysis"; "15%" fraud detected (vendor claim, method unstated); no duplicate-photo or AI-image claim on the page — [Bdeo](https://www.bdeo.io/en/blog/detecting-auto-insurance-fraud-in-the-age-of-generative-ai/)
- Anyline VIN scanning: reads VINs on chassis, door sticker, under the windshield, SDK for iOS/Android/Web, customers include a Toyota dealer, Shopmonkey, NAPA, BMW service; no check-digit statement — [Anyline](https://anyline.com/products/vin-scanning)
- VIN check digit (position 9) is "compulsory for vehicles in North America and China", not in Europe, and the article is silent on Korea, Japan and Uzbekistan — [Wikipedia VIN](https://en.wikipedia.org/wiki/Vehicle_identification_number)
- HF odometer sets exist (`unc061/cz-stk-odometer`, `si-tp-odometer`, CC-BY-4.0, small) — [HF](https://huggingface.co/api/datasets?search=odometer)
- Sponsors are Toyota Royal Motors and BYD Royal Motors; no insurer or UzAuto Motors on hackathon material (T) — [hackathon_intel.md](hackathon_intel.md)

#### Queries run
Fetched Bdeo, Inspektlabs, Anyline, Wikipedia VIN; App Store US/RU: "warranty claim" style queries ("гарантийный случай автосервис фото" 0), "СТО приемка автомобиля фото осмотр" (checklist app Inspectro only); gh repos "warranty claim photos" (none); HF "odometer", "vin".

### Inferences
- Because VIN check digits are mandatory only in North America and China, validating one would false-flag valid Toyota (Japan/Europe-built) and possibly Chevrolet-UZ VINs; it can only be a soft signal.
- Warranty failures are mostly proven by diagnostic trouble codes, measurements and returned parts, not by a visible defect in a photo, so the "VLM verifies failure visible against the OEM checklist" step will rarely apply.
- The party that loses money on fraudulent or inflated warranty claims is the importer/OEM, not the dealer; the dealer cannot choose a claim format that the OEM portal does not accept. No Uzbek warranty claim volumes or rejection rates were found, so USD 0.50 per claim cannot be sized.

### Gaps
- Uzbek OEM warranty systems and photo requirements (Toyota, BYD, UzAuto Motors); claim volumes and chargeback rates; whether Uzbek dealers use a DMS with an integration point.

## wow-10 Wrench-Proof: 20-second verified service receipts

### Takeaway
The "missing Carfax layer" is real in Uzbekistan, but the state is already building the registry for the proposed gas-kit wedge, shops have no reason to pay, and a shop-filmed hash proves integrity rather than honesty.

### Cited Findings
- UGaz (UZINFOCOM, 11 Sep 2025): "records all transactions in real time at each filling station"; includes a licence-plate recognition system that "verifies the vehicle's CNG equipment registration and inspection status", video surveillance; tariff 25 UZS per m3 from 24 Sep 2025 — [UZINFOCOM](https://uzinfocom.uz/news/378)
- Cabinet measures of 18 Jun 2026: Interior Ministry and Digital Ministry must integrate Eco-Himoya gas-cylinder inspection data into U-Gas so station operators can check a car's cylinder equipment; 6,196 fuel facilities inventoried in 2025, 46,737 violations, 464 stations suspended; the article has no rules for installers or workshops — [Spot](https://www.spot.uz/ru/2026/06/28/gas-station-safety/)
- Sergeli intermediaries sold parts worth about 100,000 UZS for 1,000,000 UZS — [Anhor](https://anhor.uz/ru/news/zloumyshlenniki-vstrechali-posetitelej)
- Service-history apps exist: CARFAX Car Care ("Join 50 million+ drivers", 127,813 ratings) and, in Russia, Splus ("Уже более 1000 автосервисов подключены к нашей системе") — [CARFAX Car Care](https://apps.apple.com/us/app/carfax-car-care/id552472249); [Splus](https://apps.apple.com/ru/app/%D1%81%D0%B5%D1%80%D0%B2%D0%B8%D1%81%D0%BD%D0%B0%D1%8F-%D0%BA%D0%BD%D0%B8%D0%B6%D0%BA%D0%B0-splus/id6472249499)
- Checklist and photo tooling for mechanics: Inspectro (2026-09-01) with photos and defect circling — [Inspectro](https://apps.apple.com/ru/app/%D0%BC%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA-inspectro/id6805891374)
- carVertical lists "1000+ data sources across 45+ countries" from insurance, police and registration data, no service records and no country list on the page — [carVertical](https://www.carvertical.com/)
- No Uzbek vehicle-history product found (T) — [uzbekistan_market.md](uzbekistan_market.md)
- `c2pa-rs` (MIT/Apache) offers Content Credentials signing for the "tamper-evident" part (T) — [tech_radar.md](tech_radar.md)

#### Queries run
App Store US: "vehicle service history record maintenance log", "verified service record mechanic proof"; RU: "история обслуживания автомобиля сервисная книжка"; gh repos none relevant; fetched UZINFOCOM, Spot, carVertical.

### Inferences
- UGaz plus Eco-Himoya make the state the verified record for CNG equipment, which pre-empts "starting with regulated gas-kit installs".
- A shop controls the camera, so staging remains possible; plate + odometer + part box + old part in one continuous shot reduces but does not remove it. The party defrauded by phantom replacements is the fleet owner or the car buyer, not the shop.

### Gaps
- Who installs and certifies CNG/LPG kits in Uzbekistan and what paperwork they issue; willingness of shops to adopt; whether Splus-style networks have Uzbek coverage.

## usedcar-15 FreshOrOld: damage age dating for handover disputes

### Takeaway
No product or paper dates vehicle damage from photos, which is also what makes it implausible: the dispute window is weeks, the proposed labels cover 0-48 hours, and confounders dominate. Verdict: kill.

### Cited Findings
- arXiv API: `all:"damage age" AND all:vehicle` 1 irrelevant hit; `all:"old damage" AND all:vehicle` 0; `all:scratch AND all:aging AND all:paint AND all:image` 0 — [arXiv API](https://export.arxiv.org/api/query?search_query=all:%22old+damage%22+AND+all:vehicle)
- Crossref "age estimation of scratches on vehicle paint forensic dating damage": no relevant paper (a 1986 windscreen-scratch forensic note and unrelated results); GitHub "damage age": none relevant — [Crossref](https://api.crossref.org/works?query=age+estimation+of+scratches+on+vehicle+paint+forensic+dating+damage&rows=6)
- Inspektlabs only mentions "attempts to conceal prior damage" in its fraud FAQ and does not explain how old and new damage are told apart; Ravin's page mentions "Tracks vehicle condition changes over time" (Ravin Eye) and no dating; Tractable's page does not address pre-existing damage or dating — [Inspektlabs](https://www.inspektlabs.com/); [Ravin](https://www.ravin.ai/); [Tractable](https://tractable.ai/)
- Since 1 Nov 2024 imported vehicles undergo technical testing at a specialised polygon in Piskent; the article says nothing about damage or condition inspection; January 2025 imports 1,796 vehicles (USD 31M) vs 10,152 (USD 221.4M) in January 2024 — [Kun.uz](https://kun.uz/en/news/2025/02/24/uzbekistan-sees-sharp-decline-in-passenger-car-imports-amid-non-tariff-restrictions)
- EV imports Jan-Jun 2026: 37,874 units, USD 447.7M, 99.2% from China — [Spot](https://www.spot.uz/ru/2026/08/14/ev-import-rise/)
- Baseline-delta handover scanners are an existing crowded category (ProovStation, DeGould, UVeye, Ravin Autoscan, carpodX) with no Central Asian deployment found (T) — [saturation_map.md](saturation_map.md)

#### Queries run
arXiv x5; Crossref x1 (+ S2 attempts rate-limited); gh repos "damage age"; fetched Inspektlabs, Ravin, Tractable, Kun.uz, Spot.

### Inferences
- The "natural damage checkpoint at Piskent" is the idea's assumption: the cited article describes technical testing only.
- Training labels of 0, 6, 24, 48 hours cover the first two days; transit disputes span weeks and the confounders (dirt, rain, wax, lighting, paint type, plastic vs metal) swamp the signal. Extremes (minutes vs years with rust) are separable but are not what a dispute needs. Using the output as liability evidence also exposes the team to false accusations.
- If the team wants the import angle, the credible product is a baseline-delta capture chain (C2PA-signed photos at origin and arrival), which is crowded globally and thin locally.

### Gaps
- No damage-dating study found (limited by search outage); the insurance patent the idea cites was not located; no data on dispute frequency or cost in Uzbek imports.

## service-11 Spot-Oil: oil blotter-spot photo to keep-or-change

### Takeaway
The blotter-spot method is a standardised test (ASTM D7899, dispersancy merit) and I found no phone-ML product around it, so novelty is real, but the method measures dispersancy only, ground truth is missing and gas engines may defeat it.

### Cited Findings
- ASTM D7899, "Test Method for Measuring the Merit of Dispersancy of In-Service Engine Oils with Blotter Spot Method" (editions 13, 19, 24) — [DOI](https://doi.org/10.1520/d7899-24). ASTM D4291 "Trace Ethylene Glycol in Used Engine Oil" is a separate chemical test — [DOI](https://doi.org/10.1520/d4291-21). (ASTM page itself returned 403.)
- App Store US/RU, Play, GitHub (repos and code), arXiv, Crossref: no app, repo or paper combining a blotter/oil-spot card with phone image analysis. App hits were lab-sample logistics apps only (Shell "Oil Sample Analysis", Polaris "HORIZON"); Play "Oil Lab" (Amer M A) has no retrievable description. "TapCorder" (named in the idea) returned no App Store result — [Shell app](https://apps.apple.com/us/app/oil-sample-analysis/id1575505655); [Play search](https://play.google.com/store/search?q=engine%20oil%20spot%20test%20analysis&c=apps&hl=en)
- Wikipedia "Oil analysis" describes lab practice only (wear metals, additives to decide whether an extended drain interval may be used) and does not mention blotter tests — [Wikipedia](https://en.wikipedia.org/wiki/Oil_analysis)
- Avtoelon.uz fuel-typed car listings (read 2026-10-10): petrol-gas 9,965 of 29,402 (33.9%), gas 448 — a proxy for dual-fuel prevalence in used cars, not taxi fleets — [Avtoelon](https://avtoelon.uz/)
- Yandex Go works through fleets, 135 fleets in Mar 2023 (T, KSE thesis) — [uzbekistan_market.md](uzbekistan_market.md)
- No public oil-stain dataset: HF datasets "oil stain" 0 results.

#### Queries run
App Store US: "engine oil analysis", "oil spot test blotter", "oil analysis camera AI engine oil condition", "TapCorder"; RU: "анализ масла двигателя" (0), "тест масла"; Play: "engine oil spot test analysis"; gh repos "oil spot test", "oil analysis computer vision"; gh code "blotter spot oil", "oil dispersancy spot"; Crossref x5; arXiv none specific.

### Inferences
- D7899 judges dispersancy; it does not measure TBN depletion, viscosity change, oxidation or fuel dilution, so a "safe to extend" verdict is not supported by the method's scope. Coolant flags would need glycol chemistry (D4291) to validate.
- For CNG/LPG dual-fuel engines (my reasoning, unsourced): low soot may leave spots looking clean while oil is oxidised or TBN-depleted, the opposite of the diesel conditions the method was built around. This is the key risk to test.
- Ground truth must come from lab analyses, not mechanics' opinions; a few dozen lab reports would turn a gimmick into evidence.

### Gaps
- Lab cost and availability in Tashkent; oil change prices and intervals for taxi fleets; any CIS commercial spot-test kit (none found; Russian-language search was limited to App Store).

## fleet-1 FuelTruth: photo fuel audit, no telematics

### Takeaway
Fleet fuel-fraud tooling is established (cards, receipts, hardware sensors) and the CV parts are commodity OCR; the genuine local hook is the Uzbek e-waybill's mandatory fuel fields and the "blank cheque" fraud mode, not gauge physics.

### Cited Findings
- Piston Fuel (2024): "keep track of drivers' fuel purchases... reconcile them with the gas station owner... limiting fraud"; Coast: receipts, memos, odometer data added by the user; Fleetio Go: "Log fuel-ups and see fuel history", photos and documents; WEX Fleet SmartHub: "Spot card misuse" — [Piston Fuel](https://apps.apple.com/us/app/piston-fuel/id6502339448); [Coast](https://apps.apple.com/us/app/coast-smart-fuel-cards/id6502646569); [Fleetio Go](https://apps.apple.com/us/app/fleetio-go-fleet-management/id1100421418); [WEX](https://apps.apple.com/us/app/fleet-smarthub/id1051414868)
- Hardware incumbent in CIS: Omnicomm (Russia) claims "average fuel saving reaches 30%" and that its solution "rules out theft, draining and under-filling" (vendor claims, under its fuel-consumption-control page, no price) — [Omnicomm](https://www.omnicomm.ru/)
- Uzbek electronic waybill (pilot from 1 April): must contain "информация о количестве топлива, его остатке, экономии топлива"; applies to passenger and freight companies including non-commercial carriers; self-employed drivers in own cars are out of scope; domestic trips electronic only; no mandatory start date stated — [Buxgalter.uz](https://buxgalter.uz/publish/doc/text196198_kak_budet_rabotat_sistema_ucheta_elektronnyh_putevyh_listov)
- UGaz logs every CNG transaction in real time with plate recognition (see wow-10) — [UZINFOCOM](https://uzinfocom.uz/news/378)
- Methane retail price 5,700-5,800 UZS/m3 from 1 Jun 2026 (about 5,250 before), wholesale 2,700 — [Anhor](https://anhor.uz/news/metan-na-zapravkah-uzbekistana-podorozhal/)
- A "Soliq OFD" page exists at ofd.soliq.uz (content not rendered, so whether CNG receipts expose litres by QR is unverified) — [Soliq OFD](https://ofd.soliq.uz/)
- HF: `tomasko1987/seven-segment-digits` (MIT, 72 downloads), small odometer sets (CC-BY-4.0), no gas-pump display set; GitHub "fuel receipt ocr": one tiny repo; "fuel pump display ocr": none — [HF](https://huggingface.co/api/datasets?search=seven%20segment)
- PP-OCRv6 covers Latin digits; Cyrillic needs PP-OCRv5 multilingual (see matrix-11)

#### Queries run
App Store US: "fuel receipt odometer fleet", "fuel theft fleet", "fuel log receipt scan fleet driver", "AI odometer photo OCR"; RU: "учет топлива фото чек одометр" (0), "путевой лист топливо водитель" (REIS, DropLane); Play: "fuel receipt odometer photo fleet" (FleetKO, Fuelio, Drivvo, Technoton Fuel Rate Monitor); gh repos x2; HF x3; fetched Omnicomm, Technoton, buxgalter, UZINFOCOM.

### Inferences
- The idea's 8% skimming rate and the USD 3 price come with no source; at 2.8M UZS per month (about 487 m3 at 5,750 UZS/m3 [CALC]) an 8% saving is about USD 19, so USD 3 is 16% of the claimed saving.
- Dash fuel gauges are damped and non-linear (my reasoning), so needle-based physics checks only catch blatant lies; the strong signal is the time series of litres vs odometer across fills.
- Taxi drivers in Uzbekistan mostly pay their own fuel (self-employed on own cars), so the payer must be corporate, municipal or logistics fleets covered by the e-waybill.

### Gaps
- Fraud rate in Uzbek fleets; which fleets already use fuel sensors; whether CNG receipts carry a verifiable QR; whether the e-waybill moved beyond the pilot.

## fleet-3 QueueLedger: crowd plate-chain border wait times

### Takeaway
The pain is documented (400+ trucks, 24 h waits) and no Uzbek-Kazakh queue app turned up, but similar apps for Belarus run on official e-queue data, CV is marginal against a GPS-only baseline, and the crowd cold start cannot be proven in a hackathon.

### Cited Findings
- Tazhen crossing: over 400 trucks stuck (Kazakhstan State Revenue Committee), design 150 trucks per day one-way, actual 200-250 per day, cause the new Dovud ota-Kungirot highway; no electronic queue or monitoring mentioned — [Kun.uz](https://kun.uz/en/news/2024/11/26/hundreds-of-trucks-stranded-at-kazakhstan-uzbekistan-border-amid-increased-traffic)
- June 2025: Zhibek Zholy-Gisht-Kuprik crossing closed for renovation, drivers "stuck in lines for more than 24 hours", traffic redirected to Kaplanbek, Kazyghurt-Mayskiy, Yallama; no e-queue or app mentioned — [Fergana Agency](https://en.fergana.agency/articles/138264/)
- BYBO (2026-10-04) and Belarus Borders (2026-10-08): live queues for Belarus-EU crossings from official data (belarusborder.by), e-queue registration, forecast, alarm-style call notification; Belarus Borders also sells a "Следить" feature where the phone camera watches the car ahead and wakes the driver when it moves — [BYBO](https://apps.apple.com/ru/app/bybo-%D0%BE%D1%87%D0%B5%D1%80%D0%B5%D0%B4%D1%8C-%D0%BD%D0%B0-%D0%B3%D1%80%D0%B0%D0%BD%D0%B8%D1%86%D0%B5-by/id6808294602); [Belarus Borders](https://apps.apple.com/ru/app/belarus-borders-%D0%BE%D1%87%D0%B5%D1%80%D0%B5%D0%B4%D1%8C-%D0%BA%D0%BF%D0%BF/id6809185331)
- US/Mexico/Canada: CBP Border Wait Times, Bordify, Border Traffic (live video) apps; Play: "Пульс границы Беларусь" (Road Soft), "Border AI", "Granica" — [CBP app](https://apps.apple.com/us/app/cbp-border-wait-times/id948812864); [Play search](https://play.google.com/store/search?q=border%20queue%20wait%20time%20trucks&c=apps&hl=en)
- Not verifiable: the idea's "Kazakhstan CargoRuqsat books slots, not real waits" (cargoruqsat.kz and ruqsat.kz do not resolve; kgd.gov.kz refused the connection; customs.uz homepage text shows no queue system).
- fast-plate-ocr (MIT) and open-image-models (MIT), RF-DETR (Apache-2.0), MoGe-2 (MIT tag, vits 141 MB) (T); Central Asia plates dataset, 500 images, CC BY 4.0 (T) — [tech_radar.md](tech_radar.md); [saturation_map.md](saturation_map.md)
- TIMOCOM UK homepage: no border-wait feature visible (Live Shipment Tracking only) — [TIMOCOM](https://www.timocom.co.uk/)

#### Queries run
App Store US: "border wait times trucks"; RU: "очередь на границе грузовики" (0 then 7 hits, mostly games), "очередь на границе", "граница КПП очередь Казахстан"; Play EN/RU border queue queries; domain probes; fetched Kun.uz, Fergana Agency, customs.uz, TIMOCOM.

### Inferences
- The state e-queue model (slot booking plus public queue counts, as in Belarus) is what drivers use elsewhere; a crowd product competes with, or must wait for, an Uzbek-Kazakh equivalent.
- A GPS-probe baseline (location shared by drivers or trackers) gives order and wait without any plate chaining, so the CV has to demonstrate added accuracy over it.
- Revenue sketch [CALC, assumptions]: Tazhen handles 200-250 trucks per day; 20% adoption at USD 3 is about USD 135 per day, about USD 4k per month; WTP is unvalidated.

### Gaps
- Existing Uzbek-Kazakh e-queue or booking systems; how truckers share queue info today (Telegram groups not checked); real truck-day idle cost in Uzbekistan.

## fleet-4 SagScan: suspension-sag overload triage for weighbridges

### Takeaway
The core method is the subject of an active Renault patent and Geotab/Korean follow-ons, it needs a per-model calibration and empty baseline that a gate guard does not have, and the named payer weighs trucks, not Damas vans. Verdict: kill.

### Cited Findings
- US 9,989,402 B1 (Renault SAS, granted 2018-06-05, anticipated expiry 2033-07-24): smartphone photographs a wheel, finds wheel-centre-to-housing clearance, gets the opposite axle by tilt (inclinometer) change, converts clearance to load with a per-model map built by a factory calibration step; every independent claim requires the tilt measurement; the photograph-only approach is in the description but not claimed alone; ask users to re-photograph the empty vehicle once or twice a year for creep; notes loading/unloading hysteresis; states no numeric accuracy; tyre pressure not mentioned — [Google Patents](https://patents.google.com/patent/US9989402B1/en)
- Cited by Geotab US12036998B2, US12291215B2, US12208811B2 ("Systems and methods for determining an estimated weight of a vehicle"); similar documents KR20230057524A ("System and Method for Calculating an Image based Vehicle Loading Weight"), US10060782B2 and US20180136032A1 (Renault) — [Google Patents](https://patents.google.com/patent/US9989402B1/en)
- arXiv `all:"vehicle weight" AND all:suspension AND all:image` 0 results; Crossref hits unrelated; GitHub "truck overload detection" gives student YOLO repos; Play has only calculators ("Truck Weight Calculator", "Axle Weight Scale for Trucker") — [Play search](https://play.google.com/store/search?q=vehicle%20weight%20payload%20estimate%20suspension&c=apps&hl=en)
- Prior vision-overload work (T): Tsinghua weighing-station system 93.18% capacity estimate, SAE vehicle-dynamics method 96.0% on one model — [saturation_map.md](saturation_map.md)
- Damas 78,712 and Labo 25,129 units sold Jan-Sep 2025 — [Kun.uz](https://kun.uz/en/news/2025/10/11/uzbekistans-car-market-sees-growing-competition-as-chevrolet-share-drops)
- Planned Tashkent ban on trucks over 10 t gross weight (Uzbekistan-2030 plan, T) — [uzbekistan_market.md](uzbekistan_market.md)

#### Queries run
Fetched Google Patents twice (claims and description/cited-by); arXiv x3; Crossref x3; gh repos x2; Play x1; App Store "vehicle weight estimate photo load".

### Inferences
- Without a model-specific empty reference, unit-to-unit ride-height spread, tyre pressure and wear are likely larger than the sag from a few hundred kilograms (my physics reasoning), so absolute kg is not credible and only coarse triage remains.
- Quarry, cement and bazaar haulers weigh heavy trucks with stiff leaf springs where sag is small relative to load; the vans the method suits rarely visit weighbridges, so the named payer and vehicle class do not match.
- Freedom-to-operate: the independent claims of the Renault patent include tilt measurement; a photo-only design may avoid the claims but this is not legal advice.

### Gaps
- Whether any Uzbek weighbridge operator or enforcement body screens light vans; spring rates for Damas/Labo; Renault's commercial use of the patent (not found).

## Cross-cutting: programme time and licence traps for all ten

### Takeaway
The finals programme gives about 8.5 on-campus hacking hours (T), so any idea needing field data collection must shrink to a scripted slice; several models named in the ideas are gated or limited in language.

### Cited Findings
- Finals programme: roughly 5 h on Day 1 plus 3.5 h on Day 2 before the 13:00 code freeze; 20 teams per track, 7+3 minute pitches; weights 25/25/25/15/10 (innovation, technical complexity, impact, presentation, business) (T) — [hackathon_intel.md](hackathon_intel.md)
- SAM 3 and DINOv3 weights are gated with manual approval and custom licences; SAM 2.1/EdgeTAM are Apache-2.0; PP-OCRv6 handles Latin scripts only (50 languages), Cyrillic needs PP-OCRv5 multilingual; WebGPU full support on iOS only from Safari 26 (T) — [tech_radar.md](tech_radar.md); [PaddleOCR README](https://github.com/PaddlePaddle/PaddleOCR)

### Inferences
- Fallbacks should be named in each build plan: SAM 2.1/EdgeTAM if SAM 3 approval is slow, WASM inference for phones without WebGPU, and PP-OCRv5 multilingual for Cyrillic receipts.

### Gaps
- No measured phone-browser latency for any model above; judges' phones are unknown.
