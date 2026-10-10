# Adversarial verification, batch 04: Alibi, ChipCheck, VoltTow, Drip-ID, Smoke-Read, PlugLens, WearMap, BeamAim, SwirlMap, HueMatch (WIUT CV track, automotive / road safety, 2026-10-10)

> Method note (not a findings section). The shared WebSearch quota (200 per session, about 100/hour refill, shared with every agent) was exhausted for most of this run. Of about 25 WebSearch attempts only 5 returned results (3 for Alibi, 1 for BeamAim, 1 for Drip-ID). I did not route around the limit through general search engines, proxies or archives. Instead I used: (a) direct fetches of primary pages; (b) catalogue APIs of code, model and app registries: GitHub (`gh search repos`), Hugging Face Hub API, Apple iTunes Search API (US storefront; the RU storefront returned nothing, the UZ storefront returned one weak hit), arXiv API, Iowa Environmental Mesonet METAR API. "None found" therefore means "none in these catalogues plus the few web searches that ran", which is weaker than a full web search; I capped novelty at 4 for that reason. Facts tagged [teammate] come from the saturation map / Uzbekistan fact sheet pasted in the task or from tech_radar.md and hackathon_intel.md in this folder and were not re-fetched. Figures tagged [computed] are my own arithmetic.
>
> Queries that returned results: WebSearch "Amped Authenticate shadows sun position filter"; "verify photo capture time from shadow direction sun azimuth image forensics" (extended); "insurance claim photo sun position shadow verification"; "phone camera headlight aim alignment check app"; "AI identify car fluid leak from photo color puddle app". Blocked by quota (not answered): windshield chip AI app; EV tow dispatch photo; exhaust smoke AI diagnosis; spark plug AI app; tyre wear alignment AI; paint colour match app; paint swirl AI; insurance weather verification. iTunes US catalogue: headlight aim alignment; windshield chip repair; windshield damage; auto glass damage AI; car fluid leak identifier; coolant leak; oil leak under car; tire tread depth scanner; tire wear pattern; car paint color match; car paint code finder; automotive paint matching; paint thickness gauge; spark plug reading; exhaust smoke diagnosis; tailpipe smoke; car swirl marks detailing; car swirl scratch detection; paint correction inspection; headlight restoration; headlight beam; tow truck dispatch photo; EV towing; vehicle roadside assistance photo. GitHub repos: headlight, headlamp, spark plug, tire wear, tyre wear, tire tread depth, oil leak, car leak, exhaust smoke, smoke opacity, car paint, paint code, paint defect detection, scratch swirl, windshield, windshield crack, tow truck, shadow sun position photo, sun azimuth estimation image. HF datasets: windshield, tire wear, tyre, spark plug, headlight, car paint, oil leak, smoky vehicle, exhaust smoke, paint defect, vehicle color, shadow. arXiv: tire wear, spark plug, windshield damage/crack, smoky vehicle, headlight aim, paint defect, fluid leak, sun position verification, vehicle colour fade, EV towing (mostly empty).

## wow-1 Alibi: sun-and-weather witness for claim photos

### Takeaway
The physics is real and academically validated (single-shadow sun-position checks, 91.5% detection of falsified photos in a research prototype) and no insurer-grade product that automates shadow-azimuth plus weather checks was found, but the pipeline as specified has four weak points: camera yaw is unknown in Telegram photos, the "free weather API" is non-commercial and 25 km coarse, the check cannot catch locally edited real photos, and there is no local evidence of GenAI-staged OSAGO claims. Scores: novelty 4, feasibility 2, demo 3, business 2, moat 3, local fit 3. Verdict: sharpen.

### Cited Findings
- Amped Authenticate's "Check Sun Position" is a menu item that opens a pre-configured SunCalc page; the comparison is visual and manual (user supplies GPS, time, object height, measures the shadow), "the tool does not measure shadows or produce a verdict" — [Amped blog, Mar 2020](https://blog.ampedsoftware.com/2020/03/03/easily-verify-time-and-location-metadata-with-amped-authenticate)
- Amped's Shadows filter checks whether user-clicked shadow points are consistent with one point-like light source (wedge constraints); it does not estimate sun azimuth or capture time — [Amped blog, Nov 2020](https://blog.ampedsoftware.com/2020/11/10/how-to-quickly-find-manipulated-objects-with-amped-authenticates-shadows-filter/)
- Academic prior art: IMAGEGUARD (Li et al., 2018) estimates the sun position from a single shadow and compares it with the astronomical value for the claimed time and place; with thresholds of 9.4 degrees (sun position) and 5 degrees (altitude) it identified 91.5% of falsified photos (abstract; datasets and limitations not on the page) — [arXiv 1811.08951](https://arxiv.org/abs/1811.08951)
- Earlier academic version: "Verifying Temporal Data in Geotagged Images via Sun Azimuth Estimation" (Kakar et al., IEEE TIFS 2012); PDF returned HTTP 403 so contents unread. A search summary states camera data is needed for the azimuth but not for the altitude angle, and "I did not find a current, standardized forensic tool or workflow" — [ResearchGate listing](https://www.researchgate.net/publication/254059115_Verifying_Temporal_Data_in_Geotagged_Images_Via_Sun_Azimuth_Estimation)
- OSINT tool: Bellingcat ShadowFinder (MIT, 619 stars) takes object height, shadow length and date/time and outputs a map of possible locations; inputs are user-supplied — [GitHub](https://github.com/bellingcat/ShadowFinder)
- A search result for "insurance claim photo sun position shadow verification" found only the Li et al. paper, SAE accident-reconstruction papers on sun models, and vendor blogs; the summary states "I found no insurer or regulator standard that requires sun-angle checks" — WebSearch result (summary, pages not all read)
- Claim-photo vendors checked, none documents sun/shadow-versus-claimed-time checks: Truepic confirms location and time (metadata-based) — [Truepic](https://www.truepic.com/blog/odometer-fraud); Bdeo lists five checks (VIN, GPS, plate, odometer, screen detection) — [Bdeo](https://bdeo.io/en/blog/detecting-auto-insurance-fraud-in-the-age-of-generative-ai/); Attestiv lists metadata review, duplicate/reuse, forensic signals and AI-content detection — [Attestiv](https://www.attestiv.com/); Carpe Data says "lighting and shadow checks test whether the altered region belongs in the original scene" (intra-image consistency) and mentions weather only in a figure caption — [Carpe](https://carpe.io/resources/blog/how-to-detect-ai-generated-photos-in-insurance-claims/)
- Capture-attestation neighbour: Succinct's zCam SDK embeds C2PA-based capture proofs (optional zero-knowledge proofs) in an insurer's own app and checks time and location against the claim; requires the insurer's app — [zCam](https://zcam-sdk.succinct.xyz/usecases/insurance_claim_verification)
- Milliman (3 Dec 2025): "Human detection is only 50% accurate" for AI-manipulated damage photos; "metadata analysis may help detect fake content but can be easily bypassed" — [Milliman](https://www.milliman.com/en/insight/accidents-that-never-happened-genai-motor-fraud)
- Open-Meteo: "The free API is for non-commercial use", limits 10,000 calls/day; commercial plans Standard 1M, Professional 5M, Enterprise 50M+ calls/month, prices not shown; historical, climate and ensemble APIs "require the Professional API Plan or higher" — [Open-Meteo pricing](https://open-meteo.com/en/pricing)
- Open-Meteo Historical Weather resolution: ERA5 0.25 degrees (about 25 km), ERA5-Land 0.1 degrees (about 11 km), ECMWF IFS 9 km; values are model-based and the page gives no accuracy figure for local rain — [Open-Meteo docs](https://open-meteo.com/en/docs/historical-weather-api)
- Free observed weather exists: the Iowa Mesonet archive lists Tashkent station UZTT with archive start 1931 and returns half-hourly METAR for 2026-10-08 (for example "UZTT 080000Z 08006KT CAVOK 12/06 Q1026 NOSIG"); about 1,474 observations for Oct 2025 (1,475 lines including the header) — [IEM Uzbekistan network](https://mesonet.agron.iastate.edu/geojson/network/UZ__ASOS.geojson); [IEM request API](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=UZTT&data=metar&year1=2026&month1=10&day1=8&year2=2026&month2=10&day2=11&tz=UTC&format=onlycomma)
- Sun geometry for the finals window [computed, NOAA-style solar position, Tashkent 41.31N 69.28E, 2026-10-11 UTC+5]: azimuth 214 deg at 14:00, 242 deg at 16:00, 263 deg at 18:00; elevation 35, 18.5 and -3 deg; azimuth rate 15 deg/h at 14:00 and 11 deg/h at 16:00; a 1.5 m car casts a 2.1 m shadow at 14:00, 4.5 m at 16:00, 10.5 m at 17:00
- Model licences: MoGe-2 MIT (HF tag; repo MIT with DINOv2 code Apache-2.0) — [HF vitl](https://huggingface.co/Ruicheng/moge-2-vitl-normal), [repo README](https://github.com/microsoft/MoGe); SigLIP 2 base Apache-2.0 — [HF](https://huggingface.co/google/siglip2-base-patch16-224); SAM 2.1 Apache-2.0 — [HF](https://huggingface.co/facebook/sam2.1-hiera-tiny)
- Context: Claude's docs say it cannot detect AI-generated images, and the open forensic detectors TruFor and B-Free are non-commercial [teammate] — [Anthropic vision docs](https://platform.claude.com/docs/en/build-with-claude/vision), [TruFor](https://github.com/grip-unina/TruFor)
- OSAGO context [teammate]: 2025 OSAGO payouts 245,224 mln UZS and voluntary motor payouts 107,758 mln UZS; 15 days for the insurer to decide — [NAPP Q4 2025](https://napp.uz/storage/files/shares/opendata/2025/4Q/%D0%A1%D1%82%D1%80%D0%B0%D1%85%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%28IV%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%B0%202025%20%D0%B3%D0%BE%D0%B4%20%29%20rus%20%E2%80%94%20filtr.pdf)

### Inferences
- With azimuth drifting 11-15 deg/h and a published 9.4 degree tolerance, time resolution is about one hour at best, and worse once camera yaw error is added; the check catches gross mismatches (wrong time of day, day versus night, fully synthetic scene), not "claimed 15:00, actually 14:00".
- Camera yaw is the hidden blocker: Telegram photos carry no heading (stated in the idea, not verified), phone compasses are distorted beside a steel body, and an OSM road bearing leaves a 180 degree front/rear ambiguity and fails in car parks. iPhone EXIF normally holds GPSImgDirection (my knowledge, unverified here), so gallery uploads could supply a heading where Telegram does not.
- It does not detect damage painted onto a real photo taken at the right place and time; the honest claim is "photo liveness" (taken now and here), which is coherent only when the trusted claim time is the server receive time.
- Weather evidence discriminates weakly in arid Tashkent (most days are dry and clear), so sun elevation, day/night and observed METAR (free, observed, half-hourly) are better legs than ERA5 wetness.
- Revenue pool is small: 245 bn + 108 bn UZS is about $29.9M of motor payouts at 11,800 UZS/USD; 2% leakage (assumption) is about $0.6M a year; $0.05 per check on about 275k claims (all-line count) is a $14k ceiling.

### Gaps
- No evidence of GenAI-staged or recycled-photo claims in Uzbekistan (only Russia [teammate]); no insurer interview.
- Web search for existing products was largely quota-blocked; absence of a shipping "shadow azimuth vs claimed time" product is only supported by the vendor pages above and one search summary.
- Kakar et al. paper unreadable (403); azimuth error numbers from it are unknown. Telegram EXIF stripping not verified.

## insurer-6 ChipCheck: bank-card-scaled windshield repair-or-replace triage

### Takeaway
Photo repair-versus-replace triage already runs at Belron brands (Autoglass UK since about 2017, Carglass Estonia with DriveX since Jan 2024); the bank-card scale is a small twist on the old coin/banknote rule of thumb, and there is no evidence of chip-repair demand or glass-cover volume in Uzbekistan. Scores: novelty 2 (famous), feasibility 3, demo 3, business 2, moat 2, local fit 3. Verdict: kill.

### Cited Findings
- DriveX AI at Carglass Estonia (Windrox, a Belron franchise), announced 8 Jan 2024: AI "suggests whether to repair or replace", automates 80% of self-service windshield claim reporting, "up to 36% of glass damage can be repaired"; page mentions no scale reference and no accuracy — [DriveX](https://drivex.io/?p=412)
- Autoglass UK AI app (Service Certainty "imageproof"): "over 80% accuracy", "up to 2,500 images a week"; no date in the text, byline "9 years ago" implies about 2017 (inference); no scale reference or photo protocol described — [Fleet World](https://fleetworld.co.uk/new-ai-app-can-determine-windscreen-repair-or-replacement-choices/)
- Autoglass's repair page gives no size or ADAS criteria and no photo tool; it routes to online booking — [Autoglass](https://www.autoglass.co.uk/windscreen-repair/)
- Bank cards use the ID-1 format, 85.60 x 53.98 mm — [Wikipedia ISO/IEC 7810](https://en.wikipedia.org/wiki/ISO/IEC_7810)
- A Wikipedia windshield article states repair of cracks up to 6.1 cm is "within permissible limits" (depends on local law) and lists over-rain-sensor and edge damage as problematic; it does not mention the driver's field of view or ADAS — [Wikipedia Windshield](https://en.wikipedia.org/wiki/Windshield)
- US App Store catalogue shows no windshield-chip triage app (queries "windshield chip repair", "windshield damage", "auto glass damage AI"); GitHub has only tiny crack-detection repos, e.g. [mdhakim2003 windshield crack](https://github.com/mdhakim2003/Automatic-Windshield-Crack-Detection-Using-Conventional-Image-Processing)
- Voluntary motor (KASKO-segment) context [teammate]: premiums 754,268 mln UZS (-47%), payouts 107,758 mln UZS, 660,550 contracts in force — [NAPP Q4 2025](https://napp.uz/storage/files/shares/opendata/2025/4Q/%D0%A1%D1%82%D1%80%D0%B0%D1%85%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%28IV%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%B0%202025%20%D0%B3%D0%BE%D0%B4%20%29%20rus%20%E2%80%94%20filtr.pdf)
- Model licences: RF-DETR Nano/Seg Apache-2.0 and EdgeTAM Apache-2.0 [teammate] — [RF-DETR](https://github.com/roboflow/rf-detr), [EdgeTAM HF](https://huggingface.co/facebook/EdgeTAM)

### Inferences
- At 600 px across a card the scale is about 0.14 mm/px [computed], so size error is sub-millimetre; the geometry is fine, the problem is novelty and demand.
- Glass-cover savings pool is tiny: even if glass were 5% of 107.8 bn UZS payouts (assumption) and repair saved a third, the whole market saves about $0.15M a year.
- Demo is contrived (printed chips); a cracked screen protector is not a windshield.

### Gaps
- No chip-repair price, volume or shop count for Uzbekistan; no verified public chip dataset or licence; per-model ADAS-camera zone data not found.
- Web search for similar apps was quota-blocked, so a bank-card/coin-scaled chip app could exist unseen.

## insurer-12 VoltTow: EV-aware tow dispatch from one scene photo

### Takeaway
The stated gap ("OEM first-responder guides are PDFs", "dispatch on driver-typed details") is partly outdated: an app aggregating EV towing guidance exists and dispatch platforms already advertise AI. The photo-to-equipment step is the only new piece, and it automates a one-line rule (EV/AWD means flatbed) for a tiny event volume. Scores: novelty 3, feasibility 2, demo 3, business 1, moat 2, local fit 3. Verdict: kill.

### Cited Findings
- EV Clever Rescue And Recovery (American Towing & Recovery Institute, released 17 Jul 2024, updated 30 Jul 2025): per make and model Emergency Response Guides with towing, lifting, securing and storing suggestions, battery location and fuel-type identification — [App Store](https://apps.apple.com/us/app/ev-clever-rescue-and-recovery/id6504069271)
- Urgently (roadside dispatch platform): "AI Job Insights" anomaly detection, AI communication; the page mentions no photo submission and no EV-specific towing — [Urgently](https://www.geturgently.com/)
- HONK app: tow and roadside on demand with "over 75,000 trucks" network — [App Store](https://apps.apple.com/us/app/honk-tow-roadside-ondemand/id911206852)
- moditech.com now redirects to Bliksund, whose page lists "Rescue Sheet & ERG development" and a "CRS (Crash Recovery System)" solution; feature details are not shown — [Bliksund](https://bliksund.com/)
- Uzbekistan had 122.7 thousand registered EVs on 1 Jul 2026, 88,496 (about 72%) in Tashkent city; published 21 Sep 2026 — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4)
- H1 2026 EV imports 37,874 units, 99.2% from China [teammate] — [Spot](https://www.spot.uz/ru/2026/08/14/ev-import-rise/)
- Model licences: SigLIP 2 Apache-2.0, MoGe-2 MIT (see Alibi); SAM 3 under the SAM License (commercial with conditions, gated) [teammate] — [SAM 3 licence](https://github.com/facebookresearch/sam3/blob/main/LICENSE)

### Inferences
- A caller or dispatcher can state "BYD Song Plus electric"; make/model lookup plus the OEM guidance already exists, so vision adds little over a typed model or a plate lookup.
- Single-photo cues (orange HV cable, smoke) are rarely visible on a stranded car; PHEV and BEV bodies share shells, so identification needs close-up badge photos, defeating "one scene photo".
- Volume is tiny: if 3% of 122.7k EVs need a tow in a year (assumption) that is about 3.7k events; at $0.30 that is about $1.1k revenue.

### Gaps
- No Uzbek tow-dispatch or roadside-assistance market data; EV breakdown rates unknown.
- Moditech/CRS feature details unread; web search for photo-based tow selection was quota-blocked.

## service-1 Drip-ID: puddle photo to fault and booking

### Takeaway
CV classification of fluid leaks by colour, sheen and spread already appears in fleet maintenance software and in a GM patent, but no consumer or shop-intake drip-card product was found in the app, GitHub, HF or arXiv catalogues. It is the most buildable and most demoable idea in this batch. Weak points: payer, moat, domain gap between fresh and aged fluids. Scores: novelty 3, feasibility 4, demo 5, business 2, moat 2, local fit 4. Verdict: sharpen (reposition to the used-car drip test).

### Cited Findings
- HVI (JRS Innovation Inc.; Kal Tire and T&T Construction case studies listed) says its phone-photo AI detects "oil saturation, coolant residue, power-steering seepage, and differential traces" and "uneven wear patterns" with wear direction pointing to alignment issues; "95-99%" accuracy is unsourced marketing — [HVI](https://heavyvehicleinspection.com/blog/post/ai-vehicle-analysis-fleet-management-photo-inspections)
- OxMaint (marketing post dated 1 Aug 2026): classifies leaks from standard phone photos by "color, sheen, and spread pattern", fluids coolant, engine oil, hydraulic, fuel, transmission; about 4 s per classification; no accuracy percentage or pricing — [OxMaint](https://oxmaint.com/industries/fleet-management/ai-visual-fluid-leak-detection-for-fleets)
- GM patent US11954913B2 (filed 1 Aug 2022, granted 9 Apr 2024): onboard cameras image the parking spot before parking and after leaving, compare, and classify a leak by fluid colour and location — [Google Patents](https://patents.google.com/patent/US11954913)
- A WebSearch for a consumer leak-from-photo app found fleet tools and patents but "didn't find a consumer app to recommend"; the summary also notes colour alone is a weak guide, several fluids look alike and A/C water is clear and harmless — WebSearch result for "AI identify car fluid leak from photo color puddle app"
- Catalogue checks: no leak-identifier app in the US App Store (queries "car fluid leak identifier", "oil leak under car", "coolant leak"); GitHub "oil leak"/"car leak" return unrelated repos; HF "oil leak" returns two oil-and-gas datasets; arXiv "fluid leak" returns physics papers
- Pre-purchase inspection in Tashkent from 300,000 UZS (OLX listing, 7 Oct 2026): scanner mileage, paint condition, interior, lights, endoscope, compression, chassis, VIN, brake fluid, airbags; no leak or drip test is listed — [OLX.uz](https://www.olx.uz/d/oz/obyavlenie/ot-300-000-avtopodbor-proverka-avto-probega-pered-pokupkoj-avtotekshi-ID3XhPY.html)
- Parc: 4,903.1 thousand individually owned vehicles on 1 Apr 2026 (92.8% passenger cars), +344.6 thousand year on year — [Statistics Committee](https://stat.uz/ru/press-tsentr/novosti-goskomstata/68807-zbekistonda-a-oliga-tegishli-avtotransport-vositalari-soni-4-9-mlntadan-oshdi-3)
- Used-car volume 52-57k monthly transactions in autumn 2025 [teammate] — [Kun.uz](https://kun.uz/en/news/2025/11/18/electric-car-purchases-surge-as-overall-auto-sales-recover-in-uzbekistan)
- DINOv3 uses a gated "DINO License" (commercial use allowed with conditions); SigLIP 2 is Apache-2.0 [teammate] — [DINOv3 licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

### Inferences
- Colour confusion is structural: ATF, power-steering fluid, coolant and brake fluid overlap, and self-poured fresh fluids on cardboard differ from aged real leaks; report a confusion matrix and use persistence cues (two photos 10 minutes apart separate evaporating A/C water from oil; wicking halo on paper).
- Used-car buyers and avtopodbor inspectors are a better payer than service-centre leads: a leak/drip test is a recognised pre-purchase check, absent from the sampled inspection scope, and printed calibrated cards are a consumable.
- Revenue stays small: 1% adoption of about 650k used deals a year at $1.5 per report is about $10k.
- Generic chat VLMs can already describe a stain; the defence is the calibrated card plus persistence test plus shop-labelled outcomes.

### Gaps
- No Google Play or Russian/Uzbek app checks (RU storefront returned nothing); web search mostly blocked.
- No data on willingness to pay or on how many Sergeli-market buyers would run a drip test.

## service-2 Smoke-Read: exhaust video to engine fault lead

### Takeaway
The data plan fails and the demo is impractical: the only public smoky-vehicle dataset covers black smoke with bounding boxes from CCTV and is not downloadable from its repository, no labels link smoke appearance to confirmed faults, and judges cannot test their own cars. The colour-to-fault mapping is a static chart mechanics already know. Scores: novelty 4, feasibility 2, demo 2, business 2, moat 2, local fit 3. Verdict: kill.

### Cited Findings
- LaSSoV / CoDeS paper: 75,000 smoky-vehicle images annotated with bounding boxes, plus 163 one-minute videos with segment annotations, collected from surveillance cameras (1920x1080); focuses on "visible black smoke"; "The code and dataset will be released"; false positives from shadows and wet roads — [arXiv 2207.03708 abstract](https://arxiv.org/abs/2207.03708); PDF text read locally from [arXiv PDF](https://arxiv.org/pdf/2207.03708)
- The named repository was created and last pushed on 2022-06-13, holds only a LICENSE (Apache-2.0) and a README containing the single line "# smokyvehicle"; 0 stars, 1 fork; no data — [GitHub repo](https://github.com/pengxj/smokyvehicle), [GitHub API](https://api.github.com/repos/pengxj/smokyvehicle)
- Exhaust-smoke repos are marginal: MATLAB smoke-and-plate project from 2019 [GitHub](https://github.com/srinathkr07/Exhaust-Smoke-Detection-and-Vehicle-Tracking-System); EPA-style opacity algorithms [GitHub](https://github.com/oscardpedrayes/SmokeOpacity-Alternatives)
- US App Store: only a Penn State visible-emissions training app (PSFEI Smoke School Test) for queries "exhaust smoke diagnosis" and "tailpipe smoke"; no workshop smoke-colour diagnosis app
- Decree UP-229 of 24 Nov 2025 calls for "control over vehicle emissions", tightened oversight of construction-material transport and up to 1,000 new monitoring stations; the article names no penalties — [gov.uz](https://gov.uz/en/miit/news/view/105885)

### Inferences
- Pretraining "plume masks" on LaSSoV is not possible (boxes only, black smoke only, not released); blue and white smoke classes have no public data.
- Labels from shop teardown outcomes cannot be collected in 2-3 days; composites of synthetic plumes teach segmentation, not diagnosis.
- A diesel or smoky petrol car will rarely be present at the venue, so the demo is pre-recorded clips.
- A fleet or inspectorate "black-smoke gate" fits the decree theme, but no penalty schedule or buyer was found.

### Gaps
- Whether LaSSoV was released elsewhere (for example with the 2023 journal version; DOAJ page returned 403) is unknown.
- Teammate note tech_radar.md (local file, no URL) also reports no verifiable public dataset for smoky/exhaust vehicles.
- Web search for smoke-diagnosis apps was quota-blocked.

## service-3 PlugLens: spark-plug, dipstick and fluid reader

### Takeaway
Plug-condition classification exists as a small public dataset and an Edge Impulse demo, and a chart-based plug-reading app launched in Sept 2026; the idea bundles four pipelines (plug, dipstick, brake fluid, coolant) for a payer with weak willingness to pay. The only differentiated element is CNG/LPG-specific wear evidence, which needs data the team would have to collect. Scores: novelty 3, feasibility 3, demo 3, business 2, moat 2, local fit 4. Verdict: sharpen (single wedge).

### Cited Findings
- HF dataset "Spark Plug Condition Classification": four classes (normal, carbon_fouled, oil_fouled, mechanical_damage), under 1K images, 96x96 input, BSD-3-Clause, MobileNetV2 transfer learning on Edge Impulse (updated 2026-06-24) — [HF dataset](https://huggingface.co/datasets/eoinedge/spark-plug-classification)
- "Spark Plug Reader" iOS app (released 16 Sep 2026): eight plug appearances "drawn as they come out of an engine", tap-to-match diagnosis plus torque table, works offline; no camera analysis described — [App Store](https://apps.apple.com/us/app/spark-plug-reader/id6805215335)
- arXiv spark-plug fault work uses accelerometer and acoustic signals, not images — [arXiv 2311.02282](https://arxiv.org/abs/2311.02282)
- NGK's USA site did not surface a reading guide or LPG/CNG guidance in the page fetched — [NGK](https://www.ngksparkplugs.com/)
- Fuel mix: live Avtoelon counters on 2026-10-10: petrol 16,854; petrol-gas 9,965; gas 448; hybrid 1,166; electric 827; diesel 142 (sum 29,402, so petrol-gas is 33.9% [computed]); Chevrolet 19,942 — [avtoelon.uz](https://avtoelon.uz/)
- About 11% of cars run on LPG [teammate] — [Anhor](https://anhor.uz/news/price-of/)
- Pre-purchase inspection from 300,000 UZS lists endoscope and compression checks but no plug reading — [OLX.uz](https://www.olx.uz/d/oz/obyavlenie/ot-300-000-avtopodbor-proverka-avto-probega-pered-pokupkoj-avtotekshi-ID3XhPY.html)

### Inferences
- Electrode gap is about 1 mm; measuring it from a phone macro against a 14 mm thread needs roughly 17 micrometre/pixel imaging [computed for 800 px across the thread], plausible only with close focus and calibration, and a feeler gauge already does this in seconds.
- The valuable output is evidence for the customer (photo job card), not diagnosis; CNG-specific thresholds cannot be validated without mileage and fuel-type labels.
- Scope creep (dipstick, brake, coolant) multiplies pipelines for no extra payer.

### Gaps
- No source for CNG/LPG plug-interval claims; no payer pricing evidence for independent mechanics; web search for plug-reading apps was quota-blocked.

## service-4 WearMap: tyre wear pattern to alignment fault

### Takeaway
The headline claim is already made by at least five products (HVI and four indie AI tyre apps launched 2025-26), plus Anyline for tyre-retail pros, a phone-based alignment tool (Gyraline) and an academic capstone. The remaining wedge is B2B fleet time-series, not one-shot diagnosis. Scores: novelty 2, feasibility 3, demo 3, business 3, moat 2, local fit 3. Verdict: sharpen (fleet wedge), low priority.

### Cited Findings
- TireOptic (released 31 Aug 2026): tread depth in mm on all four tires and "Uneven wear across the tread surface, so you can catch alignment and pressure problems early" — [App Store](https://apps.apple.com/us/app/tireoptic/id6778348575)
- Tire Check & Tread Scan (released 13 Jul 2026): "Detection of uneven wear patterns that may point to alignment or suspension concerns"; copy mentions Nigerian highways and ride-hailing fleets — [App Store](https://apps.apple.com/us/app/tire-check-tread-scan/id6788365448)
- Tire Check: Condition & Safety (released 13 Mar 2025): report sections include "wear pattern"; TireScan: AI Tread & Wear Check (released 11 Jun 2026) says "spot uneven or dangerous wear" — [App Store](https://apps.apple.com/us/app/tire-check-condition-safety/id6743192000), [App Store](https://apps.apple.com/us/app/tirescan-ai-tread-wear-check/id6776771818)
- HVI: "uneven wear patterns" with wear direction that "may point to alignment issues" — [HVI](https://heavyvehicleinspection.com/blog/post/ai-vehicle-analysis-fleet-management-photo-inspections)
- Anyline: "Inspect tire tread depth and wear with just a mobile phone"; customers named include Michelin, Discount Tire, Bridgestone, Continental; page does not mention wear-pattern or alignment diagnosis and gives no accuracy — [Anyline](https://anyline.com/products/tire-tread)
- Gyraline: phone plus toolkit measures total toe, thrust line, camber, caster — [App Store](https://apps.apple.com/us/app/gyraline/id6450752768)
- Academic: VIT-AP capstone "Vision-Based Detailed Tyre-Wear Recognition and Single-Wheel Alignment Screening" (Sept 2026) with HRNet width error 1.312% on two test tyres — [GitHub](https://github.com/Shanmuk4622/tyre-wear-alignment-cv); small open repos for tread depth, e.g. [pouryahoseini/Monocular-Tread-Depth](https://github.com/pouryahoseini/Monocular-Tread-Depth)
- Data: HF `NMiriams/Good_Tires` (CC-BY-4.0, under 1K images) was the only tyre set found by teammates [teammate] — [HF](https://huggingface.co/datasets/NMiriams/Good_Tires); `Shanmuk4622/tyre-wear-study` exists with licence tag absent — [HF](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study)
- Depth models for a commercial demo: Depth Anything V3 Small/Base and V2-Small are Apache-2.0 [teammate] — [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3)
- Taxi context [teammate]: Yandex Go held 86.3% of the taxi-aggregator market (Dec 2023); 361k drivers worked via aggregators in Aug 2024 (secondary thesis) — [Anhor](https://anhor.uz/news/yandex-committee/), [KSE thesis](https://kse.ua/wp-content/uploads/2025/09/Oleh_Dykyi_RIDE-HAILING-MARKET-IN-UZBEKISTAN.pdf)

### Inferences
- Sub-millimetre tread-height profiles from monocular metric depth scaled by a coin are not credible (my judgement; no source); a zone-wise (inner/centre/outer) rule layer validated against a depth gauge is the honest design.
- One-shot consumer scans are commoditising; per-vehicle history across months for taxi parks (and measured tyre-spend savings) is the defensible asset.
- Demo on a judge's own car tyre is possible outdoors, but a single tyre cannot show four-wheel cross-reasoning.

### Gaps
- Anyline's pro workflow may include wear-pattern flags not on the page read; web search for pattern-to-alignment products was quota-blocked.
- No local alignment price or taxi-park tyre-cost data.

## service-5 BeamAim: headlight aim and lens haze check

### Takeaway
Professional camera-based cut-off testers exist (quote-priced) and no phone-only aim app was found; the physical problem is real (a TU Darmstadt study found 75% of examined cars mis-aimed) but a phone cannot adjust aim precisely, iOS browsers cannot lock exposure, the demo needs a dark wall and a car, and the payer pays cents. The haze-grading half is the more sellable and demoable. Scores: novelty 4, feasibility 3, demo 2, business 2, moat 2, local fit 2. Verdict: sharpen (screening plus haze, not adjusting).

### Cited Findings
- Corghi PFT CAM: camera-based "automatic detection of the projector's boundary line using geometric and photometric analysis", laser alignment aid, touchscreen, halogen/xenon/LED compatible, "Get a Quote" pricing — [Corghi](https://www.corghi.com/en/product/pft-cam/)
- A WebSearch for a phone headlight-aim app found no ready-made app, only professional equipment, a build-your-own tool page and patents (for example US4730923, US8605269) — WebSearch result for "phone camera headlight aim alignment check app"; [RationalGo builder](https://rationalgo.ai/resources/app-builder/automotive-headlight-aim-adjustment-verification-tool)
- TU Darmstadt whitepaper (3 Apr 2025): "A study conducted by Technical University Darmstadt revealed that 75% of the vehicles examined had incorrectly adjusted headlights"; digital aiming devices reach 0.1-0.2 degrees; text read locally from the PDF — [TU Darmstadt](https://www.lichttechnik.tu-darmstadt.de/media/fachgebiet_lichttechnik/forschung_7/re_aiming_jl/Re-Aiming_Whitepaper_JL_MH_eng.pdf)
- caniuse data: the ImageCapture API is unsupported in every iOS Safari version listed (3.2 through 27.2) and supported in Chrome for Android — [caniuse data](https://raw.githubusercontent.com/Fyrd/caniuse/main/features-json/imagecapture.json)
- WebGPU has about 86% global full support and iOS Safari 26+ [teammate] — [caniuse WebGPU](https://caniuse.com/webgpu)
- US App Store queries "headlight aim alignment", "headlight beam", "headlight restoration" return no aim or haze app; GitHub "headlight"/"headlamp" return unrelated repos; arXiv "headlight aim" returns unrelated papers
- Technical inspection in Uzbekistan is run by private operators [teammate gap; the Anhor URL slug says private operators from 1 Jan 2018, the teammate wrote 2019, so the year is unresolved] — [Anhor](https://anhor.uz/society/tehosmotr-avto-s-1-yanvarya-2018-goda-budut-provodity-chastniki)

### Inferences
- Error budget [computed]: 0.5 degree phone pitch error is 0.87% slope, 4.4 cm at 5 m, against a 1.0-1.5% (5-8 cm) tolerance band, so a phone IMU supports coarse "high/OK/low" screening, not adjuster-turn guidance.
- iOS browsers cannot set exposure through ImageCapture, so a beam photo may saturate; native camera capture or Android live capture is needed.
- Judges at the venue would need a dark space, a wall and a car with working low beams; a torch-and-cut-off-mask substitute is not the product.
- Whether Uzbek inspection lanes already own optical headlamp testers is unknown and decides the payer.

### Gaps
- No Uzbek data on headlight misaim, glare complaints or inspection-lane equipment; web search beyond the one query was quota-blocked.

## service-6 SwirlMap: paint swirl density and before/after proof

### Takeaway
No swirl-quantification product was found, but there is also no buyer pull: detailers already prove results with inspection lamps and before/after photos, capture needs controlled light, and the local detailing market is unevidenced. Scores: novelty 3, feasibility 3, demo 3, business 1, moat 1, local fit 2. Verdict: kill.

### Cited Findings
- Nearest adjacent products: NexPTG paint thickness gauge with a phone app, 100+ car body templates, reports (repaint detection, not swirls) — [App Store](https://apps.apple.com/us/app/nexptg/id1316943346); DAMAGE iD before/after video comparison for rentals — [App Store](https://apps.apple.com/us/app/damage-id/id1024284443)
- OEM-style paint-defect detection repos exist (YOLO on defects), e.g. [Hacksutates/paint-defect-detection](https://github.com/Hacksutates/paint-defect-detection), [Jplusgogo/car_paint_defects_test_dataset](https://github.com/Jplusgogo/car_paint_defects_test_dataset); HF `imaadd05/paint-defects-roboflow-main` is CC-BY-4.0 — [HF](https://huggingface.co/datasets/imaadd05/paint-defects-roboflow-main)
- Catalogue queries "car swirl marks detailing", "car swirl scratch detection", "paint correction inspection" return detailing-service and game apps; arXiv "swirl paint" returns nothing
- No Uzbek detailing market size found (see Gaps)

### Inferences
- A numeric swirl index adds little to a customer who judges by shine; sales proof already works with a swirl-finder lamp and photos.
- Standardising a single-light capture (angle, distance, no direct sun) is the hard part, and the product must refuse in sun.

### Gaps
- Web search for swirl AI and detailer demand was quota-blocked; no local studio counts or prices.

## service-7 HueMatch: phone photo to paint code and fade

### Takeaway
Physically implausible for the stated payer: refinish colour matching tolerates a few tenths of a delta E, a phone plus printed card cannot get there for metallic or pearl finishes, the paint code is printed on the car, and pros own spectrophotometers; handheld colour readers also exist for the DIY end. Scores: novelty 3, feasibility 3, demo 4, business 2, moat 2, local fit 2. Verdict: kill.

### Cited Findings
- Datacolor ColorReader: "matches paint color with more than 90% accuracy" against fan decks, Bluetooth device plus app (architectural focus) — [App Store](https://apps.apple.com/us/app/datacolor-colorreader/id1217566816)
- Color Muse (Sherwin-Williams custom colour matching; measures colour and sheen from smooth, non-metallic surfaces) — [App Store](https://apps.apple.com/us/app/color-muse/id995252259)
- X-Rite MA-T12: 12 measurement angles, automotive among listed industries, "Request Pricing" — [X-Rite](https://www.xrite.com/categories/portable-spectrophotometers/ma-t12)
- Nix colour sensors (Mini 3, Spectro L, Spectro 2): the Mini 3 is described as matching to "200,000+ Paints"; page mentions neither automotive nor metallic finishes and shows no prices — [Nix](https://www.nixsensor.com/)
- Colour science: CIELAB Delta E*ab of about 2.3 corresponds to a just-noticeable difference; for automotive, Delta E CMC is "often less than 0.5 under D65/10" (article carries a maintenance banner on acceptability values) — [Wikipedia Color difference](https://en.wikipedia.org/wiki/Color_difference)
- No phone-plus-card automotive refinish matcher found in the US App Store (queries "car paint color match", "car paint code finder", "automotive paint matching") — catalogue check
- Used-car inspection scope in Tashkent lists paint condition only; no colour matching — [OLX.uz](https://www.olx.uz/d/oz/obyavlenie/ot-300-000-avtopodbor-proverka-avto-probega-pered-pokupkoj-avtotekshi-ID3XhPY.html)

### Inferences
- Phone camera plus gray card typically cannot reach the sub-1 delta E needed to blend repairs (my judgement; the cited limits show the gap), so output would be a shortlist for DIY touch-up, a market that already uses label paint codes.
- Digitising local Chevrolet/BYD fandecks and mix recipes depends on proprietary paint-shop data.

### Gaps
- Web search for automotive-specific phone matchers was quota-blocked; no local body-shop equipment survey.
