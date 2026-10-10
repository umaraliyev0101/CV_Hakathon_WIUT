# Verification batch 08: ten fleet-logistics and driver-safety CV ideas (Tashkent hackathon, due diligence)

Method note (not a findings section). Date 2026-10-10. Scores are skeptical defaults on a 1-5 scale. Currency: about 11,800 UZS per USD (teammate fact sheet). BRV is 440,000 UZS from 1 Sep 2026 (Oson Prava page lists fines at that value).

Search budget: one standard WebSearch was run per idea (10). The shared quota (200 calls across all agents) then ran out. One later Russian query succeeded; every other follow-up search was refused, so most second-pass checks (Uzbek/Russian local evidence, academic prior art, hardware alternatives) were NOT run. Remaining evidence comes from direct WebFetch of URLs, plus teammate notes in this folder (uzbekistan_market.md, tech_radar.md, hackathon_intel.md) and local pdftotext on two fetched PDFs.
Tags: (S) = seen only in a search-result snippet, page not read or fetch failed; "teammate" = taken from a sibling notes file, not re-verified by me; "my arithmetic" = derived from cited inputs.

Searches that ran: "EV charger billed kWh vs vehicle energy discrepancy audit"; "headlight brightness dip estimate battery voltage smartphone camera"; "smartphone flash retroreflectivity measurement app"; "AI contest traffic camera fine appeal evidence analysis app"; "blocked-in cars parking lot retrieval planning computer vision"; "AI photo cargo volume estimate which truck size delivery app"; "flood depth estimation from photo vehicle can I drive through"; "driver fitness for duty smartphone test fatigue alertness pre-shift"; "child left behind school bus camera AI sweep detection retrofit"; "brake light check app one person phone camera vehicle lights"; "EV fleet drivers charging personal car on company charge card fraud"; Russian "дистанционный предрейсовый медосмотр водителей смартфон камера пульс".
Searches refused by the quota (17): Russian EV-charger metrology and overbilling-complaint queries; BM2 battery monitor; halogen flux exponent; starter-sound diagnosis; smartphone retroreflectivity (two variants); conspicuity-tape inspection; camera-fine error and appeal in Uzbekistan; Tashkent flood streets; child forgotten in school bus in Uzbekistan; AI walkaround brake-light detection; dealership lot blocked cars; flood depth from submerged vehicle wheels; Uzbek pre-trip medical exam and e-waybill; Russian "pick a truck for cargo by photo".
Fetches that failed: fleetnews.co.uk (403), netfm.net (404), whichcar.com.au (403), technical.ly (403), Continental press release (404), ccjdigital.com (403), patents.justia.com (403 x2), apps.apple.com id913756562 (403), Vayyar (403), Penticton Herald (429), support.teslafi.com (DNS), sviro.kl.dfki.de (certificate error), Wikipedia "Cold_cranking_amps" (404, answered via Automotive_battery instead).

## evcng-9: Hakam, charger-meter and fleet-charging referee

### Takeaway
The fleet-card half is already sold with telematics by WEX and Lightfoot/Mina; the charger-meter half has no photo-OCR product, but SoC x capacity is too noisy to catch typical charger errors in few sessions. Verdict sharpen. Scores: novelty 3, feasibility 3, demo 3, business 2, moat 2, local fit 4.

### Cited Findings
- WEX "EV At-Home with Vehicle Fraud Protection" compares vehicle, charger and utility data with submitted home-charging sessions to catch personal or unauthorized charging before reimbursement; posted 17 Apr 2026, no numbers — [ChargedEVs](https://chargedevs.com/?p=615842)
- Mina + Lightfoot (13 Sep 2022): Lightfoot telematics record battery charge level before and after, charge time, kWh rate and location, so fleets pay only for energy actually supplied; no fraud-prevalence figure — [Automotive Fleet](https://www.automotive-fleet.com/10181054/new-partnership-eliminates-ev-charge-fraud-1)
- Vendors (Motive, Rightcharge, Deftpower, Invers) describe personal-vehicle charging and cloned or shared cards as the vectors; no independent prevalence data (S) — [Fleet News](https://www.fleetnews.co.uk/news/right-fuel-card-agree-deal-with-rightcharge-to-help-fuel-mixed-fleets); [Deftpower](https://www.deftpower.com/resources/white-papers/ev-charging-fraud-prevention)
- EVCI Global (independent inspector, first reported by The Telegraph): 31.5% of reviewed chargers over- or under-estimated energy; about 15% of cases off by more than 5%; one charger delivered 37% less than its screen; sample size and method not given; UK public meters mostly allowed up to +2% — [Carscoops](https://www.carscoops.com/2026/03/ev-charger-accuracy-study/)
- A charging network says up to 20% loss on a Level 2 charger is normal and that chargers bill energy delivered to the vehicle (vendor claim) — [Evercharge](https://support.evercharge.com/kb/article/34-why-is-the-amount-of-energy-on-my-bill-higher-than-on-my-car/)
- Owners already compare car-side and charger-side kWh informally: a TeslaFi user reports charger kWh about 10% below TeslaFi's computed figure (forum claim, page not fetched) (S) — [TeslaFi thread](https://support.teslafi.com/communities/1/topics/629-different-kwh-readings-public-charger-vs-charge-energy-added)
- Municipal case: a Penticton councillor noted nearly half of an EV-charging power bill was not delivered to vehicles; idle draw and a software issue were cited (S, fetch 429) — [Penticton Herald](https://www.pentictonherald.ca/spare_news/article_c03e9438-4867-5f7b-b03d-4215819c16d4.html)
- Uzbekistan draft resolution (comments to 17 May 2026): mandatory metrological inspection of all charging equipment, technical audit every two years, uncertified chargers banned, target 4,000 stations by end-2026; EV fleet above 100,000 at end-2025 — [Kun.uz, 5 May 2026](https://kun.uz/en/news/2026/05/05/uzbekistan-to-expand-ev-charging-network-to-4000-stations-by-end-of-2026)
- 122.7k registered EVs on 1 Jul 2026, 72% in Tashkent city (teammate) — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4)
- PP-OCRv6 (Apache-2.0, 2026-06; Cyrillic/Uzbek coverage not verified) and SigLIP 2 (Apache-2.0) exist (teammate tech radar) — [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224)

### Inferences
- Sessions needed to resolve a bias delta with per-session noise sigma: n = (1.96 sigma / delta)^2 for a 95% interval, about 2x that for 80% power. sigma 12%: delta 5% needs about 22 to 45 sessions, delta 10% about 6 to 11. With sigma about 3% (vehicle-reported kWh instead of SoC x capacity) delta 5% needs about 2 to 3. EVCI saw only about 15% of units beyond 5%, so charger ranking from SoC photos catches only gross outliers.
- Fleet-card misuse (a different car charged) is a large single-session signal and is the stronger half, but telematics vendors cover it in the West; photo-OCR is a workaround where no telematics API exists.
- Official mandatory inspections reduce the chance a regulator pays for crowdsourced estimates.

### Gaps
- No Uzbek evidence of overbilling complaints, fleet charging cards at EV taxi parks, or willingness to pay (local queries blocked).
- "Quvvat" logged sessions are a team asset I cannot verify.
- Whether BYD/Chery/Haval screens or apps show per-session kWh added was not checked.

## wild-3: No-Start Doctor, cranking video turns camera into voltmeter

### Takeaway
The physics is real (filament light output goes as V^3.4) and no camera-voltmeter product turned up, but a $10 OBD dongle or terminal monitor already gives true volts, LED and load-shedding cars break the principle, and no payer is evidenced. Verdict sharpen. Scores: novelty 4, feasibility 3, demo 3, business 2, moat 2, local fit 4.

### Cited Findings
- "Light output is approximately proportional to V^3.4", lifetime to V^-16, power to V^1.6; a 5% voltage reduction cuts light output about 16% — [Wikipedia: Incandescent light bulb](https://en.wikipedia.org/wiki/Incandescent_light_bulb)
- CCA rating = current a battery can deliver at -18 C for 30 s while holding at least 7.2 V (12 V battery) — [Wikipedia: Automotive battery](https://en.wikipedia.org/wiki/Automotive_battery)
- OBD-II service 01 PID 0x42 gives control-module voltage, V = (256A+B)/1000; PID 0x0C gives engine speed — [Wikipedia: OBD-II PIDs](https://en.wikipedia.org/wiki/OBD-II_PIDs)
- Search for brightness-to-voltage methods returned only forum threads on headlight voltage drop (one notes LED headlights may be PWM-driven so a multimeter misleads) and an unrelated Samsung patent on display brightness vs battery gauge — [club8090 thread](https://club8090.co.uk/forum/viewtopic.php?p=8051331); [US8599205](https://patents.google.com/patent/US8599205)
- Chinese brands took 17.8% of Uzbek sales Jan-Jul 2026 vs 11.6% a year earlier (teammate) — [Anhor](https://anhor.uz/ru/news/avtomobili-kitajskie-brendy-uzbekistan)

### Inferences
- Using the ratio to pre-crank brightness removes absolute calibration: V_crank = V_idle x ratio^(1/3.4). At 9.6 V the lamp is about 40% of 12.6 V output; at the 7.2 V CCA floor about 15% (my arithmetic), a large swing.
- Failure modes: auto-exposure (browser exposure lock is unreliable on iOS Safari; general knowledge, not verified here), LED headlamps (current-regulated, not V^3.4), body modules that drop lamp load during cranking.
- OBD dongle logging on 3-5 cars is a cheap honest ground truth.

### Gaps
- No data on no-start frequency, roadside-assistance or battery-swap market in Uzbekistan.
- Halogen vs LED share of the Uzbek parc and whether Cobalt/Nexia/Damas cut lamps during cranking not verified.
- Bluetooth terminal battery monitors (BM2-type) not verified (query blocked).

## wild-14: Be-Seen, flash-photo audit of trailer and moped conspicuity

### Takeaway
No phone-flash retroreflectivity tool found (only dedicated hardware with a phone interface), the method is physically sound and very demoable, and moped reflector violations are an active local problem; payer and moat are thin. Verdict keep. Scores: novelty 4, feasibility 4, demo 4, business 2, moat 2, local fit 4.

### Cited Findings
- Easylux MINI is a handheld retroreflectometer; the phone app (EASYDATA) controls it, stores results and adds photos, while measurement is done by the instrument's external light beam — [Easylux](https://easylux.com.br/info/small-light-retroreflectometers-integrated-into-your-smartphone-can-revolutionize-road-marking-inspection-work/)
- GitHub topic "retroreflectivity" has zero repositories — [GitHub topic](https://github.com/topics/retroreflectivity)
- Yandex Uzbekistan: 23,785 violations by motorcycle, scooter and moped riders in 2026; only about 8% (1,890) were document violations; 92% concerned missing reflectors or headlight, window tint and passengers; couriers fined almost 4.7 bn soum; raids increasing — [Spot, 6 Oct 2026](https://spot.uz/ru/2026/10/06/couriers)
- SBDD: 563 crashes involving mopeds and scooters in nine months of 2026, 89 killed, 488 injured; mopeds registered with plates since 1 Oct 2025; A1 licence for small mopeds — [Spot, 6 Oct 2026](https://spot.uz/ru/2026/10/06/couriers)
- Fines for scooter/moped riders from 28 Oct 2024 include 1/2 BRV for no reflectors or lights (teammate) — [BES.media](https://bes.media/amp/v-uzbekistane-vvodyat-shtrafi-za-narushenie-pdd-voditelyami-mopedov-i-elektrosamokatov-33fa43/)
- Standard Uzbek plates have a white reflective background, 520 x 110 mm (Wikipedia page lacks citations) — [Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan)
- SAM 3 exists (arXiv 2511.16719) but HF weights are gated with manual approval (teammate) — [arXiv](https://arxiv.org/abs/2511.16719); [SAM 3 repo](https://github.com/facebookresearch/sam3)

### Inferences
- A phone flash sits about 1 cm from the lens, so at 2-5 m the observation angle is well under 1 degree, where retroreflectors are brightest; a flash-minus-ambient difference image isolates them and per-pixel brightness scales about 1/d^2 (my reasoning, not sourced).
- Thresholding the difference image may make SAM 3 unnecessary and avoids the gated-weights delay.
- Score only relative classes (certified tape vs white tape vs paint), not cd/lx/m2: no reference instrument, flash power differs per phone.
- Yandex's 92% lumps reflectors with tint and passengers, so the reflector share alone is smaller.

### Gaps
- Academic smartphone or night-image retroreflectivity work probably exists but could not be searched (quota).
- Whether Uzbek rules require ECE R104-style tape on trucks or trailers was not found.
- No courier-moped headcount; iOS Safari cannot toggle the torch from a web page (general knowledge), so use native-camera pair upload on iPhones.

## matrix-1: Shtraf Shield, camera-fine evidence auditor for fleets

### Takeaway
No vision-based evidence auditor for fleets found; RicorsoAI (Italy) and ticket-appeal bots address paperwork, and in Uzbekistan the owner-to-driver transfer, early-payment discount and fine sizes leave a thin payback for a vision audit. Verdict sharpen. Scores: novelty 3, feasibility 4, demo 4, business 2, moat 2, local fit 4.

### Cited Findings
- RicorsoAI: upload a fine photo or PDF, extracts data, flags contestable defects (expired calibration, late notification, missing signage), drafts the appeal with a success probability for EUR 5.99; "launched in 2026", 2 upvotes; the page does not claim to analyse the camera's evidence image; "40% of Italian fines have formal defects" is the maker's claim — [Product Hunt](https://www.producthunt.com/products/ricorsoai)
- TIKD (US) has lawyers contest tickets; Western Australia reportedly saw thousands of successful appeals of AI-camera fines (about A$2.2M) per a search summary of a 9News item that I could not open (S) — [NetFM](https://www.netfm.net/news/9news/millions-of-dollars-in-ai-camera-fines-wiped/); [Technical.ly](https://technical.ly/?p=90578)
- Astor Legal: WA ordered a review after complaints grew (over 380,000 breaches detected, over 31,000 drivers fined in one month); an ACT driver got a fine withdrawn when the camera attributed an offence to a different vehicle; no counts of withdrawn fines — [Astor Legal](https://astorlegal.com.au/?p=37590)
- Since 1 Apr 2026 owners can transfer camera fines to the actual driver via EPIGU or the SBDD app; fee 10% of BRV (41,200 soum in the article, 37,080 via EPIGU); eligible bases include waybills and rental/car-sharing agreements; driver confirms by e-signature, Face ID or Mobile ID — [Spot, 1 Apr 2026](https://www.spot.uz/ru/2026/04/01/driver-responsibility/)
- Early payment: 50% off within 15 days, 30% off on days 16-30; at BRV 440,000: seat belt 220,000 soum (110,000 discounted), phone use 1,320,000 (660,000), speeding 20-40 km/h over 2,200,000; the page does not say how contesting affects the discount — [Oson Prava](https://osonprava.uz/ru/blog/tablica-shtrafov-2026)
- AI pedestrian-yield cameras at 10 Tashkent crossings issued about 1,700 fines in 3 days (blogger report confirmed by SBDD press secretary); no complaints, appeal route or vendor named — [Spot, Jul 2025](https://www.spot.uz/ru/2025/07/13/cameras-ai/)
- Plate formats: 01 A123BC standard and a three-digit/three-letter alternative (Wikipedia, uncited) — [Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan)
- Local plates will need an own dataset; fast-plate-ocr is MIT and fine-tunable (teammate tech radar) — [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)

### Inferences
- A belt fine of about $9 after discount, with a 41,200-soum transfer fee per notice, makes per-fine contesting uneconomic; only high fines (phone use, speeding) or wrongful fines justify effort.
- Value is mostly fleet workflow (pay at 50%, pick the real driver, track deadlines), not vision. The vision part pays only if plate/vehicle mismatches (misreads, cloned plates) exceed about 1-2% of fines.
- Judges may read it as a fine-evasion tool unless framed as payer-side verification.

### Gaps
- No Uzbek data on wrongful camera fines, cloned plates or appeal success (local queries blocked).
- WA "about 2,000 withdrawn" figure from the idea text is unverified; the snippet said thousands and A$2.2M.
- Unknown whether contesting forfeits the early-payment discount, and whether owners can download evidence images.

## matrix-2: Yard Tetris, blocked-in car planner for taxi parks

### Takeaway
Nothing found that combines vision, a blocking graph and extraction planning, but there is no evidence the problem exists at paying scale in Tashkent taxi parks, labour is cheap, and the solver is textbook. Verdict sharpen (re-aim to dealer/importer stock yards). Scores: novelty 4, feasibility 3, demo 3, business 1, moat 2, local fit 2.

### Cited Findings
- Automated-garage retrieval planning clears blockers by shifting them sideways into empty columns; planning only, no vision — [arXiv 2302.01305](https://arxiv.org/pdf/2302.01305) (S)
- Stanley Robotics (Paris, 2015): robots lift and move cars, storage software organises vehicles; customers Lyon Airport, Gatwick (VINCI); the page does not mention blocked-in retrieval — [Stanley Robotics](https://www.stanley-robotics.com/)
- Patent on autonomous vehicle retrieval from a parking space (S) — [US 12111165](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12111165)
- Cox Lot Vision is GPS/OBD-based, not camera (teammate) — [Cox](https://www.coxenterprises.com/news/cox-lot-vision-provides-dealerships-with-clear-vie)
- CARPK (89,777 cars, drone view, 40 m) is "for academic research purpose only", password-gated behind an EULA; PUCPR+ partly from PKLot — [CARPK page](https://lafi.github.io/LPN/)
- RF-DETR-Seg Nano: mask AP 40.3 at 3.4 ms on a T4, Apache-2.0; WebGPU demo exists (teammate tech radar) — [RF-DETR](https://github.com/roboflow/rf-detr)
- Yandex Go works only through fleets (135 in Mar 2023); drivers mostly self-employed on own cars (teammate, thesis) — [KSE thesis](https://kse.ua/wp-content/uploads/2025/09/Oleh_Dykyi_RIDE-HAILING-MARKET-IN-UZBEKISTAN.pdf)
- Average wage 7.09M soum in H1 2026 (about $600) — [Kun.uz](https://www.kun.uz/ru/news/2026/07/27/srednemesyachnaya-zarplata-v-uzbekistane-prevysila-7-mln-sumov)
- New-car sales fell 3.6% YoY Jan-Jul 2026 (teammate), a stock-aging pressure for dealers — [Anhor](https://anhor.uz/ru/news/avtomobili-kitajskie-brendy-uzbekistan)

### Inferences
- A Rush-Hour BFS ignores steering and turning-radius limits, so "shortest move sequence" is optimistic; oblique rooftop photos give merged masks.
- With low wages, a $50-80 lot-month fee needs more than about 20-30 saved attendant-hours a month; unproven.
- Stock-yard analytics is named in the brief and rated thin by the saturation map, so a locate/aging/blocked-by tool for dealer or importer yards is the defensible version.

### Gaps
- No evidence that Tashkent taxi parks keep packed multi-row lots; no dealer or yard counts; the dealership-lot query was blocked.

## wow-6: Cargo-Fit, one-photo vehicle and price quote

### Takeaway
Single-photo load-space estimation exists in B2B freight (Fraunhofer CargoSight, Continental) and phone-video inventory for movers (Yembo), but no cargo-taxi vehicle chooser was found; one photo works for single items, not stacked loads. Verdict sharpen. Scores: novelty 3, feasibility 3, demo 4, business 2, moat 2, local fit 4.

### Cited Findings
- CargoSight (Fraunhofer IML): one smartphone photo gives free area, remaining volume and utilisation of a truck load space, cloud, results in seconds, API into TMS/dispatch; no accuracy figures; unclear if commercial — [Fraunhofer IML](https://www.iml.fraunhofer.de/en/fields_of_activity/material-flow-systems/software_engineering/cargosight.html)
- Continental trailer-capacity service: driver photographs the cargo area with a smartphone, cloud algorithm returns fill level (2022 press release; page 404 on fetch) (S) — [Continental](https://www.continental.com/en/press/press-releases/20220728-trailer-capacity-assessment-services/)
- Yembo: smartphone capture, on-device AI and 3D reconstruction for moving and insurance inventories, 40+ countries; no accuracy figures and no truck-size recommendation on its page — [Yembo](https://yembo.ai/)
- MoGe-2 recovers a metric-scale point map from a single image; the abstract gives no numeric scale error — [arXiv 2507.02546](https://arxiv.org/abs/2507.02546)
- SAM 3 segments by short noun-phrase prompts in images and video — [arXiv 2511.16719](https://arxiv.org/abs/2511.16719)
- Jan-Sep 2025 Uzbek sales: Cobalt 95,362, Damas 78,712, Tracker 27,921, Labo 25,129, total 296,331 (Uzavtosanoat via Kun.uz) — [Kun.uz](https://kun.uz/en/news/2025/10/11/uzbekistans-car-market-sees-growing-competition-as-chevrolet-share-drops)
- MoGe-2 weights tagged MIT on HF (repo licence field says "Other", verify); SAM 3 weights gated, SAM License (teammate tech radar) — [MoGe repo](https://github.com/microsoft/MoGe)

### Inferences
- Damas + Labo = 103,841 of 296,331 new vehicles, about 35% of Jan-Sep 2025 sales (my arithmetic), so the vehicle set is right locally.
- Dimension error for stacked or occluded household loads will be far above the team's 5-8% estimate; realistic loads need a 20-30 s video.

### Gaps
- No Uzbek cargo-taxi dispatch data, wrong-vehicle rate (10-15% is an assumption), or evidence that ordering runs on Telegram groups.
- Russian-language query for photo-based truck selection blocked.

## insurer-5: FloodGauge, can my car cross this water

### Takeaway
No consumer GO/NO-GO wading app found (only research on flood depth from photos), but a safety-critical GO verdict from monocular depth is indefensible, the KASKO payer pool is small and shrinking, and local flood frequency is unsourced. Verdict kill. Scores: novelty 4, feasibility 3, demo 3, business 1, moat 2, local fit 2.

### Cited Findings
- Fire-service guidance: 10 cm is the widely recognised average safe depth for cars; 30 cm can move a car; about 60 cm can float it; any depth can flood the engine; "turn back" — [North Yorkshire Fire](https://www.northyorksfire.gov.uk/your-safety/safe-on-the-road/floodwater)
- Search summary of driver guidance pages: none estimate depth from a photo, and they warn that water hides dips in the road (S) — [What Car?](https://whatcar.com/news/flood-warning-for-drivers/n6127)
- GitHub topic flood-detection has 64 repos; none estimate depth from photos or vehicles (nearest: a MobileNet flooded/not-flooded classifier) — [GitHub topic](https://github.com/topics/flood-detection)
- 2025 Uzbek land-vehicle (KASKO-type, class 3) premiums fell 47.0% to 754,268 mln soum (about $64M); payouts 107,758 mln; contracts in force 660,550 (+13.2%) but contracts concluded in 2025 fell 41.0% to 237,405 (verified from the NAPP PDF by pdftotext) — [NAPP Q4 2025 report](https://napp.uz/storage/files/shares/opendata/2025/4Q/%D0%A1%D1%82%D1%80%D0%B0%D1%85%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%28IV%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%B0%202025%20%D0%B3%D0%BE%D0%B4%20%29%20rus%20%E2%80%94%20filtr.pdf)
- EdgeTAM (Apache-2.0) runs at 16 FPS on iPhone 15 Pro Max natively (not in a browser); DA3 Small/Metric-Large Apache-2.0 (teammate tech radar) — [EdgeTAM](https://github.com/facebookresearch/EdgeTAM)

### Inferences
- At $0.05 per driver-month, 660,550 KASKO contracts would yield at most about $0.4M a year (my arithmetic), before any adoption discount.
- Timestamped NO-GO evidence cuts against the driver: insurers could use it to deny hydrolock claims, so drivers have little reason to generate it.
- Rain, night, mud and reflections are exactly the cases where reference-object depth fails. Salvage only if inverted: never output GO, or use waterline-on-body for post-flood claim triage.

### Gaps
- The academic papers named in the idea (Kharazi and Behzadan; Notarangelo et al. 2025, 3,367 images) were not located or verified.
- No source for Tashkent flash-flood frequency or per-model wading depths of Cobalt/Nexia/Spark.

## fleet-2: PreTrip Pulse, 90-second driver fitness triage

### Takeaway
Pre-shift phone alertness tests (AlertMeter, NeuroUX), camera rPPG SDKs (Binah.ai) and CIS remote pre-trip exam kiosks with a remote doctor already exist, and drowsiness is a stock hackathon topic; "signs the e-waybill / replaces the medic" is unsupported. Verdict sharpen. Scores: novelty 2, feasibility 4, demo 5, business 3, moat 2, local fit 3; famous.

### Cited Findings
- AlertMeter: commercially available vigilance test, about two minutes, significantly correlated with the PVT in a 36-hour sleep-deprivation study (University of Denver, Mountain-Plains brief, Mar 2021); the brief cites NHTSA that 2-20% of annual traffic deaths are attributable to drowsiness — [UGPTI brief](https://www.ugpti.org/resources/reports/downloads/mpc21-432-brief.pdf)
- NeuroUX launched a PVT-based fatigue platform (article 27 Jul 2026): per-driver baselines, SMS/email/Slack alerts, no camera or wearable; independent validation for this platform not cited; no trucking customers named — [Bulk Transporter](https://www.bulktransporter.com/hazmat-safety/article/55393633/neuroux-launches-objective-fatigue-monitoring-platform)
- New Zealand linehaul firms trialled a phone-app fatigue test at shift start and breaks (S) — [Farmers Weekly NZ](https://www.farmersweekly.co.nz/?p=76055)
- Binah.ai: 35-60 s video-selfie rPPG, user at rest in good light, "validated" with no numeric accuracy on the page, no fleet use case — [Binah.ai](https://www.binah.ai/)
- GitHub topic rppg has 145 repos; rPPG-Toolbox 1.2k stars — [GitHub topic](https://github.com/topics/rppg); rPPG-Toolbox uses a Responsible AI Source Code License (use restrictions), MCD-rPPG mirrors are CC BY 4.0, MediaPipe Face Landmarker is Apache-2.0 (teammate tech radar) — [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox); [MCD-rPPG mirror](https://huggingface.co/datasets/wengziheng/mcd_rppg)
- Russia: remote pre-trip exams run on fleet-site terminals with card/ID login, a 180-degree camera, blood pressure, pulse, breath alcohol and a remote doctor; cost about half of in-person; Mosgortrans runs 240 units; forecast of up to 70% of fleets (2023) — [Rossiyskaya Gazeta](https://rg.ru/2023/02/21/medosmotr-voditelej-provedut-distancionno.html); [Retail.ru press release](https://www.retail.ru/rbc/pressreleases/udalennyy-meditsinskiy-osmotr-voditeley-2023/)
- Russian rules (Order 266n, Decree 866, from 1 Sep 2023): remote exams use medical devices transmitting data to a medical worker; remote exams not allowed for organised child bus transport, dangerous goods or intercity routes of 300 km or more — [PDD Master](https://pddmaster.ru/novoe/udalenniy-osmotr-0923.html)
- Uzbek e-waybill (Cabinet decree Jan 2024): must contain the driver's medical exam and vehicle technical inspection, cannot be opened without them; legal-entity carriers only; the sources do not say who certifies the checks; third parties may not receive the data — [Buxgalter.uz](https://buxgalter.uz/publish/doc/text196198_kak_budet_rabotat_sistema_ucheta_elektronnyh_putevyh_listov); [Spot, Jan 2024](https://www.spot.uz/ru/2024/01/25/driver-log/)
- Falling asleep caused 4.2% of Uzbek crashes in Q1 2026 (teammate) — [Kun.uz](https://kun.uz/en/50492709)

### Inferences
- The Russian model shows a paying CIS market but requires certified devices and a doctor's sign-off, so a selfie app fits only as a medic-assist layer.
- rPPG pulse is specified for a still user at rest; moving or dim cabins degrade it, so avoid medical-sounding claims without own validation.

### Gaps
- Who legally certifies the Uzbek pre-trip exam, and whether remote or automated exams are accepted, was not found (Uzbek query blocked).
- Whether the e-waybill platform exposes any API to third parties is unknown.

## fleet-6: BackRow, child-left-behind sweep for school buses

### Takeaway
Established vendors detect breathing or heartbeat or use thermal/PIR, a camera cannot see an occluded or sleeping child, the cheapest control (forced walk-through) exists, and no Uzbek mandate, incident data or fleet size was found. Verdict kill. Scores: novelty 2, feasibility 3, demo 3, business 2, moat 1, local fit 2; famous.

### Cited Findings
- EDGE3 Ambient Sensing Node (article 26 Oct 2017): AI product detecting children left in school buses, says it detects movement "as minute as heartbeats or breathing", telematics alert; retrofit status and price not stated — [School Bus Fleet](https://www.schoolbusfleet.com/news/new-device-designed-to-detect-children-left-in-vehicles)
- Dahua school-bus solution: PIR sensors plus AI video scan the bus, alarm and real-time alerts; claims "100% child safety assurance" with no accuracy data — [Dahua](https://www.dahuasecurity.com/uk/solutions/enterprise-solutions/education/school-bus)
- 2026 Sensors paper: infrared-array child-presence detection for school buses, 92.5% overall success in random tests; dataset and limitations not described on the page — [CiteDrive](https://www.citedrive.com/en/discovery/child-presence-detection-algorithm-in-school-buses-based-on-infrared-array/)
- GitHub topic child-presence-detection has no repositories (hackathon projects likely use other tags) — [GitHub topic](https://github.com/topics/child-presence-detection)
- OEM interior sensing exists (DTS AutoSense, teammate) — [SoCalTech](https://socaltech.com/dts_aims_at_passenger_monitoring_using_ai_camera/s-0080663.html)
- anomalib v2.7.0 is Apache-2.0 (MVTec AD benchmark is non-commercial); Raspberry Pi AI HAT+ 2 is $200 and a Pi 5 16 GB $305 (teammate tech radar) — [anomalib](https://github.com/open-edge-platform/anomalib); [AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/)

### Inferences
- A child under a jacket or between seats shows as a generic anomaly, producing either missed children or constant false alarms on bags and clothing.
- The $505 edge node (my sum of the two list prices) is far above the $3/bus/month price, so the phone-only version is the only one that fits the cost model.

### Gaps
- Uzbek school-bus fleet size, any mandate, and incident data not found (query blocked); SVIRO licence unverified (site certificate error).

## fleet-14: Lamp Roll-Call, solo brake-light and hyperflash check

### Takeaway
No phone-camera solo lamp checker found, but that is because the problem is trivial: a second person or a garage-door reflection does it, and indicator faults already reveal themselves through hyperflash. Verdict kill as a headline (fine as a module in a pre-trip walkaround). Scores: novelty 4, feasibility 5, demo 3, business 1, moat 1, local fit 3.

### Cited Findings
- Turn signals must flash 60-120 times per minute (1-2 Hz) and regulations require a bulb-failure alert via faster or slower flashing — [Wikipedia: Turn signal](https://en.wikipedia.org/wiki/Turn_signal)
- Single-person vehicle light test patent: a controller runs the lamps in sequence; no phone (2005 application) (S) — [Justia 20050258948](https://patents.justia.com/patent/20050258948)
- Apparatus for checking vehicle brake lights and air brakes (S) — [Justia 10343669](https://patents.justia.com/patent/10343669)
- Ford app offers a trailer light check (remote lamp activation, no camera) (S) — [Ford](https://www.ford.com/support/how-tos/ford-app/ford-app-supported-features/how-do-i-use-the-trailer-light-check-feature-in-the-ford-app)
- A UK guide notes a garage-door or window reflection check for brake lights, no app needed (S) — [Blackstone71](https://blackstone71.com/how-to-check-your-cars-brake-lights/)
- Moped/scooter riders face a fine of 1/2 BRV for no reflectors or lights since 28 Oct 2024 (teammate); Yandex says 92% of 23,785 moped-rider violations in 2026 were reflectors/headlight, tint or passengers — [BES.media](https://bes.media/amp/v-uzbekistane-vvodyat-shtrafi-za-narushenie-pdd-voditelyami-mopedov-i-elektrosamokatov-33fa43/); [Spot](https://spot.uz/ru/2026/10/06/couriers)
- E-waybill requires vehicle technical-inspection data before a waybill opens — [Buxgalter.uz](https://buxgalter.uz/publish/doc/text196198_kak_budet_rabotat_sistema_ucheta_elektronnyh_putevyh_listov)

### Inferences
- The alternative cost (a colleague for one minute) is near zero, so $1/vehicle-month has no hard-cost payer; couriers, not fleets, pay the moped lamp fines.
- Three of this batch (fleet-2, fleet-14, wild-14) are pre-trip checks that the e-waybill already requires; one combined "pre-trip pack" is a stronger product than any widget.

### Gaps
- AI-walkaround vendors (query blocked) and the identity of the App Store hit id913756562 (403) were not checked.
