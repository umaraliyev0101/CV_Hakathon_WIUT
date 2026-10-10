# Due-diligence verification, batch 02 (dealer / used-car segment): 10 ideas for the WIUT CV hackathon (Tashkent, 10-11 Oct 2026)

> Read-me. Written 2026-10-10 by the verifier for ideas safety-11, evcng-5, evcng-11, wild-7, wild-9, matrix-8, matrix-10, matrix-13, wow-3, wow-9.
> Tools: WebSearch (the 200-call session quota shared by all agents ran out mid-run, so every idea got at least one English search but several planned Russian, Chinese and Uzbek follow-ups did NOT run) and WebFetch of primary pages. Tags: **[SNIPPET]** = taken from a search-result summary, page body not read; **[CALC]** = my arithmetic on cited inputs; **[INFERENCE]** = my judgment; **[UNOFFICIAL]** = blog, vendor ad or aggregator. Conversion used: about 11,800 UZS per USD (teammate's `uzbekistan_market.md`). Judging weights context (from teammate's `hackathon_intel.md`, source [CV track page](https://hackathon.wiut.uz/tracks/computer-vision/)): Innovation and Originality 25%, Technical Complexity 25%, Impact and Practicality 25%, Presentation 15%, Business Potential 10%.
> Absence of a search hit is weak evidence: most "nothing found" statements below rest on one query.

---

## safety-11 HandoverFit: delivery-day seat and mirror setup coach

### Takeaway
No seat/mirror-fit coaching product turned up, but that looks like absent demand rather than an open niche: the dealer-paid revenue pool is tiny (about 0.54M USD a year at 100% attach across all new cars), the safety benefit cannot be validated in three days, and the outward-mirror method it would teach is disputed in the SAE literature. Verdict: kill.

### Cited Findings
#### Queries run
- EN extended: "AI app check driver seat position mirror adjustment blind spot camera". Not run (quota): Russian, Uzbek, OEM-feature probe, IIHS/NHTSA fact probe (IIHS page 404, NHTSA 403).
#### Findings
- The search "didn't turn up an app that checks your seat position and mirrors"; it returned how-to guides plus AI blind-spot hardware that does not check setup (commercial-vehicle BSD systems and a dashcam with blind-spot alerts) — [Car AI Blog guide](https://caraiblog.com/how-to-adjust-mirrors-for-the-best-blind-spot-coverage/); [Autoequips BSD](https://www.autoequips.com/ai/bsd/); [AKEEYO EYES](https://www.akeeyo.com/products/eyes) [SNIPPET]
- The mirror method the app would teach (outward-set mirrors, "BGE") traces to Platzer's 1995 SAE paper; the Wikipedia article reports no test data and cites a dissenting view that the method "is a driving mistake" and more dangerous than not using it (Wren, 2007 SAE) — [Wikipedia: Blind spot (vehicle)](https://en.wikipedia.org/wiki/Blind_spot_(vehicle))
- Head-restraint geometry has better evidence: backset over 10 cm is linked to more neck-injury symptoms and a properly placed restraint can lower neck-injury chance by up to 43% in rear-end crashes (secondary source) — [Wikipedia: Head restraint](https://en.wikipedia.org/wiki/Head_restraint)
- Uzbekistan new-car market 2025: 427,103 units, Chevrolet 83.2%, Kia 6.0%, BYD 5.6%, Chery 2.6%, Haval 2.1% [UNOFFICIAL, aggregator, via teammate] — [BestSellingCarsBlog](https://bestsellingcarsblog.com/2026/02/uzbekistan-full-year-2025-chevrolet-cobalt-at-30-1-share/)
- Tooling exists: MediaPipe Face Landmarker gives head pose in the browser, SAM 3 is available under the SAM License — teammate's tech radar, see [Face Landmarker docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) and [SAM 3 repo](https://github.com/facebookresearch/sam3)
### Inferences
- [CALC] 15,000 UZS x 427,103 new cars = 6.4 bn UZS, about 0.54M USD a year at 100% attach. The five brands above sum to 99.5%, so Toyota is under 0.5% (under about 2,100 cars); Toyota plus BYD (5.6% = about 23,900 cars) is at most about 26,000 deliveries, roughly 30-35k USD a year at 100% attach.
- MediaPipe has no headrest class, so the "headrest gap" needs a SAM 3 or open-vocabulary detector prompt and a scale reference (steering wheel diameter) to turn pixels into centimetres.
- A judge sitting in a chair at the venue gets a meaningless headrest gap; the live demo needs a car or staged photos.
### Gaps
- No Russian or Uzbek probe; no OEM driver-position feature check (some OEM occupant-monitoring systems measure seating position for restraints); IIHS/NHTSA primary fact pages unreachable; no evidence that any dealer pays for delivery briefings.

---

## evcng-5 Quvvat: used-EV battery capacity from charging photos

### Takeaway
No photo-only battery-health tool was found; the same job is done by OBD, telematics and certificate services (Aviloo, Altelium, Recurrent, TUV/Twaice) that do not list BYD-class Chinese EVs on the pages read. The physics limits accuracy (1% SoC steps, flat LFP curve, display error), so it works as a screening band, not a precise SOH. Strongest local fit in the batch. Verdict: keep.

### Cited Findings
#### Queries run
- EN extended: "used EV battery health check from charging session photo app state of health". EN standard: "Aviloo battery health test price used EV certificate flash test Recurrent battery report". Not run (quota): BYD OBD support probe, Chinese and Russian photo/charging-based SOH probes, Uzbek used-EV battery-check services.
#### Findings
- "The results don't show any app that estimates battery health from a photo of a charging session"; closest are EVScanner (records the charging curve via OBD), VoltChek (estimates from make/model/mileage) and SoH Checker (Android, drive data) — [VoltChek](https://voltchek.app/); [SoH Checker](https://play.google.com/store/apps/details?id=com.checker.soh&hl=en_US) [SNIPPET]
- EVScanner needs an OBD adapter and its supported list is VW group, BMW/MINI, Porsche Taycan, Skoda Enyaq, CUPRA, Ford; BYD and other Chinese EV brands are not listed; Android version "coming soon" — [EVScanner](https://evscanner.app/)
- Aviloo FLASH test: plug-and-play, result in about 3 minutes, claims ca. +/-3% against its own PREMIUM test, ordered through a business shop (business customers only), vehicle coverage in a separate database, BYD not mentioned on the page, no price — [Aviloo](https://www.aviloo.com/); a trade-press snippet says it runs via the OBD port [SNIPPET] — [Charged EVs](https://chargedevs.com/newswire/aviloo-extends-its-used-ev-battery-warranty-to-norway-and-denmark/)
- Other sellers: Altelium (UK; certificate via Assurant and GardX), Recurrent (tracks 20,000 EV owners; Black Book), TUV Rheinland with Twaice "Battery Quick Check"; a battery is about 40% of a new EV's price; Aviloo found battery health can vary by up to 30% after 100,000 km (page date not shown) — [Reuters via HL](https://www.hl.co.uk/news/reuters-the-battery-test-race-to-work-out-what-used-evs-are-really-worth)
- Recurrent builds a Range Score from VIN-specific battery data and about 40 million real EV trips [SNIPPET] — [Recurrent](https://www.recurrentauto.com/sell/tesla-model-3-y)
- Conventional advice treats charging history as context only: "Charging-history information adds context, but it shouldn't replace direct capacity and cell measurements" — [Recharged guide](https://recharged.com/articles/how-to-check-ev-battery-health)
- State-of-charge is hard to read: "Li-phosphate has a very flat discharge profile, making voltage estimations for SoC estimation difficult"; "SoC readings on new EV batteries can be off by 15 percent" (design-engineer quote); coulomb counting drifts without calibration — [Battery University BU-903](https://batteryuniversity.com/article/bu-903-how-to-measure-state-of-charge)
- BYD packs differ by variant on one body (Atto 3: 49.92 kWh, 60.48 kWh, 74.8 kWh Evo; all LFP Blade) so a nominal-capacity table is required — [Wikipedia: BYD Atto 3](https://en.wikipedia.org/wiki/BYD_Atto_3)
- Local demand: 122.7k EVs registered on 1 Jul 2026 (+66.7% YoY), 72% in Tashkent city — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4); EV imports H1 2026: 37,874 units, 99.2% from China — [Spot](https://www.spot.uz/ru/2026/08/14/ev-import-rise/); more than 3,000 EVs "with mileage" bought in Oct 2025 — [Kun.uz](https://kun.uz/en/news/2025/11/18/electric-car-purchases-surge-as-overall-auto-sales-recover-in-uzbekistan); EV loans at 12% (domestic) and 16% (foreign) — [Spot](https://www.spot.uz/ru/2026/01/10/ev-plans/)
- PP-OCRv6 (Apache-2.0) lists English and Chinese only; Cyrillic coverage unverified — teammate tech radar, [PP-OCRv6 small det](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det)
### Inferences
- [CALC] Quantisation alone: reading integer SoC at both ends leaves +/-1 point on the change, i.e. up to +/-3.3% of capacity for a 30-point session (+/-5% for a 20-point session; double that if the display is off by a point at each end). That uses up roughly half of the low end of the idea's +/-5-8% target, so display-SoC bias or non-linearity (worst on LFP mid-range), charger-to-pack losses and meter tolerance must fit in the rest. Aviloo claims ca. +/-3% against its own premium test.
- For 1-3-year-old cars, true SOH spread is likely within that error band, so the method can flag gross degradation (ex-taxi, high mileage) but not rank healthy cars.
- [CALC] Pool: 3,000 used EVs a month x 12 x 30,000 UZS = 1.08 bn UZS, about 91k USD a year at 100% attach; the real upside is bank/loan integration and the coming ex-taxi wave (the 300,000 EV-taxi goal is a plan, not an outcome).
- The idea's own premise (OBD support patchy on Chinese EVs) undercuts its ground truth: validation has to come from new cars (SOH about 100%) plus at least one known-degraded pack.
- Quvvat and evcng-11 are two halves of one flow (nominal kWh from nameplate, measured kWh from charging photos, ratio = SOH).
### Gaps
- No Chinese-language or Russian search for photo/charging-based SOH tools (Chinese used-car platforms may already do it); BYD OBD app coverage unchecked; Aviloo/Recurrent buyer prices not found; Reuters page undated.

---

## evcng-11 Yorliq: Chinese nameplate decoder for grey-import EVs

### Takeaway
Free Chinese-VIN decoders exist and photo translation is commodity; the new part (nameplate OCR to battery variant plus listing-mismatch flag) is thin and depends on a hand-built spec table whose key (plate or model code to pack size) is unverified. Weak standalone payer; best folded into Quvvat. Verdict: sharpen.

### Cited Findings
#### Queries run
- EN standard: "Chinese VIN decoder nameplate BYD Chery battery kWh variant parallel import". RU standard: "расшифровка VIN китайского электромобиля шильдик ёмкость батареи комплектация". Fetches: Product Hunt listing, Rozetked battery-plate article, BYD Atto 3 Wikipedia. Baidu Baike page (nameplate fields) returned 403, so the fields on the exterior vehicle plate are unverified.
#### Findings
- "Chinese Vehicle History Report": a free China VIN decoder for BYD, Geely, Great Wall, Changan; the maker says Carfax and AutoCheck do not cover China; no mention of nameplate photo reading or battery data; launch date not given — [Product Hunt](https://www.producthunt.com/p/chinese-vehicle-history-report/chinese-vehicle-history-report)
- Generic BYD VIN decoders return plant and date data but no battery capacity — [Vincario BYD decoder](https://vincario.com/vin-decoder/byd/) [SNIPPET]; Russian decoders exist — [7zap](https://7zap.com/ru/vin-decoder/)
- Battery capacity is not in the VIN; the pack carries its own nameplate (a leaked Xiaomi prototype pack plate shows 139 Ah / 101 kWh, 726.7 V, 642 kg) — [Rozetked](https://rozetked.me/news/29671); the article does not say where the plate sits on the car
- One body, several packs: Atto 3 / Yuan Plus 49.92 kWh (430 km CLTC), 60.48 kWh (510 km), 74.8 kWh Evo (650 km); markets differ in which variants they get — [Wikipedia: BYD Atto 3](https://en.wikipedia.org/wiki/BYD_Atto_3)
- A 227,000+ record CSV of Chinese vehicle type approvals compiled from the MIIT public announcement system is sold for EUR 50; its fields include model code, vehicle type, fuel type, dimensions, weights, top speed, seating and "the official VIN prefix(es) for each approved model", but battery capacity is not among the listed fields and English names are AI-generated (seller says lower-confidence rows are indicative) — [Payhip listing](https://payhip.com/b/3S6PE); so a plate or VIN-prefix to model-code table is cheap, while model-code to pack kWh still has to be built from brochures or another catalogue
- Local scale: 37,874 EV imports in H1 2026, 99.2% Chinese — [Spot](https://www.spot.uz/ru/2026/08/14/ev-import-rise/); every import must be tested at the Piskent polygon since 1 Nov 2024 — [Kun.uz](https://kun.uz/en/news/2025/02/24/uzbekistan-sees-sharp-decline-in-passenger-car-imports-amid-non-tariff-restrictions)
### Inferences
- [INFERENCE] The pack plate sits on the pack under the car and cannot be photographed at a handover, so the product must key off the exterior plate or model code; whether that maps one-to-one to pack size is the make-or-break unknown and was not verified.
- [CALC] 37,874 x 2 x 15,000 UZS = 1.14 bn UZS, about 96k USD a year at 100% attach.
- Plate OCR (PP-OCRv6 Chinese+Latin, VIN check digit) is commodity; RF-DETR plate localisation is unnecessary for a fixed-crop UI. The sellable asset is the verified mismatch data and bank/broker integration.
### Gaps
- Exterior nameplate field list for Chinese EVs not found; MIIT catalogue fields not opened; no check of Uzbek broker or bank demand; no Chinese-language search.

---

## wild-7 SafeDeal Triangle: seller, owner and car verified before payment

### Takeaway
The identity-and-payment triangle is already enforced by notaries and mandatory bank e-escrow for vehicles under ten years old since 1 Apr 2026, and MyID is a state-backed, iBeta-tested face-ID used by 31 banks and 8 marketplaces. Liveness plus face match plus ID OCR is stock KYC. Verdict: kill.

### Cited Findings
#### Queries run
- EN extended: "Uzbekistan car sale online verification seller owner tech passport MyID face escrow". Fetches: Kursiv escrow article, Kun.uz registration guide, MyID site, lex.uz Law on Personal Data, Biometric Update. Not run (quota): Russian safe-deal escrow probe, rPPG liveness robustness probe.
#### Findings
- From 1 Apr 2026, transactions in vehicles under ten years old (categories M, N, O, G, special vehicles) are verified by electronic confirmation of payments between notaries and banks; notaries approve contracts only after escrow is confirmed; Ministry of Justice registered regulation No. 3790 on 16 Mar 2026; the article mentions no ID or biometric checks — [Kursiv, 17 Mar 2026](https://uz.kursiv.media/en/2026-03-17/real-estate-and-vehicle-payments-in-uzbekistan-move-to-escrow-system/)
- Private-sale contracts between individuals must be notarized; exceptions: dealership, trade-in, leasing or credit organisation, used-car dealer, online auction platform; registration can be denied for mismatched or damaged VIN, vehicles listed as wanted or under arrest; representatives need a notarized power of attorney — [Kun.uz, 4 May 2026](https://kun.uz/en/news/2026/05/04/how-to-register-a-car-in-uzbekistan)
- MyID (UZINFOCOM): biometric ID with neural-network liveness and face recognition, passed iBeta testing for ISO 30107-3; connected: 31 commercial banks, 16 payment systems, 8 marketplaces, 6 government bodies, 45 mobile operators; use cases include online loans and KYC; price not listed — [MyID](https://myid.uz/en)
- Commodity KYC vendors already cover Uzbek documents with selfie match and liveness — [ID Analyzer](https://www.idanalyzer.com/en/supported-documents/uzbekistan); [IDMerit](https://www.idmerit.com/uzbekistan/) [SNIPPET, vendor pages]
- Law on Personal Data (No. ZRU-547, 2 Jul 2019, in force 1 Oct 2019): biometric data needs consent (Art. 26); biometric, genetic and telecom-user data must be stored in Uzbekistan (Art. 27-1); databases must be registered (Art. 20) — [Lex.uz](https://lex.uz/en/docs/4396419). The idea's "2026 amendment" could not be verified, but localisation is already in the text.
- The "triangle" fraud itself is documented in CIS press: a fraudster copies a listing, reposts it below market, poses as the owner to the buyer and as a buyer to the real seller, and takes the payment to his card (Ukraine, 28 Mar 2023; no sums given) — [Borg Expert](https://borgexpert.com/ru/news/v-ukrayne-poiavylas-novaia-skhema-obmana-pokupatelej-avtomobylej); the mandatory notary plus escrow flow above is the Uzbek countermeasure
- Used deals run 52-57k a month (teammate) — [Kun.uz Oct 2025](https://kun.uz/en/news/2025/11/18/electric-car-purchases-surge-as-overall-auto-sales-recover-in-uzbekistan); [Spot Nov 2025](https://www.spot.uz/ru/2025/12/22/auto-market/)
### Inferences
- [CALC] 30,000 UZS x 625-690k deals a year = 18.7-20.7 bn UZS, 1.6-1.75M USD a year at 100% attach, the largest pool in this batch, but the pay point is occupied by notary, bank escrow and MyID.
- rPPG pulse liveness is fragile: a replayed recording of the real owner also carries pulse; treat the passive-pulse claim as untested. [INFERENCE]
- The only non-duplicated leg is the car leg (VIN plate versus windshield versus tech passport, tamper cues), which fits inside wild-9 rather than as a product.
### Gaps
- Whether notaries use MyID or an equivalent face check was not found; fees of the new escrow regime not stated; no Russian safe-deal (Avito/Auto.ru) check run.

---

## wild-9 Proof-of-Inspection: AI audits the inspector, not the car

### Takeaway
The "audit the inspector" angle is not sold by anyone I found, but incumbents already ship the machine-checkable process controls (guided capture, gaps, reused media, VIN/plate match). The payer is unclear and the semantic claim-versus-frame checking is the weakest, hallucination-prone part. Verdict: sharpen to the verifiable layer.

### Cited Findings
#### Queries run
- EN extended: "AI audit vehicle inspection video inspector fraud used car inspection quality assurance". RU standard: "автоподборщик откат от продавца схема обман покупателя" (ran; second Russian audit query hit the quota). Fetches: Inspektlabs fraud blog, Auto.ru Mag article, Drive2 inspector post, Autonews article.
#### Findings
- Results were almost all Inspektlabs marketing; vendor claims, no independent validation; it pitches AI as the fix for human-inspector bias — [Inspektlabs](https://inspektlabs.com/) [SNIPPET]
- Inspektlabs process checks: guided capture prompts recapture of missed parts; blocks old images; flags car switching; "a gap of more than 45 seconds between two images could indicate fraud"; VIN and plate matched against external databases; sticker detection; repaint/prior-repair cues; "best used to support investigators"; no accuracy data, only a 2017 Verisk industry-leakage estimate — [Inspektlabs blog](https://inspektlabs.com/blog/detecting-fraud-in-the-automotive-industry-through-ai-inspections-inspektlabs/)
- Bdeo lists VIN, GPS, plate, odometer and screen-detection checks (teammate) — [Bdeo blog](https://bdeo.io/en/blog/detecting-auto-insurance-fraud-in-the-age-of-generative-ai/)
- Kickbacks to inspectors, anecdotal: an inspector blogging in Feb 2020 alleges inspectors reject good cars to steer clients to "grey dealers", take a cut of price reductions (15-50k RUB), and are paid by sellers to recommend cars in remote inspections; remote reports rely on photos and thickness-gauge readings that can be chosen selectively — [Drive2, 17 Feb 2020](https://www.drive2.ru/b/553065344961347648) (single blogger, Russia, not independent)
- A different CIS autoselection scam (fake firm takes custody of the car under a commission agreement; more than 10 complaints, over 10M RUB damage, four detained) — [Auto.ru Mag, 1 May 2025](https://auto.ru/mag/article/voryavtopodborshchiki-razbiraem-s-maksimom-shelkovym-kriminalnuyu-shemu-s-osmotrom-avtomobilya/); the article does not discuss kickbacks or falsified reports
- Tashkent avtopodbor starts at 300,000 UZS (OLX ad, 7 Oct 2026) — [OLX.uz](https://www.olx.uz/d/oz/obyavlenie/ot-300-000-avtopodbor-proverka-avto-probega-pered-pokupkoj-avtotekshi-ID3XhPY.html)
- Tractable's homepage mentions no estimate-review or appraiser-audit product — [Tractable](https://tractable.ai/)
### Inferences
- [CALC] 10,000 UZS per audit on a 300,000 UZS service (3.3% overhead); if every used deal (625-690k a year) were audited, about 0.53-0.58M USD a year, so realistic revenue is a small fraction.
- C2PA signing from a web page cannot prove capture authenticity (attestation needs a native camera path or secure hardware), so "tamper-evident archive" claims must be scoped to server-side hashing. [INFERENCE]
- Forty staged inspections with deliberate omissions is heavy for a 2-3-day build; scripted toy-car demo is the realistic scope.
### Gaps
- No Russian or Uzbek product search for inspection-audit tools; no local evidence of inspector collusion in Uzbekistan; number and size of Tashkent avtopodbor firms unknown.

---

## matrix-8 Sotiladi Scout: dashcam lead-finder for used-car dealers

### Takeaway
No product that reads for-sale notes off windshields was found (English query only), so novelty is genuine, but lead yield, lead exclusivity and personal-data exposure are unproven and the shared "street map" has a short shelf life. Verdict: sharpen.

### Cited Findings
#### Queries run
- EN extended: "for sale sign windshield OCR phone number dashcam app used car dealers leads". Not run (quota): Russian, Uzbek ("sotiladi"), Chinese; a DRN-style plate-network probe.
#### Findings
- "The search didn't turn up any product that combines these pieces": dealer lead apps (Cars.com, CarsForSale), phone VIN-OCR through windshields (Laser Appraiser), lot-sign generators, dashcam retail pages — [Laser Appraiser](https://laserappraiser.com/); [Cars.com Dealer app](https://play.google.com/store/apps/details?id=inc.carscommerce.app&hl=en_US); [CarsForSale dealer app](https://dealers.carsforsale.com/dealer-mobile-app/) [SNIPPET]
- Data-protection exposure: ordinary personal data may be processed with provable consent; databases of personal data must be registered in the state registry (Art. 20); data in public sources with the subject's consent is treated as publicly available (Art. 29) — [Lex.uz Law on Personal Data](https://lex.uz/en/docs/4396419); the law's treatment of a phone number on a car note is not addressed in the text read
- Used deals run 52-57k a month; Sergeli market intermediaries and fraud cases — [Kun.uz](https://kun.uz/en/news/2025/11/18/electric-car-purchases-surge-as-overall-auto-sales-recover-in-uzbekistan); [Anhor](https://anhor.uz/ru/news/zloumyshlenniki-vstrechali-posetitelej)
### Inferences
- Detecting small, glare-prone, possibly tinted-glass notes from a moving dashcam is the hard part; no dataset exists, and 500 self-collected frames in three hours is thin but fine for a demo with printed props.
- Moat is weaker than claimed: notes go stale within days, and dealers will not share leads they paid to find, so the "compounding map" conflicts with lead exclusivity. [INFERENCE]
- Keeping numbers on the device (the idea's design) avoids a server-side personal-data database, a real compliance advantage worth keeping.
### Gaps
- Notes-per-hour of driving, OCR hit rate and call-to-deal conversion all unknown; legal reading of phone numbers on windshields not confirmed by a lawyer; no CIS-language search.

---

## matrix-10 Lowside: crash-history check for used motorcycles and scooters

### Takeaway
AI motorcycle inspection exists at consumer price (Motoryk $4.99 per report; carVertical photo-damage flags), but nothing found infers past crashes from contact-point wear. The weak point is ground truth: no labeled crash-versus-wear data exists and the team cannot create it in 2-3 days. Verdict: sharpen to claims it can evidence.

### Cited Findings
#### Queries run
- EN extended: "used motorcycle damage detection AI photos crash history dropped bike". Not run (quota): Russian and Chinese motorcycle-condition probes. Fetches: Motoryk site, carVertical blog, Biker Insure page.
#### Findings
- Motoryk: record a walk-around video, AI checks "50+ points" incl. "Frame & crash damage", returns a 1-10 score, repair costs and negotiation points; free scan, $4.99 per report, $9.99 a month; says it will not find everything a lift and compression test would; drop history not stated — [Motoryk](https://motoryk.com/)
- carVertical: AI reviews the photos in a history report to flag past damage (blog 14 May 2024, no accuracy figures, motorcycles not mentioned) — [carVertical blog](https://www.carvertical.com/blog/new-ai-damage-feature); its UK motorbike check claims "Artificial intelligence is used to detect damage from motorcycle photos", reports from GBP 33.99 — [Biker Insure](https://www.bikerinsure.co.uk/motorbike-history-check.html)
- A claims-photo patent lists the difficulties: unclear angles, irrelevant photos, minor damage visible only in close-ups — [USPTO 12731402](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12731402) [SNIPPET]
- Local risk and market: 563 moped/scooter crashes and 89 deaths Jan-Sep 2026 (SBDD); mopeds plated since 1 Oct 2025; A1 licence introduced late March 2026 — [Spot, 6 Oct 2026](https://spot.uz/ru/2026/10/06/couriers); OLX motorcycles category about 12.7k listings on 2026-10-10 (teammate, LIVE counter) — [OLX transport](https://www.olx.uz/transport/)
### Inferences
- Weak labels from OLX photos plus inpainted scuffs risk circular validation; a "clean, dropped or slid" verdict is not defensible without mechanic-confirmed examples. [INFERENCE]
- Price anchor: Motoryk's $4.99 undercuts nothing at $2 per check, so the wedge has to be integration (lessor check-in/out), not the check itself.
### Gaps
- No two-wheeler crash-signature dataset found; unknown used-moped transaction volume; no price evidence for dealers or lessors in Uzbekistan.

---

## matrix-13 Trim Truth: trim and equipment verifier for car loans

### Takeaway
Trim-from-photo classifiers and apps exist, exterior photos cannot show most equipment that separates trims, and the loan-fraud pattern ("inflated trim") is assumed rather than evidenced. Strong interactive demo, weak payer. Verdict: sharpen (merge wow-3 into it).

### Cited Findings
#### Queries run
- EN standard: "car trim level equipment recognition from photo AI API loan fraud inflated trim". Not run (quota): Russian Avito/Auto.ru trim probe, bank-fraud probe. Fetches: Vehicle AI page, VinQuery VMMR page; Nyckel page returned 404.
#### Findings
- Nyckel offers a pretrained "Challenger trim" classifier with 18 labels and an API on request, and lists finance fraud detection among use cases — [Nyckel](https://nyckel.com/pretrained-classifiers/challenger-trim) [SNIPPET]
- Google's Gemini API "Vehicle AI" (a competition project from Brazil) returns make, model, colour and estimated year from a photo, no trim — [Google AI](https://ai.google.dev/competition/projects/vehicle-ai)
- VinQuery VMMR: 43 makes, 659 models, generation data 2010-2021, US and Canada passenger cars only, no trim field, "never been tested on videos" — [VinQuery VMMR](https://2016-2026.vinquery.com/vmmr)
- Russian prior art for photo-based condition and "fair price": Auto.ru (21 Mar 2024) scans listing photos, trained on more than 100,000 inspected cars — [Snob.ru](https://snob.ru/news/avto-ru-zapustil-neiroset-dlia-otsenki-avtomobilia-po-foto/); dealer photo tools such as Spyne — [Spyne](https://www.spyne.ai/)
- Parc context: Chevrolet 83% of Jan-Sep 2025 domestic-assembly sales — [Kun.uz](https://kun.uz/en/news/2025/10/11/uzbekistans-car-market-sees-growing-competition-as-chevrolet-share-drops); Chinese brands 17.8% of Jan-Jul 2026 sales — [Anhor](https://anhor.uz/ru/news/avtomobili-kitajskie-brendy-uzbekistan)
### Inferences
- [CALC] 0.50 USD x 625-690k used deals = 0.31-0.35M USD a year at 100% attach; new-car loans are invoice-based through dealers, so the inflated-trim risk is mostly a used-car or marketplace-listing problem. Banks financing used cars appear to be a small segment (not quantified).
- Labels scraped from listings are noisy because sellers misstate trims, which is the very fraud being detected. [INFERENCE]
- Exterior photos cannot show interior or mechanical equipment that separates many trims (ABS, airbags, climate control, multimedia), so an exterior-only classifier can verify wheels, fog lamps, rails, sunroof and powertrain cues but not the full equipment list; the search tool's own summary reached the same caveat and suggested checking trim against VIN decoding rather than photos alone (no accuracy figures for trim detection exist in any source read). [INFERENCE]
### Gaps
- Whether Uzbek tech passports record trim not found; Cobalt trim equipment list (UzAuto site unreachable) not checked; no loan-officer evidence of trim inflation.

---

## wow-3 Window Sticker: instant trim and price from photos

### Takeaway
Photo to make/model/price is old news (Yandex Auto.ru, 2016) and available as APIs and hackathon apps; the trim/equipment layer duplicates matrix-13 and is unreliable from the outside. Great demo, no moat or payer. Verdict: kill.

### Cited Findings
#### Queries run
- EN standard: "car make model generation recognition API photo Ximilar Carnet.ai trim" (neither Ximilar nor Carnet.ai surfaced). Not run (quota): Russian Avito photo-valuation probe. See matrix-13 for the trim/loan search.
#### Findings
- Commodity recognition APIs: Zyla Labs Vehicle Make Capture API (make, model, generation, colour, angle, years; no trim in sample) — [Zyla Labs](https://zylalabs.com/api-marketplace/automotive+%26+vehicle+data/vehicle+make+capture+api/10312) [SNIPPET]; Adaptive Recognition Carmen MMR (enterprise) — [brief](https://adaptiverecognition.com/app/uploads/DOC/Software/Carmen/MMR/mmr_brief_description.pdf) [SNIPPET]; Roboflow open make/model set — [Roboflow](https://universe.roboflow.com/hr-alpha/car-make-and-model-recognition) [SNIPPET]; consumer apps — [Car Identifier & AI Scanner](https://mwm.ai/apps/car-identifier-ai-scanner/6748849086) [SNIPPET]
- Old prior art: Yandex Auto.ru app recognised a car from a photo and showed an approximate price on 30 Jun 2016 — [Tproger](https://tproger.ru/news/chto-eto-za-mashina/); Auto.ru 2024 condition grading from listing photos — [Snob.ru](https://snob.ru/news/avto-ru-zapustil-neiroset-dlia-otsenki-avtomobilia-po-foto/)
- A hackathon-grade Gemini app does vehicle ID from a photo — [Google AI Vehicle AI](https://ai.google.dev/competition/projects/vehicle-ai)
- Local mix: Chevrolet 83.2% of 2025 new sales, Cobalt 30.1% [UNOFFICIAL] — [BestSellingCarsBlog](https://bestsellingcarsblog.com/2026/02/uzbekistan-full-year-2025-chevrolet-cobalt-at-30-1-share/); no Uzbek vehicle-history product found (teammate)
### Inferences
- [CALC] 0.20 USD x 625-690k = 0.125-0.14M USD a year at 100% attach.
- A judge testing their own car is a strong demo but a public failure mode if the trim is wrong; the 4-photo voting and "unknown" fields mitigate this.
### Gaps
- No Russian-language photo-valuation search (Avito, Auto.ru current features); no price model source for the "fair price band" (scraping OLX/Avtoelon may breach terms).

---

## wow-9 Lot Whisperer: buyer-attention heatmaps for car lots

### Takeaway
Per-vehicle dwell time, heatmaps and staff exclusion are already sold for showrooms, and V-Count markets an outdoor sensor for used-car lots; people-tracking heatmaps are stock hackathon material. The reprice recommendation needs sales-outcome data the team cannot have. Verdict: kill.

### Cited Findings
#### Queries run
- EN standard: "car dealership lot shopper dwell time AI cameras vehicle engagement analytics". Fetches: FootfallCam post, V-Count page; Crosscan page timed out twice. An outdoor-lot startup probe hit the quota.
#### Findings
- V-Count: display bays configured as zones reporting visitors and time spent per vehicle or model group; heatmap analytics; staff exclusion; "Nano Outdoor" for weather-exposed forecourts and used-car lots; per-sensor pricing, no figures; 99% door-count accuracy is vendor-stated and must be validated per site — [V-Count car showroom](https://v-count.com/car-showroom)
- FootfallCam (post 7 Nov 2025, updated 12 Nov 2025): "measure dwell time at each vehicle or zone", zone heat maps, automatic staff exclusion; ceiling-mounted showroom hardware, outdoor lots not addressed; no prices — [FootfallCam](https://www.footfallcam.com/blog/?p=7510)
- Crosscan: ceiling 3D sensor counting how often shoppers look at a vehicle's interior, compared with daily visitors [SNIPPET] — [Crosscan](https://crosscan.com/en/industries/car-dealerships)
- Service-lane CV already funded: Skaivision (Reynolds & Reynolds investment, Dec 2024), EyeQ — [Digital Dealer](https://digitaldealer.com/news/reynolds-and-reynolds-partners-with-skaivision-to-change-the-automotive-retail-landscape/162791/)
- Local: Jan-Jul 2026 new-car sales -3.6% YoY, fourth negative month — [Anhor](https://anhor.uz/ru/news/avtomobili-kitajskie-brendy-uzbekistan)
### Inferences
- RF-DETR-Nano plus a tracker and proximity zones is a one-day build, but "predicted days-to-sell after a price edit" would be invented without outcome data; judges can spot that.
- Open-air markets bring camera-access and privacy questions (people tracked in a public market) with no operator partner named. [INFERENCE]
### Gaps
- No evidence of Tashkent dealer or market CCTV availability or willingness to pay; no outdoor-lot accuracy figures from any vendor.

---

## Cross-idea notes
- Merge candidates: evcng-5 + evcng-11 (one "EV passport" flow); matrix-13 + wow-3 (one trim/equipment verifier); wild-7's car leg folds into wild-9.
- Teammate context used: `saturation_map.md` (Auto.ru 2016/2024, Spyne, Bdeo, Skaivision), `uzbekistan_market.md` (EV fleet, used deals, moped crashes, avtopodbor price), `tech_radar.md` (model availability and licences: RF-DETR, SAM 3, SigLIP 2, PP-OCRv6, MediaPipe), `hackathon_intel.md` (judging weights, sponsors Toyota/BYD/Ducati via Royal Motors).
- Not verified anywhere: the "2026 biometric amendment" in wild-7, the CNG/EV-taxi plans as outcomes, Cyrillic coverage of PP-OCRv6, phone-browser FPS claims in matrix-8 and wow-9.
