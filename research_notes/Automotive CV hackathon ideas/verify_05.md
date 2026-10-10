# Due-diligence notes, batch 05: service-aftermarket ideas (service-8, -10, -12, -14, -15, -16, -17, -18, evcng-8, wild-1)

> Method note (not a findings section). Compiled 2026-10-10. The shared WebSearch budget (200 calls across all agents) ran out during this run, so only 12 searches succeeded (the last one got through after a partial refill); the other second-pass English, Russian and Uzbek queries were refused. I did not work around the limit. Gaps were closed with direct WebFetch of specific pages that surfaced in results, plus the teammates' notes in the same folder (saturation_map.md, uzbekistan_market.md, hackathon_intel.md, tech_radar.md). Tag **(S)** = claim taken from a search-result summary; the page itself was not read (blocked or not fetched). Untagged = page fetched and read. Scores in the final structured output follow the brief's 1-5 scales; USD conversions use about 11,800 UZS per USD (teammate fact sheet).
>
> Searches that ran: (1) counterfeit auto parts smartphone authentication app packaging; (2) ceramic coating water contact angle measurement smartphone app; (3) dashboard warning lights ignition on check video AI detect missing airbag light used car; (4) wheel nut torque verification computer vision tyre shop; (5) underbody inspection camera service pit AI defect detection garage; (6) brake pad thickness photo AI app tyre shop wheel off; (7) windshield pitting night glare measurement smartphone photo; (8) paintless dent repair stripe reflection dent depth estimate app deflectometry smartphone; (9) soap bubble leak test video computer vision automatic detection; (10) smartphone video motion magnification engine vibration diagnosis car mechanic; (11) RU: обманка подушки безопасности лампа SRS не горит при покупке авто; (12) RU: контрафактные автозапчасти Узбекистан подделка фильтры масла.
>
> Searches refused for budget (not run): Authentic Vision/Zebra; AlpVision fingerprint; UZ original parts QR; Asl Belgisi marking of oils/tyres (the later query 12 partly covered this); instrument-cluster telltale camera test; RU airbag emulator; loose wheel nut witness-mark app (twice); brake pad wear deep learning / AI tyre-shop app (three attempts); Dent Light app (three attempts); ceramic coating contact angle detailer (twice); RU Tashkent detailing prices (twice); RU pit/underbody neural net; RU GBO soap-test practice (twice); Eulerian magnification engine NVH; engine mount phone video.
>
> Pages that failed to load: securingindustry.com (DNS), thebrakereport.com (403), collisionrepairmag.com (403), Justia patent 6115118 (403), SWOV (bot wall), iris.cnr.it (403), PakWheels thread (title only), PropertyCasualty360 (title only), aslbelgisi.uz (expired TLS certificate), Optica Express (placeholder), Wards (404), checkpointindicators.com (DNS), apps.apple.com (one 429, retry worked), USPTO scanned PDFs 12333708 and 12366499 (image-only; read via Google Patents instead).

## service-8: FakeStop (counterfeit filter, pad and plug scanner)

### Takeaway
Phone-based authentication of auto parts is already sold by funded vendors (Authentic Vision with Dana and Bilstein, Alitheon backed by BMW i Ventures, Ennoventure, AlpVision), so the technique class is not new; the open wedge is brand-agnostic, no-label-change checking for CIS bazaar parts. The staged print-scan-reprint demo is circular, and local counterfeit prevalence and a paying brand owner are unverified. Verdict: sharpen.

### Cited Findings
- Authentic Vision (Salzburg HQ, US office in El Segundo) sells a "Holographic Fingerprint" label checked by a smartphone camera against a database; lists Automotive Aftermarket among industries and Dana Incorporated and Bilstein under "Trusted by Brands"; ownership not stated on the page — [Authentic Vision](https://www.authenticvision.com)
- Dana added a security label with serialised QR code and hologram, and its app returns a genuine-or-counterfeit result; built with Authentic Vision (search-result summary; page not read) (S) — [Securing Industry](https://securingindustry.com/electronics-and-industrial/automotive-parts-supplier-dana-rolls-out-security-labels/s105/a16198)
- Alitheon: "digital fingerprinting" of minute manufacturing features, matched from a photo against cloud records to tell a certified part from a fake; BMW i Ventures invested (amount undisclosed); article dated 9 Mar 2022; business clients only at that time, consumer app planned; cites a US Department of Commerce estimate of about $45 billion for the counterfeit auto-parts industry — [Newsweek](https://www.newsweek.com/bmw-makes-big-investment-fight-counterfeiting-1685025)
- Ennoventure: invisible cryptographic signatures in OEM part packaging, phone scan returns genuine or suspect; no customers named; its "20-40% of OEM cost", "20% counterfeit penetration" and "840,000 fake spark plugs" figures are uncited vendor claims — [Ennoventure](https://ennoventure.com/counterfeit-auto-parts)
- AlpVision: Cryptoglyph (20-60 micron holes in varnish or solid colour, needs a printing change) and Fingerprint (existing surface irregularities of molded or stamped products, reference images stored); lubricants named among markets; no auto-parts customers or accuracy numbers on the page — [AlpVision](https://alpvision.com/smartphone-product-authentication)
- Brembo released a smartphone app to help determine whether a product bearing its name is genuine (search summary only; page returned 403) (S) — [The Brake Report](https://thebrakereport.com/app-can-determine-counterfeit-brembo-parts/)
- Entrupy: smartphone AI authentication for bags, footwear, apparel and jewelry, advertises 99.86% accuracy; no auto-parts category; the page does not mention a clip-on microscope — [Entrupy](https://www.entrupy.com)
- anomalib v2.7.0 (Apache-2.0) ships MH-PatchCore, RAD and few-shot FoundAD; MVTec AD is CC BY-NC-SA, "not allowed ... for commercial purposes" — [anomalib](https://github.com/open-edge-platform/anomalib); [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Chevrolet was 83% of new-vehicle sales in Jan-Sep 2025 (89% a year earlier) — [Kun.uz](https://kun.uz/en/news/2025/10/11/uzbekistans-car-market-sees-growing-competition-as-chevrolet-share-drops)
- Tashkent police raided the Sergeli car market over intermediaries who sell parts at several times the price (about 100 people identified); this is overcharging, not proven counterfeiting (teammate fact sheet) — [Anhor](https://anhor.uz/ru/news/zloumyshlenniki-vstrechali-posetitelej)
- Uzbekistan, 27 Apr 2022: the State Tax Committee tasked CRPT Turon (the local arm of Russia's CRPT marking operator) with studying mandatory marking of tyres, auto parts and carpets because "the level of counterfeit and falsified goods is high"; no figures, no deadlines, the Asl Belgisi name is not used; status after 2022 not found — [Anhor](https://anhor.uz/news/tires/)
- Uzbekistan, 5 Jan 2017 (Sputnik via Anhor): a GM Uzbekistan after-sales representative said counterfeit body parts threaten driver safety; an original hood is about 800,000 sum at dealers vs about 650,000 sum for a fake at car markets; only body parts are discussed, no market-share figure — [Anhor](https://anhor.uz/news/poddelynie-avtozapcasti/)
- Russia, Dec 2025 (Gruzdev Analyze with the Fit Service repair network): 32% of repair shops met fake parts in 2025 (37% in 2024); most reported categories were motor oils and technical fluids 48%, filters 32%, suspension and steering 17% — [Za Rulem](https://www.zr.ru/content/news/975517-ehti-detali-dlya-avtomobilya-podd/)
- Russia, Oct 2024 (Aftermarket-Data founder): counterfeit volume about 0.25% of the parts market, the "up to 50% of oils are fake" claim is "clearly overstated"; counterfeiters refill premium-brand canisters with cheap base oil; turnover by channel: independent service stations 33%, e-commerce 26%, offline stores 24%, dealers 17% — [Autonews](https://www.autonews.ru/amp/news/670e265d9a794740b472edcd)
- Russian roundups rank oils first and filters second among fakes and say fakes are almost impossible to spot by eye (S) — [Harant](https://harant.ru/blog/grazhdanskoe-pravo/kontrafakt-na-polkah-kak-ne-kupit-poddelnye-zapchasti-v-2026-godu/); a Gazeta.ru item says fakes grew several-fold since 2022 (S) — [Gazeta.ru](https://www.gazeta.ru/auto/2026/05/20/22954393.shtml)

### Inferences
- Prevalence evidence is Russian, dated and contested (0.25% of volume vs 32% of shops meeting fakes are different metrics); it concentrates in consumables (oils, filters), and for oils the fake often sits in a genuine refilled canister, which no packaging check can catch. Filters, pads and plugs in printed boxes are the physically checkable categories.
- If Uzbekistan's marking study (2022) ever produced mandatory codes on auto parts, official codes would replace much of the "is this box genuine" question; the surviving problem would be cloned codes, which print-microtexture checks can address. Status unknown.
- The technique (hologram, invisible marker, natural-feature fingerprint) is occupied by enterprise vendors that need brand cooperation; FakeStop's difference (no brand cooperation, bazaar focus) is also its accuracy risk, because a reference built from 10 purchased boxes cannot capture batch-to-batch print variation.
- Self-made print-scan-reprint fakes differ from offset originals in halftone structure, so the demo will separate them almost by construction; judges can call it circular. Professionally offset-printed bazaar counterfeits are the real test.
- The payer is a brand owner or importer, not a garage; with Chevrolet at 83% of new sales, a UzAuto Motors parts-chain partnership is the most plausible anchor (my inference, no contact evidence).

### Gaps
- No current Uzbek counterfeit-prevalence figure found (the 2017 and 2022 Uzbek items give no share); whether any auto-parts marking was ever mandated after the 2022 study is unverified, and Asl Belgisi coverage of oils, filters or tyres could not be checked (site certificate expired); no list prices for Authentic Vision, Alitheon or Ennoventure; Brembo app mechanics and Dana article body not read.

## service-10: Bead-Check (water-droplet coating health meter)

### Takeaway
Smartphone contact-angle measurement exists for labs (Droplet Lab's Dropometer at $4,990; the free Joninn app, unsuitable above 85 degrees), but no car-detailing product was found. The physics is sound and the demo is cheap; the market is tiny, and an absolute "coating verdict" is ambiguous because waxes, sealants and ceramics overlap. A drift-tracking, dealer-channel version is better. Verdict: sharpen.

### Cited Findings
- Droplet Lab Dropometer: portable AI drop-shape analyser on a configured Google Pixel or an iPhone (iOS 15+); measures contact angle, surface tension, surface energy, sliding angle; one-time $4,990 (education unit $999); targets scientists, educators, R&D and QC; no mention of automotive, detailing or ceramic coatings — [Droplet Lab](https://dropletlab.com)
- Droplet Lab method: backlit sessile drop, Otsu threshold, Young-Laplace or polynomial fit; static angles matched a Kruss DSA100E "within ~1 degree"; hardware includes a levelled stage and a syringe, i.e. flat lab samples — [Droplet Lab blog](https://dropletlab.com/blog/measuring-contact-angles-with-a-smartphone/)
- Joninn (free iOS app, v1.1 dated 27 Nov 2017): top-down photo with a mm ruler, modified Bikerman method; good below 75 degrees, limited 75-85, unsuitable above 85; only hydrophilic and neutral surfaces; no ratings — [App Store](https://apps.apple.com/app/id1214623938)
- Steven Abbott's browser sessile-drop tool: spherical-cap fit is fine for small drops (example 10 microlitres), larger drops need Young-Laplace; camera tilt and lighting are the main problems; "If +/-1 degree makes little difference ... you're fine" — [Abbott](https://www.stevenabbott.co.uk/practical-coatings/Sessile-Drop.php)
- Registered EVs 122.7k on 1 Jul 2026 (+66.7% YoY), 72% in Tashkent city — [Statistics Committee](https://stat.uz/en/press-center/news-of-committee/70264-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-4); Chinese brands 17.8% of Jan-Jul 2026 sales vs 11.6% — [Anhor](https://anhor.uz/ru/news/avtomobili-kitajskie-brendy-uzbekistan) (both from teammate fact sheet)

### Inferences
- Well-cured ceramic coatings are strongly hydrophobic (commonly quoted around 100-115 degrees, background knowledge not verified here), i.e. in the range where Joninn is unsuitable.
- Wax, sealant and ceramic angle ranges overlap, and real-world error is about +/-2-3 degrees, so an absolute verdict cannot identify the product; comparing against the same car's baseline reading is more robust.
- A $4,990 lab instrument shows the measurement itself is not the product; the value for a detailer is a plate-linked certificate and workflow.
- The sponsors (Toyota Royal Motors, BYD Royal Motors) are dealers; dealer paint-protection add-ons are a plausible payer channel (my inference; local add-on practice not verified).

### Gaps
- Number of Tashkent detailing studios, coating prices and any coating-warranty disputes unknown (RU query refused); a second English search for coating-specific contact-angle apps was refused, so absence of one rests on a single query.

## service-12: Key-On (dashboard lamp sequence to tamper flag)

### Takeaway
No automated key-on lamp checker was found, but it is a feature, not a product: a human sees a dark lamp in seconds, a diagnostic scan reads stored SRS/ABS/engine fault codes even when a bulb has been removed (SRS and ABS need enhanced, not generic, OBD support), and airbag resistor emulators reproduce a normal lamp flash and also leave no fault, so the tool adds nothing against lazy tampering and gives false assurance against sophisticated tampering. Verdict: kill as a standalone, fold into a broader used-car check.

### Cited Findings
- English search found only manual advice: the airbag lamp normally stays on roughly 4-8 seconds when the ignition goes to ON, and a 2005 article warns that inexpensive devices can make the lamp flash normally although the airbag is gone (summary only; body not readable) (S) — [PropertyCasualty360](https://www.propertycasualty360.com/amp/2005/08/19/airbag-scams-know-the-warning-signals/); the only video mention is a paywalled lesson — [CarBuyingSupport](https://carbuyingsupport.com/video-library/how-to-inspect-the-dashboard-warning-lights/)
- Russian search returned no material on emulators or CV tools; a Kia manual says a lamp that does not light, or stays lit, needs a workshop check (S) — [Kia manual](https://ownersmanual.kia.com/full_webhelp/SK3/2025/ru_RU/20240516001538_images_xml/topics/chapter3_4_8.html)
- Used-car volumes: Oct 2025 57.4k secondary deals (-5% YoY) vs 38k new; Sep 2025 52.6k vs 31.9k new — [Kun.uz](https://kun.uz/en/news/2025/11/18/electric-car-purchases-surge-as-overall-auto-sales-recover-in-uzbekistan); [Anhor](https://anhor.uz/news/increase-15/)
- A Tashkent business advertises pre-purchase inspection "from 300,000 UZS" (mileage by scanner, body and paint, electrics, airbags, VIN) — [OLX.uz](https://www.olx.uz/d/oz/obyavlenie/ot-300-000-avtopodbor-proverka-avto-probega-pered-pokupkoj-avtotekshi-ID3XhPY.html)
- Teammate tech radar found no verifiable dataset for dashboard warning lights — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)

### Inferences
- Revenue ceiling: if 5% of about 52.6k monthly used deals bought the check at $0.50, that is about $1.3k per month (my arithmetic), against a $25 inspection whose advertised scope already lists airbags and a scanner mileage check.
- Against lazy tampering (bulb removed or taped) a scan shows stored fault codes; against a resistor emulator both the lamp video and the scan look clean, so only physical inspection of the airbag circuit catches it (background knowledge plus the 2005 article summary above).
- The ideas' three merged variants (LampTest, Lamp Proof, Key-On) show the concept is obvious to idea generators, but obviousness plus a dominated alternative (OBD) lowers the business case.

### Gaps
- Could not verify prevalence of lamp tampering in Uzbekistan, or whether factory end-of-line cluster camera testers are a direct precedent (telltale query refused).

## service-14: Wheel-Safe (wheel-fitting video to torque proof)

### Takeaway
The photo "loose-nut" half has close prior art (a granted Volvo Truck patent that compares a phone image with a reference image; witness-mark methods are older) and is also covered by cheap hardware, while the video "torque proof" half cannot establish a torque value and is not buildable in 2-3 days. Tyre shops also carry little liability that would make them pay. Verdict: kill.

### Cited Findings
- Volvo Truck Corp, US12333708 (priority 2019-12-05, granted 2025-06-17): a smartphone image of a disc wheel is compared with a reference image (air-valve or QR reference point) to find loose fasteners; background cites earlier art (JP 2009-210276) that relies on a mark on the fastener — [Google Patents](https://patents.google.com/patent/US12333708)
- Korean wheel-nut engagement checking patent for nut-runner positioning (S) — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10322509); factory vision guides nut runners (Cognex) — [Cognex](https://www.cognex.com/industries/automotive/chassis-systems/wheel-and-lug-nut-torquing); an AI-vision torque-check claim at wheel-build stations is vendor marketing — [iFactory](https://ifactoryapp.com/industries/automotive-manufacturing/tire-wheel-assembly-inspection-vision-ai)
- The search found no tyre-shop computer-vision product for lug-nut torque (summary conclusion) (S)
- Tashkent runs 1,707 buses on 170 routes (Sep 2026) — [Spot](https://www.spot.uz/ru/2026/09/04/university-transit/) (teammate fact sheet)

### Inferences
- A shop can show a torque wrench in video that is set wrong, and an impact gun can be followed by a wrench tap, so video evidence of tool type does not prove torque.
- Hardware nut-loosening indicators and logging torque wrenches cover the same risk cheaply (background knowledge; vendor page fetch failed, not verified).

### Gaps
- No wheel-off incident statistics for Uzbekistan; hardware-indicator and smart-torque-wrench market not verified; witness-mark follow-up search refused.

## service-15: Pit-Scan (pit-floor action cam underbody upsell sheet)

### Takeaway
AI underbody inspection for service upsell is UVeye's and ProovStation's category, so the "AI finds defects" claim is not novel and the CV is the hard part; the genuinely local angle is a $40 camera in the inspection pits common in small garages (my assumption) plus a Telegram customer sheet. Verdict: sharpen to per-VIN change detection.

### Cited Findings
- UVeye: drive-through underbody, tyre and exterior scanner; more than 1,000 systems; $191M raised on 29 Jan 2025 ($380.5M total); no public pricing — [uveye.com](https://uveye.com/); [Pulse 2.0](https://pulse2.com/uveye-ai-based-vehicle-inspection-technology-company-raises-191-million/amp/) (teammate saturation map)
- UVeye's own survey: nearly 60% of tyre checks are purely visual, only 6% of service departments rely solely on automated systems; automated underbody check under 90 seconds (search summary; page not read) (S) — [Wards](https://www.wardsauto.com/news/archive-wards-better-vehicle-inspection-systems-key-safer-highways/795235/)
- Skaivision service-lane computer vision, integrated into Hunter Quick Check Drive — [Motor.com](https://www.motor.com/2024/10/hunter-engineering-announces-integration-with-skaivision/) (teammate saturation map)
- Oxmaint markets robotic AI undercarriage scanners for heavy-vehicle fleets (vendor) (S) — [Oxmaint](https://oxmaint.com/industries/fleet-management/robotic-undercarriage-inspection-ai-vision); a patent describes first-pass vision inspection with deep learning only if the first result is poor (S) — [Justia](https://patents.justia.com/patent/20210398271)
- Open-source detectors for an open-vocabulary route: Grounding DINO, OWLv2 (Apache-2.0), Florence-2 (MIT) — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)

### Inferences
- A fixed pit camera sees the same view each visit, so change detection between visits needs no defect-class training data and is more honest than absolute defect detection from 40 cars.
- Underbody stitching from a rolling car is fragile (dark, textureless, repetitive); the demo should rely on human confirmation of flagged frames.

### Gaps
- Pit prevalence in Uzbek garages is my assumption (RU query refused); no local garage pricing found (teammates also found no service labour rates).

## service-16: Brake Peek (wheel-off brake audit at tyre swap)

### Takeaway
Photo-based pad measurement is claimed by a Nissan US patent (endoscope-based) and by no tyre-shop product I could find; the idea is among the more practical in this batch but over-claims rotor-lip depth and one-photo geometry. Verdict: sharpen.

### Cited Findings
- Nissan North America / Nissan Motor, US12117059B2 (filed 2022-05-25, granted 2024-10-15): a camera image of the brake pad assembly is used to determine pad dimension and compare it with a stored reference thickness; mobile device can run it; claim 11 uses a tangential view and an alignment bracket; the description uses a wireless endoscope; machine learning is not claimed — [Google Patents](https://patents.google.com/patent/US12117059)
- Bendix Brake Pad Identifier identifies a pad from a photo on a white background and returns a part number, not thickness (S) — [App Store](https://apps.apple.com/app/id843103884); RationalGo offers a pad-thickness logger that records readings (S) — [RationalGo](https://rationalgo.ai/resources/app-builder/automotive-brake-pad-thickness-measurement-logger)
- Vendor reference bands: new pads 10-12 mm front, 8-10 mm rear (vendor guidance, not a manufacturer spec) (S) — [Oxmaint](https://oxmaint.com/industries/fleet-management/brakes-pad-wear-checklist)
- Teammate map: brake-pad/rotor wear estimate through the wheel has only a vehicle-mounted-camera patent, rail systems and generic logging tools; nothing for passenger-car shops — [saturation_map.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/saturation_map.md)
- Chevrolet 83% of new sales (Jan-Sep 2025) — [Kun.uz](https://kun.uz/en/news/2025/10/11/uzbekistans-car-market-sees-growing-competition-as-chevrolet-share-drops); individually owned vehicles 4,903.1k on 1 Apr 2026 (+344.6k YoY) — [Statistics Committee](https://stat.uz/ru/press-tsentr/novosti-goskomstata/68807-zbekistonda-a-oliga-tegishli-avtotransport-vositalari-soni-4-9-mlntadan-oshdi-3)

### Inferences
- Pad thickness is seen from the tangential gap, not head-on, so "one photo with a scale card" needs guided capture geometry; rotor-lip depth of about 0.5 mm is below what a phone photo measures reliably.
- The US patent is territorial; it matters for later expansion, not for an Uzbek pilot.
- The idea's own "$40-120 per brake job" is an estimate; Uzbek labour and parts prices were not found, so the upsell value is unverified.

### Gaps
- Second English/RU searches for existing brake-photo apps were refused; no local service price list; tyre-swap seasonality in Uzbekistan unknown.

## service-17: Glare-Gauge (windscreen night-glare score)

### Takeaway
No phone product exists, but there is also no standard, no regulator and no payer, and the measurement is confounded by the phone's own optics and dirty glass. Verdict: kill.

### Cited Findings
- US 6115118 (1997): digital imaging of windshield zones, pitting expressed as density per square inch, notes peripheral glare from pits (search summary; page 403) (S) — [Justia](https://patents.justia.com/patent/6115118)
- A 1985 lab method measured light scatter of a localised beam through the windshield (stray-to-direct ratio) and needed specialised equipment (S); SWOV describes hazing from salt, sand and pebbles causing glare, possibly contributing to night-time accidents (S) — [SWOV](https://swov.nl/en/publicatie/vision-impairing-wear-windshields)
- US11100616B2 "Optical surface degradation detection and remediation" (S) — [Google Patents](https://patents.google.com/patent/US11100616); dashcam forum anecdotes tie radiating streaks to wiper micro-scratches (S) — [DashcamTalk](https://dashcamtalk.com/forum/threads/a119s-glare-from-light-sources.28195/post-326641)
- Windshield repair-vs-replace triage is contested by Belron-linked tools and DriveX (teammate saturation map) — [DriveX](https://drivex.io/?p=412)

### Inferences
- Pitted glass is rarely a replacement trigger, and nobody pays for a "glare certificate"; mobile browsers give limited exposure control, so HDR bracketing needs a native camera plus upload.

### Gaps
- No local windshield replacement price or insurer policy found; only one search ran for this idea.

## service-18: Dent-Lines (stripe-reflection dent triage for PDR)

### Takeaway
Stripe-reflection dent detection is established in PDR practice and in gantry products, and a 2015 "Dent Light" app reportedly used screen reflections for hail dents; laptop-screen depth triage is a modest twist. The stronger angle is gauge-free repaint and filler mapping for used cars, where teammates found no commercial phone product. Verdict: sharpen.

### Cited Findings
- Autoscan GmbH, US10976262B2 (priority 2017-01-11, granted 2021-04-13): a drive-through arch projects a grid pattern and cameras track its reflection; dents bend the lines; size and position are measured, depth is not mentioned — [Google Patents](https://patents.google.com/patent/US10976262)
- PDR practice: reflection boards show breaks and kinks in lines; a commercial "DellenChecker" reflector exists (S) — [PDR training](https://onlinepdrtraining.thedingking.com/paintless-dent-repair/module-b-explanation-of-lights-and-reflections/); [DellenChecker](https://meinhartje.com/en/busching-dent-repair-set-dellenchecker/89655609)
- A 2015 report describes a "Dent Light" app to measure, photograph and repair hail-size dents using multi-coloured reflection images (details unverified; page 403) (S) — [Collision Repair](https://collisionrepairmag.com/?p=7024)
- University of Maribor built a gate-type deflectometry dent detector (S) — [FERI](https://gemma.feri.um.si/projects/industrial-projects/device-for-damage-detection/)
- Teammates: no commercial product claims repaint detection from ordinary phone RGB photos; Auto.ru flags "poor-quality repairs" in listing photos (trained on more than 100,000 specialist-inspected cars) — [Snob.ru](https://snob.ru/news/avto-ru-zapustil-neiroset-dlia-otsenki-avtomobilia-po-foto/)
- Sep 2025: 52.6k secondary-market deals vs 31.9k new — [Anhor](https://anhor.uz/news/increase-15/)

### Inferences
- Manual inspectors already check panel reflections for waviness; a striped screen makes that quantitative, which is why a repaint and filler map is plausible but unproven (needs a falsification test against a paint-thickness gauge).
- Depth in millimetres needs screen-to-panel geometry; body lines mimic dents; so a relative index is the honest output.

### Gaps
- Willomitzer et al. mobile deflectometry could not be opened (Optica placeholder); PDR market size and prices in Uzbekistan unknown; Dent Light follow-up search refused.

## evcng-8: Ko'pik (soap-bubble leak-test witness for gas conversions)

### Takeaway
The bubble-area-to-leak CV technique is patented (Duke Energy) and sold in industrial form (Roboflow), but the vehicle CNG/LPG joint-coverage witness record is not, and the local fit is the strongest in this batch: a third of used listings are petrol-gas and Cabinet Resolution 315 of 18 Jun 2026 is wiring cylinder status into U-Gas. The resolution says nothing about leak tests or conversion workshops, so the payer hook still has to be proven. Verdict: keep.

### Cited Findings
- Duke Energy Corp, US12366499B2 (priority 2022-02-22, granted 2025-07-22): apply soap solution, capture sequential images with a portable device camera, a trained CV model finds total bubble area, a second model converts it to flow rate using component size and gas pressure; dependent claims cover 10+ second video, on-device processing, pipes, valves and fittings, natural gas, propane and refrigerant; aimed at utility leaks below about 1 SCFH — [Google Patents](https://patents.google.com/patent/US12366499)
- Roboflow "Pressure Test Leak Detection AI": RF-DETR trained on your own tank, lighting and parts; tracks bubbles across frames to separate leak streams from ambient bubbles, maps the origin to a location, writes pass/fail; also covers soap-solution bubbles on fittings; no accuracy numbers, no vehicle use case — [Roboflow](https://roboflow.com/ai/pressure-test-leak-detection)
- A 2026 Chinese paper reports a lightweight YOLOv11 bubble segmenter with 95.3% detection accuracy and 0.19/0.30 mm mean/max error; another Chinese paper uses five-frame differencing for soap-bubble airtightness (summaries only; URL-to-paper mapping uncertain) (S) — [SciOpen](https://www.sciopen.com/article/10.16791/j.cnki.sjg.2026.07.015); [CPS Journals](https://www.cpsjournals.cn/data/article/zkkxyjsxb/preview/pdf/10.13922/j.cnki.cjvst.202401019.pdf)
- Cabinet Resolution No. 315 of 18 Jun 2026 (in force 22 Jun 2026): stations may not fill vehicles overdue for periodic inspection or with untested or past-life cylinders (Annex 6, section 2(b), sub-clause "z"); MVD with the Digital Ministry must integrate cylinder-inspection data with Eco-Himoya and U-Gas by 1 Sep 2026 (point 4(v)); U-Gas is mandatory for CNG compressor stations (clause 6(a)); no provisions on leak testing or conversion workshops; no fines specified — [Lex.uz](https://lex.uz/ru/docs/8280849)
- 2025 drive: 6,196 fuel facilities inventoried, 46,737 violations and defects, 464 stations suspended, 7,166 defects fixed on site; the article gives no count of vehicle cylinders and no rule requiring workshops to record leak tests — [Spot](https://www.spot.uz/ru/2026/06/28/gas-station-safety/)
- 18 Aug 2022 Syrdarya: a bus cylinder exploded during methane refuelling, 6 killed, 15 injured — [Podrobno](https://podrobno.uz/cat/proisshestviya/v-avtobuse-v-syrdare-vzorvalsya-gazovyy-ballon-shest-chelovek-pogibli-15-postradali-/); 18 Sep 2025 Fergana propane cylinder filled with methane by mistake exploded — [Podrobno](https://podrobno.uz/cat/proisshestviya/oshibka-voditelya-privela-k-vzryvu-na-avtozapravke-v-fergane/); 5 Sep 2026 Andijan Spark cylinder exploded at a CNG station — [Podrobno](https://podrobno.uz/cat/proisshestviya/na-agnks-v-andizhane-vzorvalsya-gazovyy-ballon-avtomobilya-spark/)
- Petrol-gas is 9,965 of 29,403 fuel-typed Avtoelon car listings (33.9%, live read by teammate) — [avtoelon.uz](https://avtoelon.uz/); about 11% of cars run on LPG — [Anhor](https://anhor.uz/news/price-of/)
- SAM 3 is under the Meta SAM License (commercial use with conditions); SAM 2.1 is Apache-2.0 — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)

### Inferences
- The regulatory rail (cylinder status checked at the pump) exists, but a leak-test attestation is a missing link, not a mandate; workshops will pay only if a station, insurer or regulator asks for it.
- Staged air micro-leaks are not CNG; claim "test witnessed" and a bounded minimum detectable leak, not a safety certification.

### Gaps
- Russian query on soap-test practice in Uzbek conversion workshops was refused; no workshop count, no insurer stance; Duke patent is US-only (freedom to operate matters only for US expansion).

## wild-1: TremorLab (phone-video engine and wheel vibration diagnostics)

### Takeaway
Phone-based motion amplification is sold for industrial machines (KCF Motion Insights, Erbessd DragonVision) but I found nothing for automotive engine diagnosis. The magnified-video demo is a crowd pleaser; the diagnostic claims (worn mount, misfire, bent rim) cannot be validated in 2-3 days and overlap with a cheap OBD-II read. Verdict: sharpen to a marker-based engine-rock measurement.

### Cited Findings
- KCF Motion Insights: phone camera, "No new hardware required", contrasted with a "$30,000 camera"; about 30 seconds processing; claims handheld-shake stabilisation and low-light handling without detail; automotive appears only as the setting of the founder's pilot visits — [KCF](https://kcftech.com/resources/blog/machine-health-solutions/why-we-built-motion-insights-the-story-behind-the-camera-that-was-already-in-your-pocket/)
- Erbessd DragonVision: motion-magnification software on Windows from ordinary video, industrial use cases, no automotive mention, price on request — [Erbessd](https://new.erbessd-instruments.com/en/products/dragon-vision)
- WaveCam analyses smartphone video; one article notes detectable frequencies are limited by frame rate (S) — [Turbomachinery Magazine](https://www.turbomachinerymag.com/video-processing-of-vibration-measurements)
- A 2026 engine-vibration paper reports at least 86% state-classification accuracy, validated with a GoPro and a conventional sensor, not a phone alone (source identity uncertain; page 403) (S) — [CNR IRIS](https://iris.cnr.it/handle/20.500.14243/404741)
- Teammates found no actively maintained production motion-magnification tool; open repos are research code — [tech_radar.md](file:///Users/diyorbekumaraliyev/Projects/wiutHackathon26/research_notes/Automotive%20CV%20hackathon%20ideas/tech_radar.md)
- Modal L4 costs $0.000222 per second (scale-to-zero, $30 monthly free credit) — [Modal](https://modal.com/pricing)
- Petrol-gas is 33.9% of fuel-typed Avtoelon listings — [avtoelon.uz](https://avtoelon.uz/)

### Inferences
- Cost per 8-second clip is about $0.002 on an L4 (my arithmetic); handheld physiological tremor (about 8-12 Hz, background knowledge) overlaps the idle crank order (about 13 Hz at 800 rpm), so a tripod and a body reference are required.
- Large engine rock under a throttle blip is the one worn-mount symptom visible without magnification, and a fiducial tag gives millimetre displacement directly.

### Gaps
- No labelled fault clips exist; engine-mount and misfire searches were refused; RDI Technologies not checked; rolling-shutter limits not sourced.
