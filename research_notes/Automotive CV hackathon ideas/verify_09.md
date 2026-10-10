# Due diligence batch 09: driver and rider safety ideas (fleet-17, fleet-18, safety-1, safety-2, safety-3, safety-4, safety-5, safety-10, safety-15, safety-16)

> Research date: 2026-10-10 (hackathon finals, Day 1). Method: 18 WebSearch queries returned results (11 extended, 7 standard; one of the extended calls ran two internal searches) plus about 50 direct page fetches and local PDF text extraction. The session-wide WebSearch quota ran out repeatedly: about 15 attempts were REFUSED, a few were re-run later with different wording, and the topics that stayed uncovered are listed under each idea's Gaps (notably: a Chinese-language probe for camera-scored driving ranges, glad-hand / trailer-inspection AI, Uzbek extinguisher-kit and child-seat rules in Russian, number of Uzbek driving schools, Yandex Go child tariff). Per-idea first-pass queries: coupling verification; extinguisher gauge/expiry OCR; learner mirror-check head pose; child-seat installation CV; courier helmet check; phone-flash retroreflective clothing; webcam hazard perception; driving-school range camera scoring; motorcycle road-surface warning; child backover AI; plus Russian/English follow-ups on the Uzbek exam, courier helmets, ADTT tracks, extinguisher apps and phone retroreflectivity. Tag (S) = taken from a search-result summary, page itself not read (403, binary, paywall or not fetched): lower confidence. Untagged = page fetched and read. Vendor figures are vendor claims. FX 11,800 UZS per USD (teammate fact sheet). "Inferences" are my judgment. Scores are 1-5 as in the brief.
>
> Local files used (not URLs): `tech_radar.md`, `uzbekistan_market.md`, `saturation_map.md`, `hackathon_intel.md` in the same folder.

## fleet-17 CouplingCheck: semi-trailer hitch verification (photo of five hitch points)

### Takeaway
No computer-vision coupling verifier turned up in one extended query; today's art is sensor hardware (for example Fontaine's vibration-signature patent). But the headline check, "kingpin lock", is under the trailer and cannot be shown from a side photo, "chains" are not part of a standard fifth-wheel procedure, and no Uzbek or Central Asian trailer-separation statistics exist to prove payback. Novel but hard to demo and hard to monetise.

### Cited Findings
- Extended query "semi-trailer coupling verification computer vision fifth wheel kingpin lock air hoses camera AI pre-trip inspection" returned manual procedures and patents only: image sensor in the hitch region (low light makes detection hard), a vibration sensor, a magnetic marker on the kingpin; no production camera-based pre-trip coupling system, no CV for glad-hand or air-hose connection (S) — [HVI fifth wheel guide](https://heavyvehicleinspection.com/blog/post/fifth-wheel-kingpin-inspection-coupling-checks); [Wikipedia: Fifth-wheel coupling](https://en.wikipedia.org/wiki/Fifth-wheel_coupling); [US 11731471](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11731471)
- Fontaine Fifth Wheel Co. patent US 11,731,471 B2 (priority 27 Apr 2018, filed 25 Apr 2019, granted 22 Aug 2023): a sensor picks up the vibration made when a fifth wheel couples to a kingpin and a controller checks for a predetermined vibration component; accelerometer, piezo, inductive/magnetic and inertial-switch variants; no cameras or image processing (first 100k characters of the page read) — [Google Patents](https://patents.google.com/patent/US11731471B2/en)
- Georgia CDL coupling steps: safety unlocking handle in automatic lock; connect emergency and service air lines to glad hands; build air, lock trailer brakes; "Raise the landing gear slightly and pull forward gently" to confirm lock; "no space between upper and lower fifth wheel"; "jaws have closed around the shank of the kingpin"; "safety latch is in position over locking lever"; plug the electrical cord; "Raise the landing gear all the way up". The page mentions no safety chains for semitrailers — [Georgia DDS Section 6.3](https://dds.georgia.gov/section-63)
- A commercial inspection guide lists "glad hands sealed, electrical connected, safety chains secured", lists "missing or disconnected safety chains" as out of service, says the locking jaws need you to "get under with a flashlight and visually confirm", and says "A visual pass alone isn't enough" because the tug test proves the coupling — [HVI checklist](https://heavyvehicleinspection.com/blog/post/tractor-trailer-fifth-wheel-coupling-inspection-checklist) (commercial blog, as summarised by the fetch tool); the chains item is not in the Georgia procedure above — [Georgia DDS](https://dds.georgia.gov/section-63)
- Model stack: RF-DETR Nano is Apache-2.0 with ONNX and WebGPU builds; SAM 3 is under Meta's SAM License (commercial use allowed with conditions) and its Hugging Face weights are gated — local file `tech_radar.md` sections 1-2 ([RF-DETR](https://github.com/roboflow/rf-detr), [SAM 3 licence](https://github.com/facebookresearch/sam3/blob/main/LICENSE))
- Uzbek context: 330.3k individually owned trucks on 1 Apr 2026 (statistics committee, via `uzbekistan_market.md`); border queues of hundreds of trucks (Nov 2024, Tazhen crossing) — [Kun.uz](https://kun.uz/en/news/2024/11/26/hundreds-of-trucks-stranded-at-kazakhstan-uzbekistan-border-amid-increased-traffic)

### Inferences
- Of the five items, four can be photographed from the side or rear (glad-hands coupled versus parked on dummy couplers, plug seated, landing gear raised, release handle and safety latch plus the zero-gap line). The kingpin jaw state itself is under the trailer, so "AI confirms kingpin lock" would be a proxy, not verification; a wrong "go" on a coupling check is the worst failure mode.
- "Chains" apply to drawbar or pintle trailers, not to a standard fifth-wheel semi; keeping them in the pitch invites an expert judge to discredit the domain knowledge.
- Frequency of detachments is unknown locally; severity is high but the buyer cannot compute payback, and the legal-grade check remains the manual tug test. Cargo insurers or forwarders on the China to Central Asia corridor are a more plausible payer than $6 per tractor per month (about 71k UZS).
- Demo: a judge cannot hitch a truck; toy semi or uploaded photos only, and five photos per check is slow on stage.
- Scores: novelty 4 (nothing found, but only one extended query plus 2 blocked), feasibility 3, demo 2, business 2, moat 2, local fit 3, verdict sharpen.

### Gaps
- Blocked by quota: "glad hand detection camera trailer air line connection AI yard automation" and "trailer pre-trip inspection AI camera app photos fifth wheel glad hands landing gear startup". Telematics vendors (Samsara, Motive, Lytx) may already offer photo-DVIR AI; not checked.
- No trailer-separation or coupling-failure statistics for Uzbekistan, Kazakhstan or the corridor; no Tashkent truck-yard access arrangement.
- Whether Shacman, Howo and FAW trailers use any chains: not verified.

## fleet-18 KitCheck: mandatory-equipment audit with expiry clocks

### Takeaway
Extinguisher inspection by computer vision already exists as an academic prototype app (Weqaa), a published dataset and detector (Alayed et al. 2024) and vendor marketing (Overview.ai); the paper explicitly could not read the expiry date, which is the one gap KitCheck fills. Easy and very demoable, but at about $1 per vehicle per month with no moat and no verified Uzbek kit rule it is a feature, not a business.

### Cited Findings
- Alayed et al., "Real-Time Inspection of Fire Safety Equipment using Computer Vision and Deep Learning" (ETASR, Feb 2024, CC-BY 4.0): 7,633 images with 16,092 labelled instances; six classes hose_exist, pin_exist, rust, gauge_good, gauge_bad, expire_date; YOLOv8n best with mAP0.5 87.2% and 2.7 ms inference; per-class AP: expire_date 81.4, gauge_good 84.3, gauge_bad 88.9 (precision 67.9); gauge errors rise when the gauge is far, dark, behind the extinguisher or covered by the tag; the system detects the presence of an expiry label but "cannot determine whether the extinguisher has expired" because Python libraries could not extract Arabic text (PDF text extracted locally) — [DOI 10.48084/etasr.6753](https://doi.org/10.48084/etasr.6753)
- Follow-up mobile app: "Weqaa: An Intelligent Mobile Application for Real-Time Inspection of Fire Safety Equipment" (ETASR vol. 14 no. 3, Jun 2024), Flutter app integrated with the detection model, inspectors reported faster and fewer errors; no accuracy figures given — [ETASR 7229](https://www.etasr.com/index.php/ETASR/article/view/7229)
- Vendor marketing: Overview.ai walkthrough claims "Reads analog pressure gauges; confirms needle within green band", "OCR of inspection tags", hydrostatic-test date reading, fixed or mobile deployment; no customers, no accuracy numbers, "ask us about advanced inspection packages" — [Overview.ai](https://www.overview.ai/resources/walkthroughs/fire-extinguisher-visual-inspection/)
- 2017 IEEE paper segments the gauge's green band and computes needle angle (small-scale) (S) — [IEEE 7883695](https://ieeexplore.ieee.org/document/7883695)
- GitHub topic fire-extinguisher has 7 repos, none about gauge reading or inspection — [GitHub topic](https://github.com/topics/fire-extinguisher)
- Non-AI extinguisher inspection apps exist (TracerPlus, InspectExtinguishers) (S) — [TracerPlus](https://www.tracerplus.com/mobile-app/mobile-inspections/fire-inspection/)
- PP-OCRv6 (Apache-2.0) HF tags list English and Chinese only, so Cyrillic or Uzbek words are unverified (digits are fine); RF-DETR Nano Apache-2.0 — local file `tech_radar.md` ([PP-OCRv6](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det))
- Inspection regime: since 1 Sep 2022 technical inspection is "exclusively voluntary" for individually owned passenger cars under 10 years old, 10-14 years every two years, over 15 years annually, and "the schedule does not apply to taxis" — [Spot, 29 Jun 2022](https://www.spot.uz/ru/2022/06/29/tint-decree/); Tashkent 2021 schedule: passenger-transport companies are inspected twice a year, legal-entity vehicles 1 Mar-30 Jun, 7,099 companies listed — [Spot, 17 Feb 2021](https://www.spot.uz/ru/2021/02/17/inspection/). Neither page lists first-aid kit, extinguisher or triangle requirements.
- Yandex Go works only through fleets (135 fleets in Mar 2023) — local file `uzbekistan_market.md` section 6 ([KSE thesis](https://kse.ua/wp-content/uploads/2025/09/Oleh_Dykyi_RIDE-HAILING-MARKET-IN-UZBEKISTAN.pdf))

### Inferences
- Novelty is 2: detector classes, expiry-label detection and a phone app all exist; only Latin/Cyrillic date OCR, car-specific kit items (first-aid kit, triangle) and expiry reminders are new, and they are thin.
- Money: $1 per vehicle per month is about 11.8k UZS; a 100-vehicle fleet pays about $100 a month. No fine amount or inspection rule for missing kit was found, so there is no proof it pays off; the only structural payer is passenger carriers still on mandatory twice-yearly inspection.
- Demo is the strongest part: a judge can point a phone at any extinguisher in the building.
- Scores: novelty 2, feasibility 4, demo 5, business 2, moat 1, local fit 3, verdict sharpen (module of a wider pre-shift compliance product, not standalone).

### Gaps
- Uzbek rule requiring extinguisher, first-aid kit and warning triangle (and fines) not verified; the Russian query was blocked by quota.
- Alayed dataset availability and licence (public download?) not checked; Overview.ai's product reality unverified.
- Taxi fleet sizes in Uzbekistan beyond the 135-fleet figure not found.

## safety-1 MirrorMark: phone-scored mock driving exams

### Takeaway
The concept exists twice: Microsoft Research India's HAMS is a windshield-phone automated licence-test system that checks seat belt and whether the driver checks mirrors before a lane change (piloted in Dehradun, 2019), and a UK app (Clutch) scores mirror checks from phone cameras against DVSA standards. Worse for the pitch, the real Uzbek practical exam is a sensor-scored range test whose published description never mentions mirrors, signals or belts, so mirror-check scoring cannot "predict pass risk". The defensible remainder is on-road coaching for driving schools.

### Cited Findings
- HAMS (Microsoft Research India): ACM MobiCom 2018 demo of a windshield-mounted Android phone whose front camera monitors the driver ("drowsiness, distraction and gaze") and back camera monitors driving ("vehicle ranging, lane detection") — [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/hams-driver-and-driving-monitoring-using-a-smartphone/); the project docs describe an "automated driver license test" on a windshield-mounted smartphone that can "verify the test-taker's identity, confirm seat belt use, and check whether the driver checks mirrors before a lane change", contrasted with pole-mounted high-resolution cameras on earlier instrumented tracks — [HAMS docs](https://microsoft.github.io/HAMS/); code is MIT-licensed in the "Automated License Testing" repo, archived 11 Jun 2026, 8 stars — [GitHub](https://github.com/microsoft/HAMS); Nov 2019 report: tested with Dehradun's regional transport office, more than 50 people a day attempt the test with HAMS devices and only about half pass — [Tproger](https://tproger.ru/news/hams-msr-india)
- Clutch (learnwithclutch.com): "tracks your mirrors, speed, and driving smoothness", mockup labels "Head Tracking" and "Mirror Detection", mirror checks "scored against DVSA competency standards", free to download, UK App Store and Google Play; the blog says the front camera must see "you" and the rear camera the road, phone mounted high and central by the rear-view mirror; "Parent & instructor progress sharing"; link "Try Clutch for instructors"; 31 blog posts dated Jun-Sep 2026; launch date, company, pricing, accuracy and users not stated; no shoulder-check, signal, belt or pass-risk claim found — [Clutch home](https://www.learnwithclutch.com/); [About](https://www.learnwithclutch.com/about); [phone mounts blog](https://www.learnwithclutch.com/blog/best-phone-mounts-for-driving-lessons); [driving-school blog](https://www.learnwithclutch.com/blog/how-to-grow-your-driving-school-2026); [sitemap](https://www.learnwithclutch.com/sitemap.xml)
- Extended query "learner driver mirror check shoulder check head pose detection AI app driving lesson scoring phone camera" found no peer-reviewed accuracy for phone mirror/shoulder-check detection (S) — [Ultralytics DMS blog](https://www.ultralytics.com/blog/vision-ai-powers-driver-attention-monitoring-systems)
- Uzbek practical exam (Kun.uz, 10 Dec 2025): takes place on "the center's driving range"; five manoeuvres (slalom "snake", ramp start, parallel parking, turning in a confined space, stopping at a designated line); 100-point scale with penalties, "if the total deductions exceed 99 points" the exam is failed; example "stopping too close to a pedestrian crossing" -5; "All examination vehicles are equipped with sensors that detect violations in real time... trajectory, distance, and lane markings"; mirrors, turn signals and seat belts "None of these are mentioned"; "practical exam results cannot be appealed"; retake after seven days at 1.2M UZS; practical fee 1M UZS (same day as theory) or 1.2M; 40-50 practical hours; extra instructor-led practice from 100,000 UZS per hour, 200-250k for newer cars; full course 6-7M UZS — [Kun.uz](https://kun.uz/en/news/2025/12/10/how-to-obtain-a-drivers-license-in-uzbekistan)
- Driving-school rating from 1 Sep 2022: up to 30 points for graduates' exam results, 30 for graduates' penalty points in the first year, 30 for crashes in which graduates were found guilty, 10 for qualified teaching staff; green 81-100, yellow 51-80, red 50 or less — [Spot, 6 Jul 2022](https://www.spot.uz/ru/2022/07/06/driver-schools/)
- About 500,000 people receive driving licences each year, 25% of them skip classes (2022) — [Spot, 11 Feb 2022](https://www.spot.uz/ru/2022/02/11/driving-schools/)
- From 1 Feb 2026 theory classes are optional for categories A and B (practical exam unchanged) — [Kun.uz](https://kun.uz/en/news/2025/10/27/uzbekistan-to-make-theory-classes-optional-for-a-and-b-drivers-licenses-starting-feb-2026); [UzDaily](https://www.uzdaily.uz/en/uzbekistan-introduces-new-driving-license-rules-from-2026/)
- MediaPipe Face Landmarker: Apache-2.0, 478 landmarks, facial transformation matrix (head pose), JavaScript; head-turn rules need no training — local file `tech_radar.md` ([docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker))

### Inferences
- The idea's "city checks" and "mirror-signal-shoulder window" have no counterpart in the exam as described, so "pass-risk before the 1.2M-soum retake" is unsupported; the real exam rewards trajectory, distance and line discipline, which is what safety-10 targets.
- What survives: on-road lessons (40-50 practical hours) where mirror scanning matters for safety and for the school rating (60 of 100 rating points depend on graduates' later penalty points and crash guilt), but that signal lags a year.
- Numbers: 150k UZS per car per month is about $12.7; 30k UZS per report is about $2.5 against a 1.2M UZS (about $102) retake. Rough scenario (assumption): if 5% of 500k annual licence applicants paid 30k UZS that is about 750M UZS (about $64k) a year.
- Because HAMS (MIT code, Microsoft Research) and Clutch both exist, the technique is validated but the idea is a re-implementation; the only local differentiators are Uzbek/Russian feedback, the Uzbek rubric and a school-rating link. Best use is as the on-road lesson module of safety-10, sold to the same schools.
- Scores: novelty 2 (HAMS plus Clutch; nothing found for CIS), feasibility 4, demo 4, business 2, moat 2, local fit 3, verdict sharpen.

### Gaps
- Whether the exam also scores seat belt or signals is not in the Kun.uz guide (rule text not read); the "30 cm hill-start rollback" limit is not on that page either.
- A Russian-language query on driving-school analysis apps found only Russian school scheduling/theory apps and a St Petersburg drowsiness app (Drive Safely), none scoring mirror checks (S) — [Tproger/HAMS item above](https://tproger.ru/news/hams-msr-india); a Chinese-language check was blocked by quota.
- Clutch pricing, user numbers and accuracy are unknown; HAMS accuracy figures for mirror-check detection were not on the pages read.

## safety-2 SeatProof: photo-verified child seats for taxi rides

### Takeaway
No photo-based or AI child-seat installation checker was found (only in-person technicians and a hardware chest-clip sensor), so the novelty is real. But the decisive safety check (tightness) is physical, a false "pass" carries liability, child seats are not currently mandated in Uzbekistan (a June 2026 draft is not adopted), and the payer amounts are tiny.

### Cited Findings
- Extended query "child car seat installation check computer vision AI photo harness chest clip misuse detection app": no evidence of an AI or CV app that checks installation from photos; results were sensor hardware, occupant-detection patents and guides; installation criteria include under 1 inch movement at the belt path, harness pinch test, chest clip at armpit level, connected top tether (S) — [Evenflo SensorSafe](https://www.evenflo.com/pages/sensorssafe-faq); [Bambi Baby guide](https://www.bambibaby.com/blogs/learning-center/how-to-tell-if-your-car-seat-is-installed-correctly) (secondary, cites "46.7% of installations too loose" from NHTSA, unverified)
- Evenflo/CYBEX SensorSafe: Bluetooth chest clip alerts if the clip is unbuckled while moving, cabin above 95F or below 45F, seated over 2 hours, or child left buckled; works only with certain Evenflo Gold and CYBEX seats; the page says nothing about checking installation or tightness — [Evenflo FAQ](https://www.evenflo.com/pages/sensorssafe-faq)
- Misuse types listed: loose installation, incompatible belt, incorrectly fastened harness, bulky winter coat under the harness, wrong recline; installation is checked by certified technicians and police inspection sites; no camera or app-based checks mentioned — [Wikipedia: Child safety seat](https://en.wikipedia.org/wiki/Child_safety_seat)
- Uzbek seat-belt rule: front passengers must buckle; exemption for "children under 12 on the rear seat"; the article mentions no child restraints (4 Oct 2023) — [Anhor](https://anhor.uz/news/belt/)
- Prosecutor General's draft Article 128-11 (19 Jun 2026, public discussion, not adopted): unsafe transport of children 30 BRV (12.36M UZS) plus a licence ban of 1-3 years; 50 BRV (20.6M UZS) or up to 15 days arrest for unlicensed drivers or repeats within a year; the article does not say what the current seat rule is — [Podrobno](https://podrobno.uz/cat/obchestvo/v-uzbekistane-namereny-uzhestochit-otvetstvennost-za-narushenie-pravil-perevozki-detey/)
- Russia: Federal Law 580-FZ (in force 29 Dec 2025) obliges carriers to provide matching certified child seats and lets aggregators route such orders only to equipped cars — local file `saturation_map.md` section 3A ([PDD Master](https://pddmaster.ru/novoe/kresla-1225.html))
- Yandex photo control (Habr, 2018) covers exteriors (dark-photo filter, plate OCR, make/model/colour, damage/dirt) — [Habr](https://habr.com/post/433386)
- Yandex Go's "Детский" (child) tariff in Russia (article of 25 Jul 2022): Yandex Go subsidised partner fleets' purchase of about 1,000 RANT Matrix Safety Line seats; only top-rated drivers take children, and they get extra training in installing and adjusting the seats and helping parents buckle the child; the article does not say anyone checks the installation — [Moskvich Mag](https://moskvichmag.ru/?p=244936). A Russian search found no evidence of a separate child tariff with seats in Tashkent; Yandex Go operates in Tashkent and about a dozen other Uzbek cities through YandexGo UB (S) — [Inc. Russia](https://incrussia.ru/news/yandeks-taksi-s-lyulkoj/)
- Stack: SAM 3 (gated weights, SAM License), RF-DETR Nano and Qwen3.5-4B (Apache-2.0), Qwen-Image-Edit-2511 (Apache-2.0) — local file `tech_radar.md`

### Inferences
- Photos can plausibly check seat type and direction versus a declared child age, harness routing and twist, chest-clip height, visible belt path; they cannot check tightness, recline accuracy under load, or a child's real shoulder-to-harness fit on a doll. A "pass" label should therefore read "no visible errors; tightness not verified".
- Gating a ride on that label shifts liability to the aggregator, which has little reason to accept it before the law passes. No Uzbek "child tariff" was found.
- Money: 500 UZS per verified ride is about $0.04 and 5k UZS per parent check about $0.42; unit cost is low (about $0.0003 per VLM call), but revenue per user is also tiny.
- Scores: novelty 4, feasibility 3, demo 3 (props at the booth), business 2, moat 3 (misinstallation library, seat-to-car compatibility), local fit 3, verdict sharpen.

### Gaps
- Uzbek child-restraint law currently in force, Yandex Go / inDrive child options in Uzbekistan, and any local misuse statistics: queries blocked by quota (Russian, Uzbek rule).
- No public dataset of correct versus misinstalled child seats was found.

## safety-3 ShiftGate: courier gear check before orders

### Takeaway
The helmet-check-before-you-ride pattern is already shipped by micromobility operators and is a stated plan at Meituan, and helmet detection is a stock hackathon project. The distinctive part is the non-helmet checks keyed to Uzbek moped fines, and the local timing is excellent (moped plates since Oct 2025, courier raids, Yandex quoting about 4.7bn UZS of courier fines). But no Uzbek helmet rule was found, a one-off pre-shift check is gameable, and couriers are scarce so platforms may not want to block shifts.

### Cited Findings
- Bird "Helmet Selfie" (end-of-trip selfie, AI check, 25p discount) and Voi (helmet-selfie loyalty) — [Zag Daily, 31 Oct 2020](https://zagdaily.com/tech/using-ai-to-check-helmet-use-and-rider-impairment/); Voi reports 95% classifier accuracy and Helbiz's HelmetChecker lets operators block unlocking until a helmet is confirmed (99.7% accuracy, company claim) (S) — [Traffic Technology Today](https://www.traffictechnologytoday.com/news/machine-vision-alpr/voi-rewards-riders-who-take-helmet-selfie-to-encourage-safer-e-scooter-riding.html); [Business Wire, 13 Jul 2022](https://www.businesswire.com/news/home/20220713005567/en/Helbiz-to-Offer-its-AI-Helmet-Selfie-Technology) (both returned 403 to the fetch tool; content from the search summary)
- Meituan (30 Dec 2024): "Meituan will enforce stricter compliance with traffic regulations", steps include "mandatory helmet checks, pop-up safety alerts, and order acceptance restrictions for non-compliance", described as plans in response to a Chinese algorithm-regulation initiative — [ChinaTravelNews](https://www.chinatravelnews.com/article/185420/); smart-helmet programmes at Meituan, JD, Ele.me (Aug 2026) — [TNW](https://thenextweb.com/news/china-ai-smart-helmets-delivery-riders-jd-meituan-price-war)
- Yandex Eats (Russia) monitors bike-courier speed and warns in the app, working in all its cities; no helmet photo check found (S) — [Inc. Russia](https://incrussia.ru/news/yandeks-eda-sledit/)
- Stock-project evidence: DeliverGuard GitHub pipeline checks rider photos for helmet compliance (S) — [GitHub](https://github.com/msazunayed/Smart-Delivery-Verification); SmartClick sells a helmet-detection API (S) — [SmartClick](https://smartclick.ai/api/helmet-detection/); helmet-detection topic has 88 repos — local file `saturation_map.md` section 2I ([GitHub](https://github.com/topics/helmet-detection))
- Spot, 6 Oct 2026: Yandex Uzbekistan (VP quoted) counts 23,785 violations by motorcycle, scooter and moped riders in 2026, about 4.7bn UZS of courier fines excluding impound costs (about 198k UZS per violation, Spot's own division); only 1,890 (about 8%) were missing documents, the rest "included missing reflectors or headlight, window tinting and carrying passengers"; SBDD counts 4,312 violations and 563 crashes with 89 deaths among moped and scooter riders (nine months of 2026); moped plates since 1 Oct 2025; the article does not mention helmets — [Spot](https://spot.uz/ru/2026/10/06/couriers)
- Helmets are not mentioned in Kun.uz's moped/scooter licensing guide (3 Nov 2025), the economist.kg report on the 2024 fines, or the BES.media list of fines (1/3 BRV for ignoring signals, 1/2 BRV for no reflectors or lights, 3 BRV drunk riding) — [Kun.uz](https://kun.uz/ru/news/2025/11/03/kak-poluchit-prava-na-skutery-i-mototsikly-i-skolko-eto-stoit); [economist.kg](https://economist.kg/novosti/2024/09/20/v-uzbiekistanie-odobrieny-shtrafy-dlia-voditieliei-skutierov-mopiedov-i-vielosipiedov/); [BES.media](https://bes.media/amp/v-uzbekistane-vvodyat-shtrafi-za-narushenie-pdd-voditelyami-mopedov-i-elektrosamokatov-33fa43/)
- Courier labour is scarce: under 3 CVs per vacancy, average pay 3M UZS a month (Dec 2024) — [Kun.uz](https://kun.uz/en/news/2024/12/12/uzbekistans-delivery-market-sees-significant-growth)
- Stack: RF-DETR Nano (Apache-2.0), FLUX.2 [klein] 4B (Apache-2.0) for dusk and rain edits, fast-plate-ocr (MIT), PP-OCRv6 — local file `tech_radar.md`; the plate model's Uzbek-format coverage is unverified there

### Inferences
- The helmet ECE approval mark is normally sewn on the chin strap, so reading it from a 10-second selfie of a worn helmet is unrealistic (my reasoning, not sourced); strap-buckled state is also a tiny visual cue.
- A single pre-shift check proves gear at the start only; it is gameable (borrow a helmet, switch the light on, ride off). Random in-shift re-checks are what would change enforcement exposure.
- The legal basis for each checked item differs: reflectors, lights, plates and documents are evidenced as fined in 2024-2026; the helmet rule was not found in any of four articles.
- Money: 2,000 UZS per courier per month is about $0.17 against about $16.8 per average fine; Yandex's 4.7bn UZS (about $398k) of 2026 courier fines is the size of the problem the buyer already feels. Yandex also has in-house ML teams and a CV workshop at this hackathon, so build-versus-buy risk is real.
- Scores: novelty 3 (helmet part known, multi-item gate keyed to Uzbek fines is new), famous false (flagged: helmet detection is stock), feasibility 3, demo 4, business 3, moat 2, local fit 5, verdict sharpen.

### Gaps
- Uzbek helmet requirement and fine for mopeds, scooters and e-bikes: not found (Russian query run; four fetched articles silent); Yandex Eats / Uzum Tezkor / Express24 courier rules in Uzbekistan not found.
- Courier headcount per platform not found.
- Meituan's actual implementation (selfie, sensor helmet or other) is not specified in the source.

## safety-4 Seen-at-Night: flash-test your outfit's visibility

### Takeaway
No phone-flash conspicuity app for clothing was found, though camera-based retroreflectivity measurement is an established research area for signs and road markings (including flashlight plus HDR methods and phone-attached meters). The physics of a flash/no-flash difference is defensible, but the "metres of warning" number needs a calibrated field trial, and there is effectively no payer.

### Cited Findings
- Extended query "smartphone flash retroreflective clothing visibility test app pedestrian conspicuity retroreflectance measurement": only literature on retroreflective clothing; a 2023 systematic review of 43 studies finds biomotion placement plus colour retroreflective material is vital and that used tape underperforms new tape, so degraded materials need visibility testing (S) — [ResearchGate review](https://www.researchgate.net/publication/370660377_A_systematic_review_on_retro-reflective_clothing_for_night-time_visibility_and_safety)
- Standard query on phone-based retroreflectivity: no peer-reviewed study of a phone's own flash as the source; nearest analogues are SJSU/Mineta (Apr 2019) "remote retro-reflecto-meter" from a consumer camera, flashlight, GPS and screen using two simultaneous HDR images, claimed to meet FHWA accuracy for signs; Easylux phone-attached retroreflectometers certified to EN 1436 for road markings; Missouri S&T headlamp-lit video for signs; FHWA SMARTS strobe vehicle needing two operators (S for the Missouri and patent items) — [SJSU TransWeb](https://transweb.sjsu.edu/research/1878-Remote-Measurement-Road-Reflectivity); [Easylux](https://easylux.com.br/info/small-light-retroreflectometers-integrated-into-your-smartphone-can-revolutionize-road-marking-inspection-work/); [MST R044](https://transportation.mst.edu/research/r044); [US 7043057](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7043057)
- A 2009 Australian closed-circuit study found fluorescent vests without retroreflective stripes "were not a significant improvement on black clothing at night" and retroreflective strips worked better on knees and ankles than on a static jacket — [Wikipedia: High-visibility clothing](https://en.wikipedia.org/wiki/High-visibility_clothing) (secondary summary of the study)
- People misjudge their own conspicuity: "Pedestrians' estimates of their own visibility" found workers did not fully appreciate the differences between clothing configurations (S) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022437512000060) (403 on fetch)
- Pedestrians were in 2,689 of 5,895 crashes (Jan-Sep 2025, 45.6%) — [Kun.uz, 3 Nov 2025](https://kun.uz/en/news/2025/11/03/over-1300-people-killed-in-road-accidents-across-uzbekistan-since-january); the night, rain and weather share of crashes in Uzbekistan was not found (teammate gap) — `uzbekistan_market.md` section 5

### Inferences
- Geometry (my calculation): retroreflection depends on observation angle, roughly source-to-eye separation divided by distance. A car's headlight-to-driver-eye offset of about 0.6 m at 100 m is about 0.34 degrees; a phone flash about 1-1.5 cm from the lens at 2 m is about 0.3-0.4 degrees. So a close-range phone shot can mimic long-range headlight geometry, which makes a calibrated relative index credible.
- Weak points: auto-exposure and HDR make the two photos non-linear (a reference tape sample in frame is needed); web pages cannot switch the torch on iOS Safari, so the demo should use the native camera plus upload; "metres at 40-60 km/h" needs the planned night trial before it can be quoted.
- No payer at Uzbek prices: 1,500 UZS per worker per month is about $0.13; gear-seller lead fees are unquantified; employers can simply buy certified gear.
- Scores: novelty 4, famous false, feasibility 3, demo 4, business 1, moat 2, local fit 3, verdict sharpen (per-audit degradation check for employer-issued hi-vis, or a module inside a courier check).

### Gaps
- Chinese and Russian searches and the "smartphone retroreflectometer validation" follow-up were not run (quota).
- Share of Uzbek pedestrian crashes at night or in smog is unknown, so the local-impact claim is not evidenced.

## safety-5 HazardGaze: webcam hazard-scanning coach for learners

### Takeaway
No webcam product was found, but the headline metric (which hazard you missed or saw late) is not supportable at webcam-gaze accuracy (mean error 4.17 degrees in the WebGazer paper), the closest prior art is a State Farm patent and decades of eye-tracker research, the Uzbek exam has no hazard-perception component, and the payers are small and unproven.

### Cited Findings
- Extended query "webcam eye tracking hazard perception test driver training gaze novice drivers product": no webcam-based hazard-perception product; results were dedicated eye-tracker or simulator studies; a PC-based eight-video hazard prediction programme improved young drivers' scanning, with gains at five months (N=19 vs 20); NHTSA's SAFE-T review; a 2006 study found trained novices almost twice as likely to fixate risk regions (S) — [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S1369847824000287); [NHTSA](https://www.nhtsa.gov/book/countermeasures-that-work/young-drivers/countermeasures/other-strategies-behavior-change); [Pollatsek et al. 2006](https://doi.org/10.1518/001872006778606787)
- US 10,810,900 B2 (State Farm, filed 27 Feb 2017, granted 20 Oct 2020; priority 6 Mar 2012): web-streamed continuous drive with hazard modules, records where the user looks (mainly via mouse or touch clicks, plus panning to mirrors), scores hazard-detection proficiency, ties it to an insurance discount; no webcam mentioned — [Google Patents](https://patents.google.com/patent/US10810900B2/en)
- WebGazer (IJCAI 2016): mean errors of 175 and 210 pixels, "average visual angle of 4.17" (about 1.6 inches on a 24-inch monitor at 59 cm); the library is GPLv3 and as of 24 Feb 2026 "no longer guaranteed" to receive updates — [paper PDF](https://jeffhuang.com/papers/WebGazer_IJCAI16.pdf) (text extracted locally); [WebGazer site](https://webgazer.cs.brown.edu/)
- Gaze and rPPG models carry research or use-restriction licences; MediaPipe driver-monitoring rules need no training — local file `tech_radar.md` section 1
- Uzbek theory exam is a 20-question computer test (18 correct needed); practical exam is range-based with sensors; no hazard-perception element — [Kun.uz](https://kun.uz/en/news/2025/12/10/how-to-obtain-a-drivers-license-in-uzbekistan)

### Inferences
- At 4 degrees and a normal laptop distance, gaze lands within about 4 cm on screen, comparable to the size of a distant pedestrian in a clip, so "missed pedestrian" claims would be noise; zone-level (left/centre/right) coaching is the honest limit and is what an instructor can do by watching.
- Phone-in-cabin "mirror usage" cannot be measured on a webcam test of a pre-recorded clip unless mirrors are rendered in the clip.
- Payer: driving schools 10k UZS per lesson (about $0.85) and fleets 15k UZS per assessment (about $1.27); there is no regulatory pull and no evidence that hazard-perception scores change crash or insurance outcomes locally.
- Scores: novelty 3, famous false, feasibility 3, demo 3, business 2, moat 2, local fit 2, verdict kill.

### Gaps
- UK/Australian click-based hazard-perception tests not fetched; commercial webcam-gaze vendors (for example in market-research) not checked for a driving use case.
- DADA-2000 and DoTA licences not verified.

## safety-10 AutodromRef: camera-scored practice tracks for driving schools

### Takeaway
Best fit in this batch. The real Uzbek practical exam is an objective, sensor-scored 100-point range test with a 1.2M UZS retake, driving schools carry a public rating that is 30% graduates' exam results, and training autodromes are a regulated, mandatory asset. Exam-grade video-analytics tracks exist in India (Maruti Suzuki, Rosmerta), but no practice-grade, low-cost product was found. Main risks: unknown number of schools and camera accuracy on a 50 x 15 m yard.

### Cited Findings
- Exam: range-based, five manoeuvres (slalom, ramp start, parallel parking, confined-space turn, stop at line), 100-point scale, "deductions exceed 99 points" fails, vehicle sensors track "trajectory, distance, and lane markings", results cannot be appealed, retakes 1.2M UZS after seven days, schools with their own driving range charge about 7M UZS for the course, extra practice 100k-250k UZS per hour — [Kun.uz, 10 Dec 2025](https://kun.uz/en/news/2025/12/10/how-to-obtain-a-drivers-license-in-uzbekistan)
- Training-autodrome regulation (MVD order registered 1 Feb 2024): separate sites for categories A, B/C/D and with trailers; minimum 50 m x 15 m; asphalt or concrete; lighting; 2 m fence; maximum speed 5 km/h; required elements slalom, intersection, 90-degree turns, reverse entry to a box, reverse parallel parking; road signs and traffic lights; no cameras or sensors mentioned — [Spot, 6 Feb 2024](https://www.spot.uz/ru/2024/02/06/driver-ed-field/)
- Driving-school rating (from 1 Sep 2022): up to 30 of 100 points from graduates' exam results, 30 from graduates' penalty points, 30 from guilty-crash involvement, 10 from staff; green/yellow/red signs shown at the school and on training cars — [Spot, 6 Jul 2022](https://www.spot.uz/ru/2022/07/06/driver-schools/)
- About 500,000 licences a year (2022) — [Spot, 11 Feb 2022](https://www.spot.uz/ru/2022/02/11/driving-schools/); school owners (Jan 2026) expect no sharp demand drop from optional theory classes, plan stronger practical training, individual lessons and exam-task analysis, and see A1 moped training as growing — [Spot, 15 Jan 2026](https://www.spot.uz/ru/2026/01/15/driving-school/)
- India: Rosmerta awarded Rs 166.69 crore to design, build, operate and transfer ten Automated Driving Testing Tracks in Maharashtra (15 Jul 2025), with "instant video analysis across multiple camera angles to flag any errors" and facial recognition — [The Machine Maker](https://themachinemaker.com/news/rosmerta-awarded-%e2%82%b9166-69-cr-contract-to-build-10-automated-driving-test-tracks-in-maharashtra/); Maruti Suzuki automated all its Delhi test tracks using "video analytics technology", 10-minute cycle, "zero human intervention", reports pass rate about 84% (manual, 2018), about 34% after automation, about 64% now (company claim), 7 lakh candidates evaluated — [Maruti Suzuki](https://www.marutisuzuki.com/corporate/media/press-releases/2023/may/maruti-suzuki-automated-driving-test-track); seven more tracks in Tamil Nadu with HD cameras, RFID sensors and analytics (S) — [Storyboard18](https://www.storyboard18.com/how-it-works/maruti-suzuki-expands-its-road-safety-efforts-adds-7-automated-driving-test-tracks-adtts-in-tamil-nadu-under-its-csr-initiative-91033.htm)
- Standard query "computer vision automatic driving test evaluation system cones parking camera-based scoring driving school range" and the earlier extended query found no commercial camera-scored practice-track product for driving schools; closest is a patent on camera-based parking-skills assessment and cone-detection datasets (FSOCO) (S) — [US 10824882](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10824882); [Roboflow cone set](https://universe.roboflow.com/cone-dataset-training/evaluation-set-cone-detection)
- Microsoft Research India's HAMS "Automated License Testing" documentation contrasts its low-cost windshield-phone approach with "pole-mounted high-resolution cameras used in earlier instrumented tracks"; the MIT-licensed repo (archived 11 Jun 2026) lists the topics aruco, computer-vision, license-testing, but the README and docs pages read did not describe the track-scoring modules — [HAMS docs](https://microsoft.github.io/HAMS/); [GitHub](https://github.com/microsoft/HAMS)
- Stack: RF-DETR Nano (Apache-2.0), EdgeTAM (Apache-2.0, 16 FPS on iPhone 15 Pro Max), SAM 2.1 — local file `tech_radar.md` ([EdgeTAM](https://github.com/facebookresearch/EdgeTAM))

### Inferences
- The 5 km/h cap on training sites makes tracking and motion blur easy; the 750 m2 minimum yard is the problem for "one phone": a single overhead phone at pole height gives poor resolution at 50 m, so expect 4K wide-angle or two cameras, and quote accuracy as roughly ±10-15 cm until measured against a tape (my estimate), not "centimetre".
- Payback: 400k UZS per track-camera per month is about $34; two to four extra practice hours at 100-250k UZS pay for it, and the rating (30 points on exam results) is a documented incentive. The number of licensed schools with their own autodrome was not found, so TAM is unknown.
- Differentiation from India: the Maharashtra contract is Rs 166.69 crore for ten tracks, about Rs 16.7 crore each by my division, covering design, build, five-year operation and transfer; a practice-grade add-on at tens of dollars a month is a different product, and it trains learners for the sensor-scored exam rather than replacing it.
- Scores: novelty 3 (exam-grade video-analytics tracks exist in India), famous false, feasibility 4, demo 4 (toy car on a printed track), business 4, moat 3, local fit 5, verdict keep.

### Gaps
- Number of Uzbek driving schools, autodromes and exam centres: not found (Spot tag pages and articles give no totals); the Russian query was blocked by quota.
- Chinese Subject-2 camera systems and Russian range-scoring vendors not checked (queries blocked).
- Exam-sensor vendor in Uzbekistan and the "30 cm rollback" threshold: not found.
- FSOCO and Roboflow cone-set licences not verified (the FSOCO site fetch returned an unrelated page).

## safety-15 GripSense: slick-surface alerts for riders

### Takeaway
Camera road-surface classification is academically solved on large datasets (RSCD, 84.6% top-1 for a MobileNetV3-Small), including gravel and wet gravel for motorcycles; the hazards the pitch names (oil sheen, wet paint, manhole covers, sand) are mostly not in that data, warning time at courier speeds is shorter than a rider's reaction, and there is no payer.

### Cited Findings
- Venancio et al., Frontiers in Artificial Intelligence 8:1520557 (Feb 2025): road-surface classification for electric motorcycles with a gravel focus on RSCD (about 960k train, 50k test, 20k validation images; 27 classes); test top-1 EfficientNetB0 89.4%, EfficientNetV2B0 88.9%, MobileNetV3-S 84.6%, InceptionV3 87.8%, ConvNeXt-T 91%; gravel about 89-90% after fine-tuning; wet gravel 47% to 81%; Raspberry Pi 4 seconds per frame MobileNetV3-S 0.08, EfficientNetV2B0 0.46, ConvNeXt-T 7.50; no camera mounting, lookahead or night/dust/glare evaluation — [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11868262/fullTextXML); [DOI](https://doi.org/10.3389/frai.2025.1520557)
- RSCD: about 1M images, 27 classes from friction (dry, wet, water, fresh snow, melted snow, ice), material (asphalt, concrete, dirt/mud, gravel) and unevenness; 2M monocular camera on the bonnet, 360x240 crops, Beijing Jan-Jul 2022, about 700 km; the page states no licence — [RSCD](https://thu-rsxd.com/rscd/)
- Smartphone camera plus accelerometer road-anomaly detection including manholes (Lee et al., Sensors 2021) (S) — [Sensors](https://doi.org/10.3390/s21020561)
- Extended query "motorcycle phone camera road surface classification warning gravel wet paint manhole rider alert app": no shipping consumer app combining gravel, wet paint and manhole warnings for riders (S)
- GitHub topic road-surface-classification has 3 repos, none mobile or motorcycle — [GitHub](https://github.com/topics/road-surface-classification)
- Michelin discontinued its Better Roads product by end-2025 and takes no new projects; municipal pavement-AI is a hard business — local file `saturation_map.md` section 2D ([Michelin](https://mobilityintelligence.michelin.com/us/road-assessments/))
- Local data: 563 moped and scooter crashes, 89 deaths, 488 injured in nine months of 2026 — [Spot](https://spot.uz/ru/2026/10/06/couriers); cause breakdown by road surface not found
- Stack: SigLIP 2 (Apache-2.0), DINOv3 (DINO License, gated), MobileNet-class heads; Depth Anything V2 Small — local file `tech_radar.md`

### Inferences
- Warning time (my arithmetic): at 40 km/h (11.1 m/s) a 5-15 m lookahead is 0.45-1.35 s, below or near a typical perception-reaction time of about 1 s (my estimate), so the useful output is "ease off before a bend", not avoidance; aggregated hotspot maps for sweeping contractors are the municipal-buyer model that Michelin's exit suggests is hard.
- Domain shift: RSCD uses a bonnet camera pitched at the road; a handlebar phone sees more sky, vibration and motion blur; the pitch's own classes (paint, metal cover, sand, oil sheen) need roughly 2,000 self-labelled frames, which is feasible but unvalidated.
- Money: 2,000 UZS per courier per month is about $0.17; no rider-side payer.
- Scores: novelty 3, famous false, feasibility 3, demo 3, business 2, moat 2, local fit 3, verdict kill (reason in the output).

### Gaps
- Chinese, Russian and app-store checks for rider road-surface apps not run (quota); RSCD licence unknown.
- Share of Uzbek moped crashes caused by surface hazards: not found.

## safety-16 BackSafe: child-behind-car alert for courtyard gates

### Takeaway
Person detection on reversing cameras and home cameras is a commodity (and cameras are mandated in new cars in the US and EU). The specific hazard, a small child right behind the bumper, sits in the blind zone that a rear-shelf phone cannot see through the boot lid, and reverse-gear inference from audio or IMU is fragile. Weak moat and no payer.

### Cited Findings
- Extended query "backover prevention child behind car AI camera detection aftermarket reversing phone app": no aftermarket AI camera product with a phone app for detecting children behind a car was found, but the landscape is OEM and aftermarket cameras, parking sensors and blind-spot monitors; a 2006 NHTSA study found sensor-based parking aids "poor, sporadic and limited in range" and cameras the most promising (S) — [AAA](https://www.acg.aaa.com/connect/blogs/6c/home/protect-kids-from-home-backover-accidents); [NHTSA 2006 study](https://www.kidsandcars.org/document_center/download/backovers/2006-11-01-nhtsa-backover-avoid-tech-study.pdf)
- Vayyar op-ed on KidsAndCars.org: ultrasonic sensors have a range of about 10 ft (3 m) and "cannot reliably detect a child lying flat"; "Only AI-enhanced cameras – an ADAS platform – would be capable of providing automated alerts"; cameras are hampered by rain, snow or dust on the lens; large vehicles' front blind zones often extend to 16 ft (4.9 m); the author is a radar-company executive — [Kids and Car Safety](https://www.kidsandcars.org/news/stopping-frontovers-in-their-tracks)
- A 2025 KidsAndCars.org study reports a 78% reduction in child fatalities and nearly 50% drop in severe injuries since the 2018 US backup-camera mandate (S) — [Bioengineer.org summary](https://bioengineer.org/new-backup-camera-requirement-in-vehicles-leads-to-nearly-50-drop-in-severe-injuries-and-78-reduction-in-child-fatalities/)
- US backup-camera compliance date 1 May 2018; EU requirement for a reversing camera or detection system from 6 Jul 2022 — [Wikipedia: Backup camera](https://en.wikipedia.org/wiki/Backup_camera)
- Cobalt was 30.1% of 2025 new-car sales (a sedan with a boot lid) — local file `uzbekistan_market.md` section 1 ([BestSellingCarsBlog](https://bestsellingcarsblog.com/2026/02/uzbekistan-full-year-2025-chevrolet-cobalt-at-30-1-share/))
- Stack: D-FINE-N (Apache-2.0, 15 MB ONNX) and Depth Anything V2 Small (Apache-2.0) — local file `tech_radar.md`

### Inferences
- Geometry (my reasoning): a phone on a sedan's rear shelf looks over the boot lid, so the ground within a few metres of the bumper, exactly where backover victims are, is hidden; a gate or wall camera looking at the driveway avoids this but then it is an ordinary smart-home person-detection feature (widely sold; not source-checked here).
- No Uzbek backover statistics were found, so the courtyard premise is unquantified; 25k UZS one-off (about $2.1) per family or 20k UZS per depot vehicle per month (about $1.7) is not a business, and the pitch itself concedes a weak moat.
- Scores: novelty 2, famous true, feasibility 4, demo 3, business 1, moat 1, local fit 2, verdict kill.

### Gaps
- Specific aftermarket products (Brigade, Mobileye, Blaxtair, smart-home brands) were not fetched (Brigade returned 403); no Uzbek backover statistics; the Russian-language check was not run.
