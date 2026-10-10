# Adversarial verification, batch 03: ten insurer-segment and used-car CV ideas for the WIUT Hackathon 26 (Computer Vision track, Tashkent)

> Read-me. Date: 2026-10-10 (finals day 1). Ideas checked: usedcar-17 LemonBack, insurer-1 ScratchPrint, insurer-2 ProofShot, insurer-3 BumperMatch, insurer-4 EuroScene, insurer-7 Podstava Radar, insurer-9 SleepSpot, insurer-10 WashScan, insurer-13 ImpactMeter, insurer-15 GeoProof.
> Method: targeted WebSearch (extended mode for niche topics) plus direct page fetches of vendor pages, patents, registries and Uzbek news. The shared 200-call WebSearch budget ran out repeatedly; about 16 searches succeeded for me, so GeoProof got no successful search (only fetches), and the RU/UZ-language probes were thin (Europrotocol-Uzbekistan returned only RU/UA/BY pages). Novelty scores for GeoProof, BumperMatch, SleepSpot and Podstava Radar are therefore provisional.
> Flags: (S) = taken from a search-result summary, page itself not read; [CALC] = my arithmetic on cited inputs; "teammate" = taken from the saturation_map.md / uzbekistan_market.md / tech_radar.md / hackathon_intel.md notes in this folder (URLs as cited there, not re-fetched unless stated). USD at about 11,800 UZS (teammate).
> Shared Uzbek numbers used below (parsed locally with pdftotext from the NAPP Q4 2025 report, table 1.4, claims table 2.10): OSAGO premiums 406,165 mln UZS, payouts 245,224 mln; class 3 (land-vehicle, KASKO-type) premiums 754,268 mln (-47.0%), payouts 107,758 mln (-35.9%), contracts in force 660,550, concluded in 2025 237,405 (-41.0%); general-insurance claims received 274,797 (+91.2%), paid 264,961, all-lines payouts 2,845,072 mln UZS. Source: [NAPP Q4 2025 report](https://napp.uz/storage/files/shares/opendata/2025/4Q/%D0%A1%D1%82%D1%80%D0%B0%D1%85%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%28IV%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%B0%202025%20%D0%B3%D0%BE%D0%B4%20%29%20rus%20%E2%80%94%20filtr.pdf). The report gives claim counts per company, not per class.
> Shared derived estimate [CALC, assumption-heavy]: all-lines mean paid claim = 2,845,072 / 264,961 = 10.7M UZS. If motor claims average the same, OSAGO (245,224 mln) is about 23k claims and class 3 (107,758 mln) about 10k claims, so roughly 33k motor claims a year (about 65k if motor claims average half as much). At the per-claim prices the ideas propose ($0.10-0.50) the whole Uzbek motor-claim market is worth about $3k-$16k a year if every claim were screened. This is the main reason every insurer-payer idea scores low on business.
> Hackathon constraint (teammate, hackathon_intel.md): finals 10-11 Oct, about 8.5 scheduled on-campus hacking hours, code freeze 13:00 on 11 Oct, 7+3 minute pitches, weights Innovation 25 / Technical complexity 25 / Impact 25 / Presentation 15 / Business 10. So "2-3 days" is optimistic; only thin vertical slices are realistic.

## usedcar-17 LemonBack: photo-underwritten buyback guarantee for used cars

### Takeaway
Insurer-backed used-car buyback and money-back guarantees already exist (a dealer-paid repurchase guaranty backed by an A-rated insurer in 2009, Carro Certified in South-East Asia, history-only buybacks from Carfax and AutoCheck), but I found none that is seller-paid, peer-to-peer and priced from a photo-derived risk score. The pitch is attractive but its unit economics are near break-even on its own assumptions and the risk model has no ground truth.

### Cited Findings
- Carfax Buyback Guarantee is a title/history product: a third-party summary says it covers salvage or branded titles, odometer rollback, flood, fire and frame damage, "does not cover general wear and tear or mechanical problems", is "not an insurance policy", and gives conflicting windows (30 vs 90 days) and payers, so treat details as unreliable — [aboutcarsreviews](https://aboutcarsreviews.com/articles/what-is-the-carfax-buyback-guarantee-everything-you-need-to-know)
- AutoCheck (Experian) Buyback Protection covers title-brand problems missed by the DMV, with a purchase-price cap and a one-year window, backed by Experian itself, not an insurer (search summary only; the AutoCheck page returned just a title) (S) — [Experian Insights](https://www.experian.com/blogs/insights/what-is-autocheck-buyback-protection/)
- Spinny (India) Assured BuyBack: "Upfront buyback value assurance at time of purchase, valid for a period of 12, 18 and 36 months"; the page shows no fee, no insurer, and no stated link between the buyback value and its "200-point inspection" — [Spinny](https://www.spinny.com/car-buy-back-program/)
- Carro Certified (launched 5 Oct 2022 in SG, ID, MY, TH, TW): 160-point inspection whose data scientists use "AI-enabled capabilities and computer visioning", a 5-day money-back guarantee and a 12-month engine and gearbox warranty; price, insurer partner and return rates not stated — [TechNode](https://technode.global/2022/10/05/singapores-carro-launches-assurance-certification-for-used-cars/)
- Vehicle Repurchase Guaranty (Auto Remarketing, 15 Jan 2009): dealers could offer a repurchase guarantee whose performance is "backed by an insurer rated A"; buyback price is 95% of that day's Black Book trade-in value (not the purchase price); conditions include a no-cost independent inspection, offering the car to the selling dealer first, and no material damage — [Auto Remarketing](https://www.autoremarketing.com/subprime/dealers-can-now-offer-repurchase-guarantee-customers-new-service)
- Buy Auction Auto charges $125 to guarantee the accuracy of its inspection findings (search summary) (S) — [Buy Auction Auto](https://buyauctionauto.com/inspections-now-guaranteed/)
- Odometer evidence from photos alone is weak in the sources I have: Truepic reads mileage and checks tamper/time/place; carArth OdoShield scores plausibility mainly from registration, insurance and service records plus "wear-pattern analysis" (teammate) — [Truepic](https://www.truepic.com/blog/odometer-fraud); [VahanBazaar](https://vahanbazaar.in/news/ai-odometer-detection-1-in-5-used-cars-vahan-reveal-2026)
- Uzbek class-3 (KASKO-type) premiums fell 47.0% to 754,268 mln UZS in 2025, payouts 107,758 mln UZS, 237,405 contracts concluded (-41.0%); payout ratio 107,758/754,268 = 14.3% [CALC] — [NAPP Q4 2025](https://napp.uz/storage/files/shares/opendata/2025/4Q/%D0%A1%D1%82%D1%80%D0%B0%D1%85%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%28IV%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%B0%202025%20%D0%B3%D0%BE%D0%B4%20%29%20rus%20%E2%80%94%20filtr.pdf)
- Used-car market (teammate): 52.6k (Sep 2025), 57.4k (Oct), 52k (Nov) secondary deals a month; pre-purchase inspection advertised "from 300,000 UZS"; used Cobalt $11.5-12.7k (Jan-Mar 2025) — [Kun.uz Nov 2025](https://kun.uz/en/news/2025/11/18/electric-car-purchases-surge-as-overall-auto-sales-recover-in-uzbekistan); [OLX listing](https://www.olx.uz/d/oz/obyavlenie/ot-300-000-avtopodbor-proverka-avto-probega-pered-pokupkoj-avtotekshi-ID3XhPY.html); [Kun.uz Apr 2025](https://kun.uz/en/news/2025/04/23/used-car-prices-fall-across-uzbekistan)
- Tooling the pitch relies on: AutoDamageIQ carries a CC BY 4.0 tag but its card lists upstream sources CarDD (academic), VehiDE and GPT-4o auto-labelled images with no upstream licences stated — [HF AutoDamageIQ](https://huggingface.co/datasets/tugberkkalay/autodamageiq-vehicle-damage-dataset)
- Market-fraud context: Sergeli car-market raid over intermediaries selling parts at several times price (teammate) — [Anhor](https://anhor.uz/ru/news/zloumyshlenniki-vstrechali-posetitelej)

#### Queries run
- "used car buyback guarantee priced by AI inspection insurer-backed" (extended): no AI-priced insurer-backed seller-paid product; only Spinny, Cars24, Carfax, AutoCheck, Buy Auction Auto.
- "used car warranty buyback guarantee underwritten by computer vision vehicle condition report insurer partnership" (standard): Carro Certified, sgCarMart/Tokio Marine/Assurant warranty, Carsome 5-day money-back, Truepic x Warranty Inspection Services (2023), the 2009 repurchase guaranty.
- A third query for private-sale insurance-backed guarantees was refused (budget).

### Inferences
- Break-even [CALC]: a buyback costs the insurer roughly the loss on resale; assuming 10-25% of a $11.5-12.7k Cobalt-class price ($1.15-3.2k), a $25 fee breaks even at only a 0.8-2.2% claim rate before commission and adjudication costs. The pitch's "<1.5%" assumption leaves almost no margin and is unvalidated.
- Buying back at 95% of trade-in value (as the 2009 guaranty did) rather than at purchase price is a known lever to bound insurer loss; the pitch does not use it.
- 40 captured cars with inspector verdicts cannot support a gradient-boosted risk model or meaningful conformal calibration (granularity about 1/41 = 2.4%, very wide bands); losses are explicitly simulated, so the premium table is invented. Flood and mechanical hidden defects are not reliably observable from exterior phone photos in any source I found.
- A defensible variant is an evidence-bound guarantee: cover only what the dated capture can prove (undisclosed prior repair or repaint, mileage inconsistency with service records), exclude engine, gearbox and flood, and price from a rules table until losses exist.
- Moral hazard (buyer's remorse dressed as a "lemon") and claim adjudication cost are not modelled; selling a guarantee requires a licensed insurer (NAPP), not checked.

### Gaps
- No Uzbek or CIS used-car guarantee product search in RU/UZ succeeded (budget); absence not proven.
- No base rate for used-car return or major-defect claim frequency (Carvana, CarMax, Kavak) found.
- Carfax and AutoCheck official terms unreadable (403 / title-only page).
- Whether an Uzbek insurer would underwrite a seller-paid buyback was not tested.

## insurer-1 ScratchPrint: pre-existing and double-claimed damage matcher

### Takeaway
"Same damage or new damage" against an inception baseline is already a feature of inspection vendors (Inspektlabs, Click-Ins, Chex.AI), and cross-claim duplicate-image checks are sold by Shift Technology and VAARHAFT; what I did not find is a neutral cross-insurer registry that re-identifies the same physical damage in new photos. The patent cited in the pitch is mis-attributed (it is Ping An's, not Click-Ins').

### Cited Findings
- Inspektlabs describes pre-policy inspections that capture "pre-existing damages" as a baseline, claim review of "Old damage vs. new damage", an FAQ line that AI can "compare inspection history across claims" and detect "duplicate or previously used images", and GPS/timestamp checks; no numbers or named customers — [Inspektlabs blog](https://inspektlabs.com/blog/ai-in-motor-insurance-inspection-inspektlabs/)
- Click-Ins' site says users can "Easily compare different reports to highlight new damages" and mentions "Damage Print" only as a branded line under Smart City Solutions; no duplicate-claim feature is described; partner logos include Sixt Group, Sapiens, IDI Direct Insurance, KBC, OPENLANE, Volkswagen, Kavak, IAA; investors Mobilion, SUMMUS, MAY, Sixt — [Click-Ins](https://www.click-ins.com/)
- Chex.AI "flags manipulated, duplicate, or stale images before they enter a claim", reads VIN/plate/odometer, detects 21 damage types on 163 parts (95-99% claimed), and its towing/valet intake scan records pre-existing damage — [Chex.AI](https://www.chex.ai/)
- Ravin AI: RAVIN Eye "Tracks vehicle condition changes over time", RAVIN Inspect "Verifies image integrity and completion in real time", RAVIN AutoScan "Transforms CCTV cameras into vehicle scanners"; insurer logos include IAG Insurance, NRMA, Sompo and Rollin Insurance (which products each uses is not stated); the page does not mention duplicate-image detection or car-wash use — [Ravin AI](https://www.ravin.ai/)
- Patent US10410292B2 "Method, system, apparatus, and storage medium for realizing antifraud in insurance claim based on consistency of multiple images" is assigned to Ping An Technology (Shenzhen), priority 22 Jan 2016, granted 10 Sep 2019, active to about 2037; it matches SIFT keypoints between photos of the same part in one submission (RANSAC, fundamental matrix, transform, compare colour/texture) to flag forgery; it does not fingerprint photos and does not search across claims — [Google Patents](https://patents.google.com/patent/US10410292B2/en)
- A vendor-written 2026 comparison (VentaVid, caveat: written by a competitor) says Shift Technology scores cross-claim image similarity ("even if images have been cropped, resized, or slightly altered") and VAARHAFT offers duplicate checks "inside and outside your organisation", while "no tool explicitly describes a shared cross-insurer database" — [VentaVid](https://www.ventavid.com/blog/insurance-image-fraud-detection)
- Shift Technology's own home page does not mention image similarity (so VentaVid's claim above is unconfirmed) but describes an "Insurance Data Network (IDN)" as an "agentic intelligence layer that assesses cross-carrier claim history", with "4B+" policies, claims and documents analyzed, 100+ third-party data sources and 120+ customers in more than 30 countries; so a cross-carrier claims-history network already exists as a concept (not for damage geometry) — [Shift Technology](https://www.shift-technology.com/)
- A Verisk executive's column says duplicate detection means comparing photos across claims and carriers, and cites a property appraiser who reused one photo in 170 claims over two years (over $1M indemnity); no method is described — [Carrier Management](https://www.carriermanagement.com/features/2023/05/25/248724.htm)
- Verisk ClaimSearch is Verisk's industry claims database (the fetched page gave only a title, so its image-matching capability is unconfirmed); Truepic's site carries a headline on bringing verified visual evidence into the ClaimSearch workflow — [Verisk ClaimSearch](https://www.verisk.com/products/claimsearch/); [Truepic](https://www.truepic.com/)
- Uzbek hooks: all policies go through the unified EAIS since 1 Aug 2025 (teammate); the Insurance Guarantee Fund "gains adjuster functions related to notifying insurers of road traffic accidents" and a Unified Insurance Billing System is due from 1 Apr 2026; the Competition Committee received almost 1,000 insurance appeals, over 50% about OSAGO refusals or delays, and secured 2.8 bn UZS — [Buxgalter.uz](https://buxgalter.uz/ru/publish/doc/text209644_kak_izmenitsya_stoimost_polisov_osago); [Spot 24 Jul 2025](https://www.spot.uz/ru/2025/07/24/insurance-auto/); [Spot 18 Feb 2026](https://www.spot.uz/ru/2026/02/18/insurance-complaints/)
- Tooling licences (teammate tech radar): RF-DETR incl. Seg Apache-2.0, SAM 2.1 permissive, SAM 3 under the SAM License, DINOv3 under a custom licence (commercial use allowed with conditions) — [RF-DETR](https://github.com/roboflow/rf-detr); [SAM 3](https://github.com/facebookresearch/sam3); [DINOv3 licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- AutoDamageIQ upstream-licence problem as in the LemonBack section — [HF AutoDamageIQ](https://huggingface.co/datasets/tugberkkalay/autodamageiq-vehicle-damage-dataset)

#### Queries run
- "duplicate claim detection vehicle damage photos across insurers image matching pre-existing damage" (extended)
- Fetches: Inspektlabs blog, Click-Ins home, Chex.AI home, Google Patents US10410292B2, VentaVid comparison, Carrier Management column, Verisk ClaimSearch; VAARHAFT blog returned HTTP 429.

### Inferences
- The inception-baseline half is a commodity feature of at least three inspection vendors; the differentiator is the neutral cross-insurer registry and re-identifying the same physical damage in fresh photos, which no fetched source claims. Novelty is partial.
- Thin scratches are low-contrast, specular and change with viewpoint and light, so keypoint matching on a laptop lid or wet paint will often return "insufficient features"; dents, chips, decals and plate-adjacent texture match far better. Pre-stage a car with distinctive damage.
- DINOv3 patch descriptors are embeddings, not irreversible hashes; "descriptors only, no photos" is a privacy claim that needs a legal read (Uzbek personal-data law not checked).
- Volume problem [CALC]: about 33k motor claims a year makes $0.25 per claim worth roughly $8k a year nationally; fleets and rental handovers (every shift) are the volume play, and the Guarantee Fund is a plausible neutral host or buyer.

### Gaps
- Whether Click-Ins sells any "Damage Print" product to insurers is unclear (one line on its page).
- Verisk's actual image-matching capability could not be read; Shift Technology's own page was not fetched (relied on the VentaVid summary).
- No evidence of double-claiming or double-insuring prevalence in Uzbekistan.

## insurer-2 ProofShot: claim photos that prove they're real

### Takeaway
Claim-photo authenticity is a crowded, fast-moving niche (Truepic, VAARHAFT, Inaza, Shift, Arvato, Venta Capture, TrueScreen, Reality Defender) and several already combine capture-time controls with detection; the unoccupied space is an explainable physical-liveness check (torch echo, parallax, sun and weather consistency). The live "try to cheat it" demo is strong, but a general fake-detector claim would be indefensible.

### Cited Findings
- Milliman: "Human detection is only 50% accurate" (experienced professionals "no better than random chance"), metadata analysis "can be easily bypassed", "advanced forensic tools often struggle"; the page names no capture-time controls — [Milliman](https://www.milliman.com/en/insight/accidents-that-never-happened-genai-motor-fraud)
- Aviva detected "more than 18,400 suspect claims worth £233 million" in 2025 and is "seeing an increasing number of claims backed by fabricated accident scene photographs", especially motor (12 Jun 2026) — [Lawyer Monthly](https://www.lawyer-monthly.com/2026/06/aviva-direct-line-claims-fraud-ai-documents-2025/)
- German trade press (AUTOHAUS, 8 Jun 2026) lists illogical light, shadows or perspective and date mismatches as red flags, proposes forensic file analysis and on-site expert inspection, and proposes no capture-time control — [AUTOHAUS](https://www.autohaus.de/nachrichten/schadenbusiness/versicherungsbetrug-mit-ki-wenn-bilder-luegen-3811631)
- Debevoise (Jan 2026) recommends several photos from multiple angles with a scale reference, AI-image detection, native files and live remote video inspection, and warns against over-correcting into wrongful denials — [Debevoise](https://www.debevoisedatablog.com/2026/01/19/use-of-ai-generated-images-for-fake-insurance-claims-and-other-frauds/)
- Truepic collects images with "verified time, date, device, and location metadata", runs "Location authentic", "Time & date accurate" and "Device integrity" checks, flags manipulation, and has a headline on bringing verified evidence into Verisk ClaimSearch; a Jewelers Mutual testimonial and a Ford Protect claims testimonial appear — [Truepic](https://www.truepic.com/)
- VentaVid's vendor-written comparison (caveat: it sells a competitor, Venta Capture): VAARHAFT SafeCam does photo-of-photo, virtual-camera and GPS-spoofing checks and FraudScanner detects AI-generated images and duplicates; Truepic Vision Flex starts from a "1,000 USD" monthly minimum; TrueScreen "up to 2 EUR per certification"; Inaza checks fit with "weather, time, location" and metadata swaps; Shift scores cross-claim image similarity; Venta Capture uses a spoken session code and motion check — [VentaVid](https://www.ventavid.com/blog/insurance-image-fraud-detection)
- Inaza's own home page (fetched) does not mention weather/time/location checks; it lists MGA/carrier logos and "Structured PoC in 4 weeks" — [Inaza](https://www.inaza.com/)
- Bdeo markets VIN, GPS, plate, odometer and screen-detection checks and "15% fraud detected" in motor claims (vendor claim, teammate); PZU runs an in-house PhotoDetective (teammate) — [Bdeo](https://bdeo.io/en/blog/detecting-auto-insurance-fraud-in-the-age-of-generative-ai/); [Qorus](https://www.qorusglobal.com/innovations/30701-photodetective-detecting-ai-generated-and-internet-sourced-fraud)
- Russia: Rosgosstrakh (quoted as Arman Gasparyan) says fraudsters offer to generate fake damage photos with neural nets (search summary; the source is one of the autonews.ru, www1.ru or bfm.ru results, not identified which) (S) — [Autonews](https://www.autonews.ru/amp/news/688869969a79473ec8de3c10); [www1.ru, 21 Nov 2025](https://www1.ru/news/2025/11/21/mosenniki-nasli-novyi-sposob-obmanut-straxovye-kompanii-s-pomoshhiu-neirosetei.amp.html); and Za Rulem 6 Jan 2026 (teammate) — [Za Rulem](https://www.zr.ru/content/articles/976091-400-tysyach-za-foto/)
- Open-source position (teammate tech radar): "AI-image fraud checks cannot be outsourced to a VLM, and the best open forensic models (TruFor, B-Free) are non-commercial"; Community-Forensics ViT-S is MIT but unverified on current generators; c2pa-rs is MIT/Apache — [Community-Forensics ViT-S](https://huggingface.co/OwensLab/commfor-model-224); [c2pa-rs](https://github.com/contentauth/c2pa-rs)
- GitHub topic ai-generated-image-detection has 19 repositories, top one a 464-star awesome-list, so it is not a saturated stock project on GitHub — [GitHub topic](https://github.com/topics/ai-generated-image-detection)
- Telegram "Photo" mode re-encodes and strips EXIF including GPS while "File" mode keeps it (blog consensus; a 2022 Telegram bug report shows platform differences) — [Fast.io](https://fast.io/resources/whatsapp-telegram-signal-photo-metadata-preservation/); [Telegram bug tracker](https://bugs.telegram.org/c/19150)
- MDN lists a `torch` image-track property but no compatibility data was readable; Telegram's FAQ says nothing about metadata — [MDN MediaTrackConstraints](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints); [Telegram FAQ](https://telegram.org/faq)
- Open-Meteo's free API "is for non-commercial use, rate-limited to 10,000 calls/day"; commercial use needs a paid subscription (price not shown) — [Open-Meteo pricing](https://open-meteo.com/en/pricing)
- Uzbek pain point: Competition Committee says OSAGO insurers "refuse compensation or delay payments" (almost 1,000 appeals, over 50% OSAGO) — [Spot 18 Feb 2026](https://www.spot.uz/ru/2026/02/18/insurance-complaints/)

#### Queries run
- "AI-generated car damage photos insurance claims detection startup capture-time verification" (extended); "duplicate claim detection ... " (extended, shared with ScratchPrint).
- Fetches: VentaVid, Debevoise, Milliman, Lawyer Monthly, AUTOHAUS, Truepic, Inaza, Open-Meteo, MDN, Telegram FAQ and tech FAQ, GitHub topic.

### Inferences
- Truepic Vision, VAARHAFT SafeCam and Venta Capture already do capture-time controls, so "capture-time liveness plus detection" is not unique; the open space is physical-liveness (active illumination, parallax, sun/weather) presented as explainable evidence rather than a black-box score.
- Build risks: the torch-echo challenge depends on the web `torch` constraint, which I believe works on Android Chrome but not iOS Safari (general knowledge, not verified here), so half of the judges' phones may only reach the parallax and metadata paths; sun-azimuth checks need compass heading, unreliable next to a car body; C2PA credentials will be absent on most phones in Tashkent.
- Defensible product design per the tech notes is provenance + capture-time controls + human review with a "doubtful" bucket; publish a red-team table (generated fakes, screen replays, real photos) with per-attack catch rate and false-positive rate.
- Pitching it as tamper-proof evidence for claimants as well as insurers fits the regulator's complaint data and avoids looking like a denial machine.
- Volume [CALC]: about 33k motor claims a year makes $0.10-0.30 per claim worth $3k-10k a year in Uzbekistan; incumbents price from about $1,000 a month, so value must come from savings share or regional scope.

### Gaps
- Inaza's weather/time/location claim comes only from the VentaVid summary; Inaza's own page did not confirm it.
- VAARHAFT pages (429) and Truepic Vision product page (404) were not readable.
- Detector accuracy on current generators was not verified for any open model.
- No Uzbek evidence yet of AI-faked claim photos (Russian press only).

## insurer-3 BumperMatch: does the damage fit the other car?

### Takeaway
I found no product, startup or paper that automatically checks whether a measured damage height and shape on one car is physically compatible with the other car's bumper; the logic exists only in patents on collision reconstruction and in manual expert practice (Russian trasology, German expert reports). Novelty is real but the fraud signal from bumper height alone is probably weak.

### Cited Findings
- Patent US6885981B2 "System and method for estimating post-collision vehicular velocity changes" (priority 1998, granted 26 Apr 2005; original assignee Injury Sciences LLC, current assignee CCC Intelligent Solutions) infers override or underride from the damage pattern on both vehicles (damage above the bumper on one car with only bumper damage on the other reduces stiffness factors by about 50%) and asks the user to confirm — [Google Patents](https://patents.google.com/patent/US6885981B2/en)
- Accident-reconstruction suites such as PC-Crash and Virtual CRASH are expert-driven tools (arXiv paper using Virtual CRASH 3) — [arXiv 1512.00790](https://arxiv.org/pdf/1512.00790)
- Tractable's home page describes insurer products only as "AI-powered damage detection and assessment" with "certainty scores"; it lists no fraud, plausibility, compatibility or injury feature — [Tractable](https://tractable.ai/)
- Russian OSAGO practice: the independent technical expertise must be done by a certified expert-technician on the Ministry of Justice register; proving the original damage volume and its link to a specific accident is the hard part (search summary) (S) — [Finuslugi](https://finuslugi.ru/navigator/zastrahovat/stat_ekspertiza-posle-dtp-osago-2026)
- Rosgosstrakh (quoted as Arman Gasparyan): fraudsters offer to generate fake photos of damaged cars with neural nets, so insurers must check photos more carefully (search summary; source is among the autonews.ru / www1.ru / bfm.ru results) (S) — [Autonews](https://www.autonews.ru/amp/news/688869969a79473ec8de3c10); [www1.ru](https://www1.ru/news/2025/11/21/mosenniki-nasli-novyi-sposob-obmanut-straxovye-kompanii-s-pomoshhiu-neirosetei.amp.html)
- German query "Kompatibilitätsprüfung Unfallschaden ... KI" found no source using that term; results were generic AI fraud scoring and manipulated photos (S) — [AUTOHAUS](https://www.autohaus.de/nachrichten/schadenbusiness/versicherungsbetrug-mit-ki-wenn-bilder-luegen-3811631)
- Uzbek plate standard is "520 mm × 110 mm" with formats 01 A123BC and 01 123 ABC (Wikipedia, uncited, dates conflict) — [Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan)
- Tooling (teammate tech radar): MoGe-2 metric geometry "60 ms/image on A100 or RTX 3090" (HF tag MIT; repo licence field says Other), SAM 3 under the SAM License, fast-plate-ocr MIT and open-image-models MIT — [MoGe](https://github.com/microsoft/MoGe); [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
- Local fleet: Chevrolet 83.2% of 2025 new sales, Cobalt 30.1%, Cobalt plus Damas 55.6% (unofficial blog) — [BestSellingCarsBlog](https://bestsellingcarsblog.com/2026/02/uzbekistan-full-year-2025-chevrolet-cobalt-at-30-1-share/)
- OSAGO property limit is 28M UZS (35% of 80M) from 2026 — [Buxgalter.uz](https://buxgalter.uz/ru/publish/doc/text209644_kak_izmenitsya_stoimost_polisov_osago)
- German fraud scale: Allianz page fetched did not contain the GDV manipulation share; AUTOHAUS teasers cite a GDV estimate of 5 bn euros a year and "every tenth claim suspicious", not in the article body — [AUTOHAUS](https://www.autohaus.de/nachrichten/schadenbusiness/versicherungsbetrug-mit-ki-wenn-bilder-luegen-3811631)

#### Queries run
- "collision damage compatibility analysis photos staged accident bumper height mismatch AI" (extended)
- RU "трасологическая экспертиза совместимость повреждений нейросеть автоматически фото ДТП страховые" (standard); DE "Kompatibilitätsprüfung Unfallschaden Fotos KI ..." (standard); one more insurer-software query was refused (budget).

### Inferences
- Metric accuracy is plausible if the plate is the ruler (a 520 mm plate measured to a few percent gives centimetre-level height), but the decision value is doubtful: common passenger-car bumper heights differ by less than the proposed 15 cm threshold, load and brake dive shift ride height, and legitimate collisions between a van and a sedan are common, so the test mostly flags gross SUV/van vs low-sedan mismatches.
- No ground truth of staged versus genuine cases exists; the honest deliverable is a metric damage-geometry report plus a screen that only flags impossibilities above the measured error (validate MoGe-2 vs tape on 30 cars as the pitch proposes).
- An inverse mode (candidate vehicle classes from a damage height profile for hit-and-run) is a natural extension and has a public-sector buyer; I did not verify that need in Uzbekistan.
- Volume [CALC]: with about 33k motor claims a year and an OSAGO property ceiling of about $2.4k, a $0.30 two-car check is a tiny market even if every claim were screened.

### Gaps
- Uzbek prevalence of staged or phantom-vehicle collisions: no data found.
- German/Austrian expert-compatibility practice and vendors (for example claims service firms) not searched beyond one query; novelty could be overstated.
- IIHS low-speed bumper-test page (404) not retrieved.

## insurer-4 EuroScene: two-phone accident notice that fills itself

### Takeaway
Digital European accident statements already exist as state or industry apps (Ukraine's Diia, EASF, Belgium's Crashform, Russia's Gosuslugi Auto per the pitch), and a Gemini-based prototype can pre-fill a report; a video-first walkaround with plate OCR, damage-zone mapping and two-phone attestation was not found. I found no Uzbek no-police accident procedure, so the product depends on a regulatory pilot.

### Cited Findings
- Ukraine's Diia app lets drivers file a European accident report; details of those involved are checked automatically and the document goes to insurers; a 2026 resolution is meant to extend this to a full processing cycle (search summary of the article; page returned 403) (S) — [Mezha](https://mezha.net/eng/news/82d8df42_drivers_can_now/)
- EASF is a free app where both drivers complete the form on their own phones with OCR of licence, registration and policy and a signed PDF (vendor claims) (S); Crashform (Assuralia, Belgian insurance-industry app) pre-loads much of the data (S) — [EASF](https://easf.eu/info/car-accident-app); [Crashform](https://play.google.com/store/apps/details?id=be.assuralia.crashform)
- Chex.AI does guided 360 capture, reads VIN, plate and odometer and detects damage, but describes no accident-scene workflow — [Chex.AI](https://www.chex.ai/)
- A developer prototype takes photos, video and voice memos and generates a pre-filled report with Gemini (search summary) (S) — [DEV Community](https://dev.to/avsercankoc/road-accident-report-ai-assistant-fp0)
- Uzbek practice in a sponsored Euroasia Insurance case (May 2025): scene diagram, photos, written explanations and GUBDD certificates were attached to the claim, an independent expert assessed 74,876,496 UZS of damage, and the insurer waited for an official court decision; the article mentions no claims app — [Spot, sponsored](https://www.spot.uz/ru/2025/08/01/euroasia-insurance/)
- Uzbek regulation: from 1 Jan 2026 OSAGO insured sum is 80M UZS; the Insurance Guarantee Fund "gains adjuster functions related to notifying insurers of road traffic accidents"; the article describes no Europrotocol-style procedure — [Spot 24 Jul 2025](https://www.spot.uz/ru/2025/07/24/insurance-auto/)
- June 2026 amendments let accident victims apply to the at-fault party's insurer, which "is obliged to consider" the claim, as an alternative to direct settlement; no police or Europrotocol detail — [Spot 26 Jun 2026](https://www.spot.uz/ru/2026/06/26/insurance-amendments/)
- Policyholder must notify within 3 working days and the insurer has 15 days to decide (teammate) — [Buxgalter.uz](https://buxgalter.uz/ru/publish/doc/text209644_kak_izmenitsya_stoimost_polisov_osago)
- Complaints: almost 1,000 insurance appeals, over 50% OSAGO refusals or delays — [Spot 18 Feb 2026](https://www.spot.uz/ru/2026/02/18/insurance-complaints/)
- Uzbek plate formats per Wikipedia (two layouts) and no public Uzbek plate dataset found (teammate tech radar: needs own data or synthetic fonts) — [Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan)
- Tooling: fast-plate-ocr and open-image-models (MIT), RF-DETR Nano (Apache-2.0), Qwen3.5 (Apache-2.0 per tech radar) — [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr); [RF-DETR](https://github.com/roboflow/rf-detr)

#### Queries run
- "AI accident report app video walkaround plate recognition auto-fill European accident statement insurer" (extended)
- RU "Европротокол Узбекистан приложение без ГАИ ДТП оформление" and two variants (standard): returned only Russian, Ukrainian and Belarusian rules, nothing Uzbek; two other attempts were refused (budget).

### Inferences
- The scope is the problem: plate OCR on Uzbek layouts, damage zones, handheld-video top-down sketch (research-grade), VLM form-fill, fault templates and QR attestation in about one working day; a thin slice (plate OCR, damage zones, pre-filled form, QR handshake) is feasible, the sketch and fault suggestion are not, and a "preliminary fault" output is a liability risk.
- Because the observed Uzbek claim path leans on police certificates, an expert and sometimes a court, EuroScene is a regulatory-pilot pitch (Guarantee Fund, NAPP) more than a pure product sale; drivers (free) and fleets are the user, insurers or the Fund the payer.
- [CALC] about 33k motor claims a year at $0.40 per notice is at most about $13k a year nationally.

### Gaps
- Whether Uzbekistan has any no-police procedure for minor accidents: still unknown (three RU queries returned nothing; absence only).
- Diia details came from a search summary; e-Constat Auto (France) and Russian Gosuslugi Auto not fetched.
- Police response times and cost per call-out in Tashkent: no data.

## insurer-7 Podstava Radar: dashcam clip exposes induced rear-end crashes

### Takeaway
I found no validated product that scores induced rear-end crashes from dashcam video; fleet-camera vendors market "brake checking" detection with no method, accuracy or named product. The explainable, camera-agnostic, plate-as-ruler approach is not found, but the physics is noisy, no labelled data exists, and Uzbek prevalence is unverified.

### Cited Findings
- Nexar's insurance-fraud page presents dashcam footage as evidence people review; it describes no automatic staged-crash or brake-check detection and no reconstruction product; its statistics are mostly unsourced — [Nexar](https://www.getnexar.com/blog/how-dash-cams-can-protect-drivers-from-insurance-fraud)
- GoFleet (a fleet-camera reseller) claims AI "analyzes pre-impact vehicle positioning and movement patterns, detecting suspicious 'brake checking' or sudden lane changes", names no product and publishes no validation; its sourced figures are generic fraud totals (for example staged accidents "over $20 billion annually", Coalition Against Insurance Fraud) — [GoFleet](https://www.gofleet.com/the-exoneration-revolution-how-ai-cameras-are-saving-fleets-millions-in-false-claims/)
- A Fraunhofer-hosted paper (teammate) warns detectors give false positives and negatives and that generative AI can fabricate crash evidence — [arXiv 2510.19957](https://arxiv.org/abs/2510.19957v1)
- Fleet video telematics incumbents (Samsara, Motive, Lytx, Netradyne, Nauto) publish only vendor-promoted head-to-heads, for example a Motive-promoted 2022 test (Motive 89% detection vs Lytx 61%, Samsara 15%) (teammate saturation map 2B) (S) — [Business Wire](https://www.businesswire.com/news/home/20220421005327/en)
- An AI-agent vendor (Beam) says AI can recognise staged accidents by analysing connected-car telematics for abnormal driving behaviour (search summary of a German-language vendor article, marketing) (S) — [Beam](https://beam.ai/de/articles/detecting-and-preventing-insurance-fraud-with-ai-agents)
- Uzbek hooks (teammate): Yandex Go works through 135 fleets (Mar 2023) and 777k self-employed run private passenger transport (1 Jul 2025); no source on dashcam penetration or podstava prevalence — [KSE thesis](https://kse.ua/wp-content/uploads/2025/09/Oleh_Dykyi_RIDE-HAILING-MARKET-IN-UZBEKISTAN.pdf); [Spot](https://www.spot.uz/ru/2025/07/22/self-employed/)
- Datasets named in the pitch (DoTA, CCD, Nexar collision clips, BDD100K, CARLA MIT) contain accidents but no induced-collision labels (my reading of the pitch; not separately verified).

#### Queries run
- "brake check scam detection dashcam AI insurance fraud induced rear-end accident" (extended): no AI detector found.
- "AI dashcam analysis detect staged accident insurance fraud induced collision video startup" (standard): GoFleet marketing, Nextbase, general articles.
- RU "подстава на дороге ... нейросеть видеорегистратор ..." refused (budget).

### Inferences
- [CALC, pinhole model] For a 1080p dashcam with about 110 degrees horizontal field of view (f about 670 px) a 520 mm plate is about 17 px wide at 20 m and about 9 px at 40 m, so a 1 px error is about 1.1 m at 20 m and 4.6 m at 40 m; deceleration over a 1-2 s event is therefore noisy. Optical-expansion time-to-contact is scale-free but needs stable tracking.
- Brake-light state in daylight and glare is unreliable; at night lit taillights help but plates are often unreadable; with no labelled scam-versus-genuine data the "scam verdict" cannot be validated, and a wrong verdict is a defamation and wrongful-denial risk.
- A measurement-only report (headway, braking timeline, "was there a visible reason to brake?" flag for a human) for taxi parks is defensible; the payer is fleets, not insurers.
- [CALC, assumption] 20 parks x 200 cars x $1 per car-month = $48k a year; willingness to pay is unvalidated.

### Gaps
- Podstava prevalence in Uzbekistan and dashcam penetration among taxi drivers: not found.
- Russian-language and Chinese vendors not checked (RU query refused).
- Whether Uzbek courts or insurers accept dashcam video evidence: not checked.

## insurer-9 SleepSpot: verify and score where the car sleeps

### Takeaway
Insurers verify garaging with plate-scan sightings (Verisk GarageConfirm) or connected-car location (Smartcar), not with parking-spot video, so the approach is novel; but motor insurers do not price or inspect parking, Uzbek KASKO is shrinking with a 14% payout ratio, and there is no ground truth to score exposure.

### Cited Findings
- Verisk GarageConfirm uses a database of "four billion license plate scans" from Digital Recognition Network, measuring time, frequency and proximity of sightings near an address; Verisk estimated insurers lose about $2 billion a year to garaging fraud (2015 article; no error rate given) — [Insurance Journal](https://www.insurancejournal.com/news/national/2015/11/05/387518.htm)
- Smartcar (vendor) says insurers "lose $2.9 billion every year due to garaging misrepresentation" (citing a Verisk report) and that "more than 10% of personal auto insurance policies have verifiable garaging address defects" (no source); its method is a connected-car API checking vehicle location at night — [Smartcar](https://smartcar.com/blog/garaging-misrepresentation)
- Insurnest sells a "garaging address fraud AI agent" that cross-references declared address with digital footprints, licence records and utility data, not photos (search summary) (S) — [Insurnest](https://insurnest.com/agent-details/insurance/underwriting/garaging-address-fraud-ai-agent-in-underwriting-for-personal-auto-insurance/)
- Local hazard evidence is about falling trees, not hail: six cars damaged by trees in Uchtepa, Tashkent on 18 Mar 2022 after prolonged downpours; two cars (Chevrolet Tracker and Leapmotor C11) under a fallen tree on Makhtumkuli Street on a 2 September (year not stated) — [Podrobno](https://podrobno.uz/cat/proisshestviya/v-tashkente-upavshee-derevo-povredilo-shest-avtomobiley-/); [Anhor](https://anhor.uz/ru/news/bolshoe-derevo-upalo-na-avtomobili-tashkent)
- A Russian-language query for hail damage to cars in Tashkent returned no hail reports, only tree-fall and snow incidents (S) — [Anhor, snow and trees](https://anhor.uz/news/fallen-trees/)
- Uzbek KASKO-type premiums -47.0% to 754,268 mln UZS, payouts 107,758 mln UZS (payout ratio 14.3% [CALC]) — [NAPP Q4 2025](https://napp.uz/storage/files/shares/opendata/2025/4Q/%D0%A1%D1%82%D1%80%D0%B0%D1%85%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%28IV%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%B0%202025%20%D0%B3%D0%BE%D0%B4%20%29%20rus%20%E2%80%94%20filtr.pdf)
- Open-Meteo free tier is non-commercial; commercial use needs a paid plan — [Open-Meteo](https://open-meteo.com/en/pricing)
- Tooling (teammate tech radar): SAM 3 (SAM License, hosted at 2,000 images per Roboflow credit), DA3-Small (Apache-2.0), DINOv3 (custom, commercial allowed) — [Roboflow pricing](https://roboflow.com/pricing); [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3)

#### Queries run
- "vehicle garaging address verification auto insurance parking location AI photo" (extended): no photo-based product.
- RU "град Ташкент повреждены автомобили градобой страховые выплаты" (standard): no hail.
- A hail-alert/insurer-app query was refused (budget), so existing storm-alert apps were not checked.

### Inferences
- The pitch's baseline "versus a physical risk visit" is not a real cost: nothing in my sources shows motor insurers visiting parking spots, and Uzbek insurers have no parking rating factor to discount against.
- A 14.3% payout ratio on a book that halved means the insurers' problem is demand, not losses; selling them a $0.40 survey to cut losses does not match that pain.
- Storm-night alerts need weather data, not computer vision; verifying "guarded lot" has no pricing consequence without actuarial evidence.
- Buildable in a day (SAM 3 text prompts, depth for ramps, DINOv3 re-identification) but every score is a heuristic; "prove it works" is impossible without loss data.

### Gaps
- Hail frequency and storm-related motor claims in Tashkent: no data.
- Existing insurer hail-alert apps (Europe, US) not retrieved.
- Russian/Uzbek insurers' treatment of overnight parking in KASKO rating: not checked.

## insurer-10 WashScan: car washes as paid inspection stations

### Takeaway
Pre-wash vehicle capture exists as a car-wash liability product (Watch-It, Tec-Tel, AlphaVision, Spot AI), and baseline inspection exists as insurer/fleet software (Inspektlabs, Click-Ins, Chex.AI); I found nothing that pays car-wash staff per accepted baseline for insurers. The distribution idea is new, the insurer-paid market is small, and the car wash's own claims defence is a proven first buyer.

### Cited Findings
- Watch-It, sold by Big Brother Security, runs a "360-degree scan" of each vehicle entering the wash with images "linked to the transaction and license plate"; price, deployment count, claim-reduction numbers and insurer sharing not stated — [Car Wash Pro, 25 Jul 2023](https://www.carwashpro.com/entrepeneurs/2023/07/25/a-360-scan-before-every-car-wash/)
- Tec-Tel car-wash AI cameras cost $23, $39 and $63 per camera per month; they hold "a condition clip at entry and exit so a pre-existing scratch claim meets a timestamped defense"; its 8-camera example shows $840 a month of avoided damage payouts and recovered passes against a $312 subscription (vendor estimate) — [Tec-Tel](https://tec-tel.com/subscriptions/car-wash)
- AlphaVision advertises it can "reduce damage claims by 90%" and lists "95%, Success Defending Fraud" (marketing); no prices; no insurer data sharing mentioned — [AlphaVision](https://alphavision.ai/industries/car-wash)
- Spot AI, Coram and others sell car-wash camera analytics (search results, marketing) (S) — [Spot AI](https://www.spot.ai/solutions/car-washes)
- Baseline and old-versus-new damage are features of Inspektlabs, Click-Ins and Chex.AI (see ScratchPrint section) — [Inspektlabs](https://inspektlabs.com/blog/ai-in-motor-insurance-inspection-inspektlabs/)
- Uzbek KASKO-type contracts concluded in 2025: 237,405 (-41.0%); 660,550 in force — [NAPP Q4 2025](https://napp.uz/storage/files/shares/opendata/2025/4Q/%D0%A1%D1%82%D1%80%D0%B0%D1%85%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%28IV%20%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%B0%202025%20%D0%B3%D0%BE%D0%B4%20%29%20rus%20%E2%80%94%20filtr.pdf)
- Browser capability (teammate tech radar): WebGPU is about 86% of global browser share with full support including iOS/iPadOS 26+, Transformers.js v4 and ONNX Runtime Web are practical, but quote no phone-browser FPS without measuring — [Transformers.js v4 demos](https://huggingface.co/collections/webml-community/transformersjs-v4-demos)
- Plate OCR: fast-plate-ocr (MIT) reaches 0.32-0.68 ms per plate on an RTX 3090 with TensorRT, Uzbek formats unverified — [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
- A WashScan domain check found only a parked registrar page, so no name collision was visible — [washscan.com](https://www.washscan.com/)

#### Queries run
- "car wash AI damage detection pre-wash vehicle scan documentation claims" (extended)
- Fetches: Car Wash Pro, Tec-Tel, AlphaVision, washscan.com.

### Inferences
- [CALC] Pool ceiling: 237,405 new class-3 contracts x 5,000 UZS = 1.19 bn UZS, about $100k a year if every new KASKO policy used the service; at 10-20% share about $10-20k. Insurers alone are too small; lenders (car-loan collateral), leasing and ride-hailing fleets are plausible co-payers but unverified.
- The baseline has value only when a later claim happens; incentives for attendants and the insurer's trust in a bystander's footage (tampering by colluding wash staff) are open questions, which a capture-QC and plate/VIN binding partly address.
- Controlled conditions help: a clean, towel-dried car under a canopy has better paint visibility than roadside capture, but wet panels hide fine scratches and glare is a risk; the night/rain claim is easier here than for roadside ideas.
- Buildable in the window: blur and glare gates, orientation-sweep coverage ring, plate binding, panel crops; no model training is needed for the capture-QC core.

### Gaps
- Whether Uzbek insurers require pre-inspection for KASKO and how they do it: not found.
- Car-wash density and whether Tashkent washes face damage claims: not found.
- Humo/Uzcard micro-payout feasibility from the pitch: not verified.

## insurer-13 ImpactMeter: crush depth reveals impact severity

### Takeaway
The concept is already patented by incumbents (Mitchell: delta-V from damage images, then an injury-severity score to flag unrelated treatment; CCC via Injury Sciences-origin patents) and studied in research, and the physics defeats the target use: bumper standards sit at 4 km/h and published AI delta-V error is about 8 km/h, so a low-speed scrape cannot be separated from an injury-capable crash.

### Cited Findings
- Mitchell patent US11935129B2 (priority 14 Sep 2018, granted 19 Mar 2024): a first model estimates delta-V from damage images and vehicle data, a second model outputs an Abbreviated Injury Scale score, mapped to ICD codes and compared with the claim's injuries to flag treatment unlikely to be accident-related — [Google Patents](https://patents.google.com/patent/US11935129B2/en)
- Patent US6885981B2 (Injury Sciences, now CCC) estimates post-collision velocity changes from damage — [Google Patents](https://patents.google.com/patent/US6885981B2/en)
- USPTO application 12657638 "Estimating DeltaV from images": a deep-learning model trained on photos plus recorded delta-V from over 100,000 accidents to predict delta-V and direction of force from first-notice-of-loss photos (search summary; the PDF is a scanned image I could not read) (S) — [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12657638)
- SAE paper 2022-01-0998 "Using Artificial Intelligence to Determine the Speed Change of Frontal Collisions": trained on over 1,000 investigations, predicted speed change within ±5 mph in about 85% of a 250-case test set (search summary) (S) — [SAE Mobilus](https://saemobilus-test.cld.sae.org/papers/using-artificial-intelligence-determine-speed-change-frontal-collisions-2022-01-0998)
- 49 CFR 581.5 tests bumper protection at 1.5 mph (corner pendulum) and 2.5 mph (longitudinal pendulum and fixed barrier), i.e. about 2.4 and 4.0 km/h; bumper face-bar damage is exempt from the surface criteria — [eCFR Part 581](https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-581)
- Uzbek context (teammate): OSAGO bodily-injury share is 52M UZS of the 80M limit; over 2,700 complaints reported in the price-rise article — [Buxgalter.uz](https://buxgalter.uz/ru/publish/doc/text209644_kak_izmenitsya_stoimost_polisov_osago); [Spot 6 Feb 2026](https://www.spot.uz/ru/2026/02/06/car-insurance/)
- Tooling named in the pitch: MapAnything Apache weights via facebook/map-anything-apache (teammate) — [MapAnything](https://github.com/facebookresearch/map-anything)

#### Queries run
- "AI estimates crash severity or delta-V from vehicle damage photos to flag exaggerated low-impact injury claims" (standard): the patents and SAE paper above.
- Injury Sciences and CCC pages were not fetchable (SSL error / 403 / 404), so the current commercial status of CCC's injury-analytics products is unverified.

### Inferences
- [Inference from the bumper standard and the SAE error band] A ±3 cm crush bin and a ±5 mph (about ±8 km/h) delta-V band are larger than the 0-10 km/h regime the claim screen is meant to separate; NHTSA crash-test crush profiles named in the pitch are mostly high-speed tests (general knowledge, not verified here).
- Automated flagging of injury claims as exaggerated is a wrongful-denial and regulatory liability; Uzbek bodily injury is determined medically, not by crush depth.
- The useful kernel (metric damage dimensions from plate-scaled photos) belongs inside BumperMatch.

### Gaps
- No validation study of photo-based delta-V for separating genuine from exaggerated low-impact injury claims was found in the search results (absence only).
- CCC and Mitchell commercial product availability for this function: unverified.
- IIHS low-speed test data (the real low-speed ground truth) not retrieved (404).
- Content of USPTO 12657638 beyond the search summary.

## insurer-15 GeoProof: where was this Telegram photo really taken?

### Takeaway
General image geolocation exists (PIGEON, GeoSpy, GeoCLIP-class models) but stops at city level in open benchmarks, and capture-time location verification already exists (Truepic); street-level matching works only inside a pre-filmed index. I found no payer, and the premise that insurers receive GPS-less Telegram photos is avoidable by asking for files or in-app capture.

### Cited Findings
- PIGEON places "over 40% of its guesses within 25 kilometers of the target location globally"; PIGEOTTO beats prior art by up to 7.7 points at city level; the abstract reports no 1 km street-level figure — [arXiv 2307.05845](https://arxiv.org/abs/2307.05845)
- GeoSpy claims "up to meter level accuracy" with "No metadata required", is "Available for qualified government and enterprise organizations", with no pricing and no insurance use case on the page — [GeoSpy](https://geospy.ai/)
- GitHub topic image-geolocation has 12 repositories (top one about 1.6k stars), so it is niche rather than a stock hackathon project — [GitHub topic](https://github.com/topics/image-geolocation)
- Truepic's insurance capture collects "verified time, date, device, and location metadata" with "Location authentic" checks — [Truepic](https://www.truepic.com/)
- Telegram "Photo" mode re-encodes and strips EXIF/GPS while "File" mode keeps it, with platform-dependent exceptions (blog consensus and a bug report) — [Fast.io](https://fast.io/resources/whatsapp-telegram-signal-photo-metadata-preservation/); [Telegram bug tracker](https://bugs.telegram.org/c/19150)
- German trade press names illogical light, shadows or perspective and date mismatches as manipulation red flags — [AUTOHAUS](https://www.autohaus.de/nachrichten/schadenbusiness/versicherungsbetrug-mit-ki-wenn-bilder-luegen-3811631)
- No evidence found that Uzbek insurers accept claim photos through Telegram; Kafolat's app only uploads documents (teammate, WebCatalog) and a sponsored Euroasia case shows paper documents and police certificates — [WebCatalog](https://webcatalog.io/en/apps/kafolat); [Spot, sponsored](https://www.spot.uz/ru/2025/08/01/euroasia-insurance/)
- Tooling (teammate tech radar): SigLIP 2 (Apache-2.0), DINOv3 (custom, commercial allowed), LightGlue-class matchers — [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224)

#### Queries run
- "image geolocation verify claim photo location fraud insurance street-level matching GPS stripped" (extended) and three standard variants: all refused (budget). Evidence above comes from page fetches only; the novelty rating is provisional.

### Inferences
- A FAISS index of a 1 km grid (about 50k frames) is a toy; it demonstrates the method only for photos taken inside the filmed area, and judges photographing outside it get "unknown". Reference-imagery licence and Tashkent coverage (Mapillary, panoramas) were not verified.
- Roadside-assistance callers can share a location pin or call; insurers can require file uploads or in-app capture, so the pain is mostly self-inflicted.
- [CALC] about 33k motor claims a year at $0.10 per photo set is about $3k a year nationally.
- Best use is a "place check" feature inside ProofShot (capture-time GPS plus sun azimuth), not a product.

### Gaps
- No successful web search for competitors; Niantic/Google visual positioning and claims-specific vendors were not checked.
- Mapillary/KartaView coverage of Tashkent: not checked.
- Street-level (1 km) accuracy of any open model in dense urban scenes: not found.
