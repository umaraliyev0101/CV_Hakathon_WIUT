# Big scalable CV startup bets

WIUT Hackathon 26, CV track, 11 Oct 2026. A short follow-up to *Automotive CV hackathon ideas*: fewer ideas, each one a large category that investors already fund.

## What makes an idea "big" here

Each idea below passes five tests that most of the first list failed:

- **Proven category.** A company abroad has raised $50M+ or reached unicorn or IPO scale doing it.
- **One model, many buyers.** The same core model sells to three or more customer types.
- **Recurring or per-transaction revenue**, not one-off projects.
- **Data that compounds.** Every new customer makes the model better and harder to copy.
- **A clear path abroad.** Central Asia and the CIS drive the same Chevrolet and Chinese cars, so the model carries over with little retraining.

## Ranked ideas

**1. Plate Pay: your number plate is your wallet**
A camera reads the plate at a parking gate, fuel pump, car wash or toll and charges a linked bank card. No stop, no ticket, no app. Proof: Metropolis does this in US parking and raised $1.6B in November 2025 at about a $5B valuation [1]. Why it scales: revenue is a cut of every payment, and each new kind of site (parking, fuel, wash, service, toll) reuses the same plate reader and wallet. Uzbekistan has 4.98M private cars [7]. Fits the room: Uzcard, MKBANK and the Central Bank are hackathon partners. Tech hook: reading local plates at night and in rain. Demo: phone camera at a mock gate, plate read, payment receipt.

**2. Car Condition API: one phone walkaround for every car deal**
A phone walkaround returns a damage map, condition grade and repair estimate in under a minute, sold per check through an API. One model serves many buyers:
- marketplaces (Avtoelon, OLX)
- banks checking car-loan collateral
- insurers writing KASKO
- dealers pricing trade-ins (BYD Royal Motors)

Proof: UVeye raised about $380M, its latest round led by Toyota's Woven Capital [2]. Tractable is a unicorn [5]; Spinny ($1.5B) and Cars24 now inspect with AI [6]. Uzbekistan sees 52–57k used-car sales a month and has no vehicle-history product [7]. Moat: the largest labelled damage set for local and Chinese cars. Demo: a judge scans a parked car with their phone.

**3. Phone Dashcam: Motive-grade fleet safety at a tenth of the price**
An app turns the driver's phone into an AI dashcam. It flags phone use, drowsiness, tailgating, red-light running and near-misses, and scores every driver. Fleets (taxi parks, couriers, trucking) cut crashes, and insurers buy the scores to price policies. Proof: Motive filed for a NYSE IPO with $501M recurring revenue [3]; Samsara is public. Video telematics is forecast to grow from about $2B in 2025 to $8.7B by 2034 [4]. Gap: their $300+ hardware doesn't fit emerging-market fleets. Tailwind: the EU made driver-distraction warning systems mandatory in new cars from July 2026 [8]. Demo: live alert when the driver looks away or picks up a phone.

**4. Claims Autopilot: motor claims settled from a phone (Crash Twin, scaled up)**
Photos or video of a crash become a damage estimate, a fault decision and a fraud check, so the insurer pays in hours instead of weeks. Proof: insurtech raised $2.44B in Q2 2026, 99% into AI [9]; Tractable became a unicorn on photo estimates [5]. Local pain:
- OSAGO covers 6.2M policies and was repriced 3–4x in January 2026.
- It draws over half of all insurance complaints to the regulator.
- Insurers must decide a claim within 15 days [7].

Scale: insurers across Central Asia, then leasing and rental. This is Crash Twin, sold as claims automation. Demo: two crash photos in, estimate and fault summary out.

**5. Road Video Search: Google for what fleet cameras saw**
Dashcam footage from fleets and cities becomes searchable in plain language, using vision-language models: "pedestrian crossing on red at night", "overloaded truck". Fleets find incidents in seconds and insurers pull evidence. Car makers and self-driving teams buy rare scenes from roads they have never mapped. Proof: Nomadic raised an $8.4M seed in March 2026 for this [10]. Nexar ($150M raised) agreed to merge with Nauto in July 2026 to build a road-data platform [11]. Central Asian traffic (Damas vans, loose lane discipline, livestock) is missing from global driving datasets. This idea impresses technical judges most. Risk: the data buyers are far away. Demo: type a query, get matching clips.

**6. City Eyes on Wheels: buses that inspect roads and clear bus lanes**
Cameras on buses and garbage trucks log potholes, faded markings, broken streetlights and cars blocking bus lanes, each with GPS and a photo, into a city work-order queue. Proof:
- Hayden AI runs bus-mounted enforcement in New York, Los Angeles and Barcelona, and won UK approval in 2026 [12].
- Asimob pilots road inspection in 17 countries [11].
- Michelin bought RoadBotics.

Uzbekistan already runs 1,831 enforcement camera and radar points [7]. Scale: per-kilometre fees from every city in the region. Risk: slow government sales; pitch it as cameras the city never buys. Demo: upload a ride video and get a map of defects.

## Which one to pick

- **Strongest with VC judges and with the partners in the room:** #1 Plate Pay (payments plus computer vision).
- **If the team is already building Crash Twin:** keep the build and change the pitch. Present it as #4 (claims) now, growing into #2 (every car transaction) next.
- **Most impressive to technical judges:** #5 Road Video Search.
- **Avoid:** fixed-camera traffic counting at a junction. That was the elimination task, and judges have already seen it many times.

## Caveats

- Funding figures come from search results. Re-check any number before quoting it on stage.
- Motive's IPO timing and valuation are unsettled [3], and UVeye's round date is unclear [2].
- Uzbek figures come from the earlier notes in `research_notes/Automotive CV hackathon ideas/uzbekistan_market.md` [7].

## Sources

1. Metropolis $1.6B raise, $5B valuation: [Crunchbase News](https://news.crunchbase.com/venture/ai-powered-parking-platform-metropolis-seriesd/), [Wikipedia](https://en.wikipedia.org/wiki/Metropolis_Technologies)
2. UVeye $191M package, $380.5M total: [AJOT](https://www.ajot.com/news/ai-driven-vehicle-inspection-startup-uveye-raises-191-million-in-equity-debt), [CO/AI](https://getcoai.com/news/ai-vehicle-inspection-startup-uveye-raises-191m)
3. Motive IPO filing and ARR: [Fintool](https://fintool.com/news/motive-ipo-fleet-management), [TechRepublic](https://www.techrepublic.com/article/news-motive-ipo/), [FreightWaves](https://www.freightwaves.com/news/motive-raises-valuation-to-2-85b-after-closing-150m-funding-round)
4. Video telematics market forecast: [GM Insights](https://www.gminsights.com/pressrelease/video-telematics-market)
5. Tractable unicorn: [Beinsure](https://beinsure.com/news/insurtech-tractable-hits-unicorn-status/), [eeNews Europe](https://www.eenewseurope.com/en/first-computer-vision-ai-unicorn-for-financial-services/)
6. Spinny and Cars24 AI inspections: [Business Today](https://www.businesstoday.in/amp/auto/story/how-cars24-is-reinventing-itself-as-an-ai-first-used-car-platform-544318-2026-07-21), [Caplight Spinny](https://www.caplight.com/company/spinny), [Entrackr](https://entrackr.com/news/spinny-raises-131-mn-led-by-accel-leaders-fund-8910456)
7. Uzbekistan fleet, used-car, OSAGO and camera figures: earlier research notes (Statistics Committee, NAPP Q4 2025, Spot.uz, Kun.uz; links inside `uzbekistan_market.md`)
8. EU driver-distraction (ADDW) mandate: [Smart Eye](https://smarteye.se/news/advanced-driver-distraction-warning-systems-now-mandatory-across-all-new-eu-vehicles/), [Traffic Technology Today](https://www.traffictechnologytoday.com/news/enforcement/eu-mandates-driver-distraction-warning-tech-in-cars.html)
9. Insurtech funding Q2 2026: [Insurance Business](https://www.insurancebusinessmag.com/reinsurance/news/breaking-news/ai-startups-capture-record-95-2-of-insurtech-funding--gallagher-re-574376.aspx), [Risk & Insurance](https://riskandinsurance.com/ai-dominates-insurtech-funding-in-q2-as-early-stage-deals-cool-sharply/)
10. Nomadic seed: [TAM Radar](https://www.tamradar.com/funding-rounds/nomadic-seed-8-4m), [parsers.vc](https://parsers.vc/news/260403-nomadic-unleashes-searchable-video-ai-for/)
11. Nexar and Nauto merger, Asimob pilots: [Yespress](https://yespress.io/nexar-inc), [Kurrant](https://kurrant.com/?p=47198)
12. Hayden AI deployments: [Welcome to the Jungle](https://app.welcometothejungle.com/jobs/Cqz10Z_V/company)
