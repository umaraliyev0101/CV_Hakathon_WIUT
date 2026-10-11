# Showroom Pulse: pitch notes

For the jury and the Royal Motors / Ducati conversation. Every number about our accuracy comes from
`docs/EVAL.md`; every number about money below is an **ASSUMPTION** to be replaced with the dealer's
data.

## The customer-side story

A BYD or Toyota showroom on a Saturday: twelve people walk in, six stop at the Song Plus, three stay
more than a minute, and nobody walks over to them because both salespeople are at the desk. Nobody
records it. The dealer's report says "42 test drives, 11 sold", and has no way to know that the
interest was there and went unserved.

Showroom Pulse turns the camera the dealer already owns into four answers for the general manager and
the marketing lead:

1. Which cars attract attention, and which get ignored (stop rate, dwell, repeat visits, rank).
2. How customers move: the common routes, how long visits last, when the floor gets crowded.
3. **Where interest goes unserved**: "Car X: 5 of 7 interested customers got no salesperson, between
   14:00 and 15:00."
4. What to do about it, each recommendation with its evidence numbers, a confidence level and links to
   the exact clip moments, so a manager can check it in ten seconds instead of trusting a dashboard.

It is anonymous and aggregated: no faces recognised, no demographics, no tracking across visits, faces
blurred in every output, processing on the premises. Decision support only; per-salesperson coaching is
the sister product, Showroom Coach, which merges with this one.

## What is new compared with what exists

Camera retail analytics is an existing market. We say so plainly and name the differences.

| | V-Count | FootfallCam | Deepingsource | Palexy | Showroom Pulse |
|---|---|---|---|---|---|
| What they sell (as described in our design research; verify before quoting) | Dwell per car, heatmaps, staff exclusion for showrooms | Dwell per car, heatmaps, staff exclusion for showrooms | Counts, paths, dwell, funnels and family detection in a Korean car showroom | Time-to-greet and staff-interaction metrics | Customer-experience analytics from one camera |
| 1. Car-specific meaning (attention per car model, stop rate, repeat visits, reaching the sales desk) | Dwell per car | Dwell per car | Zone-level paths and funnels | Not their focus | Per-car stop rate, median and total dwell, repeat visits, desk reach, attention rank |
| 2. Unserved interest (customer interest joined to where staff are, in one engine) | Not found in their public offer; staff are excluded, not joined | Same | Not found | Staff-interaction side (time-to-greet); not framed per car | Episodes plus a live alert; the bridge to Showroom Coach |
| 3. Recommendations with evidence (numbers plus links to clip moments) | Not found; dashboards | Not found; dashboards | Not found; dashboards | Not found; metrics | Rules R1-R7, each with evidence, confidence and a clip moment |
| 4. No new hardware (existing CCTV or a phone video) | Sold with sensors | Sold with sensors | Not stated | Not stated | Any video file or a laptop webcam |
| 5. Open evaluation (measured accuracy on public data, published) | Not found | Not found | Not found | Not found | `docs/EVAL.md`, every missed target reported |

Competitor cells come from desk research done for the design spec, not from testing their products, and
"Not found" means we did not find it in public material, not that it does not exist; the last column is what this repository actually does. Our honest weak spot is the
same as the strength of established vendors: they have years of field data, and we have six school clips.

## Where we stand on accuracy (say this before the jury asks)

From `docs/EVAL.md` (MEVA indoor clips, not showrooms): entry/exit count error 50.0% on test (target
<= 10%, **missed**; 3.6% on dev); occupancy MAE 4.02 (target <= 0.5, **missed**); zone dwell median
error 42.0% (target <= 10%, **missed**); journey accuracy 52.5% (target >= 80%, **missed**); live mode
8.27 fps engine, 5.02 fps paced smoke (target >= 5, met); file mode 1.55x real time on a quiet
excerpt but 0.88x and 0.92x on two crowded 5-minute clips (partly met). The ground truth is a
poor match to what we measure (activity-level annotation of people in a school), which explains much of
the gap, but it also means **we do not yet have a showroom accuracy number**. The honest claim is:
the pipeline works end to end, is evaluated in the open, and the first thing to do with Royal Motors
footage is measure it.

## ROI model

Formula (annual, per showroom):

`value = missed-interest episodes per month x conversion uplift x margin per car x 12`

Every input below is an **ASSUMPTION**; none is measured, none comes from Royal Motors. The ranges are
placeholders chosen to show orders of magnitude, not benchmarks.

| Input | Meaning | ASSUMPTION: placeholder range | Where to get the real number |
|---|---|---|---|
| Missed-interest episodes per month | Customers who dwell at a car for the deep-interest time with no staff nearby (Pulse's "unserved episodes") | ASSUMPTION: 40-200 per showroom per month | Run Pulse on 1-2 weeks of the dealer's own CCTV (`stats.store.unserved_episodes`), scale to a month |
| Share that would have been served after acting on the recommendations | How many of those could realistically get a salesperson | ASSUMPTION: 30-60% | Pilot: change staffing for 2 weeks, re-measure the unserved share |
| Conversion rate of a served deep-interest customer to a sale | Purchase probability once served | ASSUMPTION: 3-8% | Dealer CRM: test drives and sales per served enquiry |
| Conversion rate of an unserved deep-interest customer | Baseline without contact | ASSUMPTION: 0.5-2% | Dealer CRM / walk-in leads vs sales |
| Conversion uplift | Served minus unserved conversion | ASSUMPTION: 2-6 percentage points | Derived from the two rows above |
| Margin per car | Gross margin on one car sold | ASSUMPTION: USD 1,500-4,000 (Toyota/BYD) | Royal Motors finance (per model) |
| Cost | Software is local and open; the real cost is camera setup and staff time | ASSUMPTION: near zero hardware, 1 analyst day per month | Dealer IT |

Worked example, **entirely from the ASSUMPTION midpoints**: 100 episodes/month x 45% served x 4 pp uplift
x USD 2,750 = about USD 5,000 per month, about USD 60,000 per year. Replace every input before quoting a
figure; with the lowest ends of every range the result is under USD 1,000 per month, and the model is
only worth presenting as "this is how you would compute it with your numbers".

## Questions for Royal Motors

1. Can we have 1-2 weeks of anonymised CCTV from one Toyota and one BYD showroom (or a staged
   afternoon), so we can measure accuracy on the real layout and camera angles?
2. How many cameras per showroom, where are they mounted, what resolution and frame rate, and does the
   feed leave the building? (Our tool runs on-premise on a single machine.)
3. Do salespeople wear a uniform or badge? (Our staff auto-suggest works from outfit similarity, contacts
   and presence; a uniform makes it better, and mixed clothing means the one-click correction matters.)
4. What do you record today: enquiries, test drives, sales per model per week? We can join a simple
   `car_model,period,sold` file to attention numbers.
5. What is the margin per car per model, the sales cycle, and what share of sales comes from walk-ins?
   (These replace the ROI assumptions above.)
6. Who would act on a recommendation (floor manager, marketing), how often would they want it (daily,
   weekly), and in which language (English, Russian, Uzbek)?
7. Legal and consent: how does the group inform visitors that analytics is in use (signage), what are
   the local privacy rules for CCTV analytics, and who signs off retention? (Our default deletes
   uploads after 24 hours.)
8. For Ducati: showroom layout with bikes (motorcycle class is detected) and a smaller floor; is the
   question the same (unserved interest per bike)?
9. Would the dealer let us pilot a change (staffing at a flagged car for 2 weeks) so we can measure the
   effect instead of predicting it?

## Demo script (about 4 minutes)

Prepare beforehand: `./run.sh`, open http://localhost:8000, have the stock clip
`data/stock/wm-bangalore-04.mp4` (CC BY-SA 4.0, Gpkp, Wikimedia Commons; not redistributed) or the jury's
own clip ready, and note that the first run of a new video takes a while (perception is cached after).

1. **Upload** the clip, choose the **Short clip** preset (for a long CCTV file choose Dealership).
   While the progress bar runs, tell the story above. Point at the banner: "Anonymous, aggregated
   analytics; decision support only."
2. **Zones**: the keyframe shows auto-detected car zones. Drag a vertex, rename one "Car: BYD Song
   Plus", add a rectangle named "Desk". Click "Looks right".
3. **Confirm staff**: the face-blurred thumbnails with the suggested role and reasons; flip one with a
   click. (Or press Skip and show the warning banner "Roles skipped: everyone counted as a customer".)
4. **Results**: KPI tiles, the per-car table (stop rate, dwell, unassisted share, rank), heatmap, journey
   flow, occupancy chart, then **recommendations**: open one, show its evidence numbers and the contact
   sheet / clip moment. Play the annotated video: customers in orange with dwell timers, staff dimmed
   blue, faces blurred.
5. **Re-score live**: delete or move a zone, watch the numbers change in seconds with no re-detection
   (that is the cache).
6. **Live page** (`/live`): allow the webcam, draw a zone on the first frame, stand in it; the dwell
   timer runs and, past the alert time, the "customer at ... no staff nearby" alert appears. Click a
   person to flip staff/customer.
7. **Evaluation**: open `docs/EVAL.md`, show the Targets table with the missed targets, and say what we
   would measure first on Royal Motors footage.
