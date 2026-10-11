# Showroom Coach: pitch notes (YAK-44, WIUT Hackathon 2026 Computer Vision final)

For the team building the deck and rehearsing the talk. The jury scores: technical complexity 25%, innovation 25%,
impact 25%, presentation 15%, ROI 10%. The audience includes Toyota Royal Motors, BYD Royal Motors (Royal Motors Group)
and Ducati Tashkent.

Ground rules for everything said on stage:

- Quote only numbers that are in [PROGRESS.md](../PROGRESS.md) or [EVAL.md](EVAL.md). Every ROI input below is an
  assumption until Royal Motors replaces it, and the slide must say so.
- Do not say we are "better than SKAIVISION". Say we are open where they are silent, and cheap to start.
- Do not claim staff-detection accuracy, greet-time accuracy, or showroom results. None exist yet (no showroom footage was
  available; no partner footage was used).
- Do not say "legally compliant". Say "designed as decision support, with the staff informed, under the 2026 AI law
  (ZRU-1115)", and that counsel should review before a pilot.

## 1. Deck outline (12 slides, sized for about 8 minutes; trim to the slot you are given)

| # | Slide | On the slide | Say (key points) | Serves |
|---|---|---|---|---|
| 1 | Title | "Showroom Coach: who was greeted, who waited, who walked out." Team, YAK-44. | One sentence: one camera in, coaching data per salesperson out, on a laptop. | presentation |
| 2 | Problem | A customer walks in. Nobody says hello. They leave. Four questions a manager cannot answer today: how long did they wait, was someone free, did they leave unattended, how long did each conversation last. | The cost of an ungreeted walk-in is a lost sale nobody saw. Managers cannot watch every customer. (Start the demo upload now, see slide 3.) | impact |
| 3 | Demo A: upload | Live product: upload clip, pick preset, confirm staff, results (annotated video, timeline, scorecard, alerts). | Narrate the four screens. Change one role and show the instant re-score. | innovation, presentation |
| 4 | Demo B: live | `/live` page with the webcam: boxes, timers, click a person to flip staff/customer. | Same engine as upload. Works in the browser with no install on the camera side. | technical complexity, presentation |
| 5 | How it works | The pipeline diagram from the README (sources, perception, identity, geometry, roles, events, metrics, render). | CPU only: YOLOX-nano on two tiles, ByteTrack, RTMPose-s, DINOv2-small. Distances in metres with no calibration (pinhole model). One state machine for greet, wait, engagement, walk-out, party, handoff. | technical complexity |
| 6 | Innovation: staff versus customer | Four features, one score, plain-language reasons, one click. A card: "Suggested: staff (78%). Same outfit as 2 others. Talked to 4 people." | The hard part of any showroom analytics is who is staff. The commercial leader does not disclose how. Ours is open, explainable, and a person has the last word. | innovation |
| 7 | Compared with SKAIVISION | The comparison table in section 2 (pipeline, engagement rule, hardware, privacy, staff detection, cost). | Same measurement logic and the same 2 m / 2 min rule in our Dealership preset. They have scale and GPUs and 150+ dealerships; we run on a laptop and open the staff question. | innovation, impact |
| 8 | Privacy by design | No face recognition. Faces blurred in every frame, thumbnail and live canvas. On-premise. Deleted after 24 h. Staff informed. Decision support only (ZRU-1115). | Privacy is the reason a showroom says yes. Appearance is only used to merge track fragments within one video, never stored as identity. | impact, technical complexity |
| 9 | Results (honest) | Measured numbers from PROGRESS.md and EVAL.md only. See section 3. Plus a plain "What we have not measured yet". | Say the misses out loud: file mode ran at 0.976x and 0.713x real time, not 1x; live mode is 5.2 to 6.3 fps for small scenes and about 3.2 fps for crowded ones. No showroom data yet. Honesty is a feature here. | technical complexity, presentation |
| 10 | ROI | The formula and one worked example, all inputs labelled "ASSUMPTION — replace with Royal Motors numbers". Break-even in cars per month. | Not a forecast. A model the partners fill in. At the example inputs the system pays for itself with about one extra car every four months. A pilot measures the real inputs. | ROI, impact |
| 11 | Roadmap | Pilot, real numbers, sales-data join, car zones, calibration in the UI, Uzbek and Russian UI, multi-camera. | Next two weeks versus next two quarters. | impact |
| 12 | Ask | A two-week pilot in one showroom: one camera, staff informed, a manager to review scorecards, the real ROI numbers. | What we need from Royal Motors Group and Ducati Tashkent, and what they get back. Close on the one-line promise. | impact, presentation |

### Demo script (about 3 minutes, slides 3 and 4)

1. Before going on stage: run the demo clip through the app once, and keep that results page open in a second tab (job
   data stays for 24 h). It is the fallback if the live analysis is slow or the network misbehaves.
2. During slide 2, upload the clip on the main tab. Analysis takes about as long as the clip on this laptop (the 60 s
   clip took 61.5 s), so a 60 s clip is ready by slide 3. A crowded 5-minute clip took 421 s: ask the jury for a clip of
   about a minute, or trim it.
3. Confirm staff: point at one reason line, flip one role with one click, press "Looks right".
4. Results: play the annotated video, show the timer over a waiting customer, the alert feed, the scorecard. Flip a
   role again to show the re-score taking a fraction of a second.
5. `/live`: allow the camera, wait for two or three people to be confirmed, click one to flip the role, show the timers.
6. If the jury brings a video: mp4, mov or avi, up to about 5 minutes. Pick the "Short clip" preset for clips under a
   minute and "Dealership" for real dealership footage.
7. If the webcam page fails: `scripts/live_file_client.py` feeds a video file through the same WebSocket, or fall back to
   the pre-run results tab.

## 2. SKAIVISION comparison

Source for the SKAIVISION column: the CTO's slides at Embedded Vision 2025,
<https://www.edge-ai-vision.com/wp-content/uploads/2025/09/BW04_Fayling_SKAIVISION_2025.pdf>. We have only those public
slides; anything they do not state is marked "not stated".

| Aspect | SKAIVISION (US, 150+ dealerships) | Showroom Coach |
|---|---|---|
| Pipeline | Detect people every frame; track across cameras; compute relationships and distances between tracked entities; turn events into alerts | Detect (YOLOX-nano on two tiles), track (ByteTrack), pose (RTMPose-s), merge fragments by body appearance, distances from a calibration-free pinhole model, an event state machine, alerts. One camera per job; no cross-camera tracking yet |
| Engagement rule | Employee within 2 m of a customer for over 2 minutes | The same rule is the "Dealership" preset (2.0 m / 120 s). A "Short clip" preset (2.0 m / 8 s) exists for short demo clips. All thresholds are in one config file |
| Waiting customers | Each customer gets an ID on entry and a timer runs until an employee interacts | Same idea: timer from entry, wait alert after 60 s (Dealership) that also names the staff who were free at that moment, and a walk-out event |
| Tracking quality | Report that ByteTrack and BoT-SORT score 90% on public MOT benchmarks but about 65% in dealerships | We use ByteTrack plus an appearance-based merge of broken tracks. Our MEVA tracking numbers are in [EVAL.md](EVAL.md). We have no dealership footage, so no like-for-like comparison |
| Hardware | An edge box with an 8 GB RTX 3050/4060 handles 15 SD cameras | One CPU-only laptop (Intel i5-10210U, no GPU), one camera. Measured: 0.976x real time on a 60 s clip, 0.713x on a crowded 5-minute clip; live 5.2 to 6.3 fps with about 4 people. Not like-for-like: they have scale, we have a low entry cost |
| Privacy | Silhouettes only, no faces | No face recognition. Faces blurred in every rendered frame, thumbnail and the live canvas. On-premise, nothing leaves the machine, data deleted after 24 h |
| Staff versus customer | Not stated in the slides | Open and explainable: four visible features, plain-language reasons, one click to confirm or flip. A person's correction always wins |
| Cost | Not stated in the slides | Permissive open-source stack (Apache-2.0 code, no licence fee); cost is the hardware, set-up and support. See the ROI model |
| Maturity | In production in 150+ dealerships | A hackathon prototype. Built and measured on public data (MEVA); no showroom footage was available |

What to say: the measurement logic is the same, the staff question is open, and the entry cost is a laptop.
What not to say: that we match their accuracy, scale or maturity.

## 3. Results slide: what may be quoted

From [PROGRESS.md](../PROGRESS.md), measured on the i5-10210U laptop:

- Detector: YOLOX-nano on two tiles, 36 ms per frame and 26.8% recall, against YOLOX-s on two tiles at 361 ms and 30.9%
  (MEVA school clip; recall is understated for every model because MEVA only boxes annotated actors).
- File mode: real-time factor 0.976 (60 s clip, about 11 people) and 0.713 (5-minute clip, about 12 people). Both below
  the 1x goal.
- Rendering the annotated video: 7.4x real time. Re-scoring after a role change: 0.07 to 0.1 s.
- Live mode: 5.2 to 6.3 fps with about 4 people; about 3.2 fps with 11 to 15 people.
- Distance model: within 0.1% on synthetic scenes (maths only; real accuracy not measured).

MEVA evaluation numbers: see [EVAL.md](EVAL.md). Copy them onto the slide from there, with their caveats.

Always add the line "Not yet measured: staff-suggestion accuracy and greet-time error on real showroom footage".

## 4. ROI model

The ROI weighs 10% of the score, but it is also where a sceptical partner decides if this is a toy. Keep it simple,
show the formula, and make every input visibly theirs to change.

### Inputs

Every value in the "Example value" column is hypothetical. Replace them with Royal Motors numbers before presenting
them as anything else. Currency is US dollars for convenience; convert to UZS as needed.

| Symbol | Input | Example value (hypothetical) | Status |
|---|---|---|---|
| W | Daily walk-ins per showroom | 30 | ASSUMPTION — replace with Royal Motors numbers |
| u | Share of walk-ins who leave without ever being greeted | 15% | ASSUMPTION — replace with Royal Motors numbers (the product measures this in a pilot) |
| c_g | Close rate of visitors who are greeted | 8% | ASSUMPTION — replace with Royal Motors numbers |
| c_u | Close rate of visitors who are not greeted | 3% | ASSUMPTION — replace with Royal Motors numbers |
| G | Average gross profit per car sold | $1,200 | ASSUMPTION — replace with Royal Motors numbers |
| r | Share of walk-outs who would stay and be greeted if staff reacted faster | 20% | ASSUMPTION — replace with Royal Motors numbers (the weakest input: measure it in the pilot) |
| D | Selling days per month | 26 | ASSUMPTION — replace with Royal Motors numbers |
| S | Subscription or support cost per showroom per month | $300 | ASSUMPTION — replace with Royal Motors numbers |
| H | One-off hardware and camera set-up per showroom | $700 | ASSUMPTION — replace with Royal Motors numbers |
| M | Months over which H is spread | 36 | ASSUMPTION — replace with Royal Motors numbers |

### Formula

```
walk-outs per day             = W x u
recovered visitors per day    = W x u x r
extra cars per month          = D x W x u x r x (c_g - c_u)
extra gross profit per month  = extra cars per month x G
monthly cost                  = S + H / M
net benefit per month         = extra gross profit - monthly cost
ROI                           = net benefit / monthly cost
break-even extra cars / month = monthly cost / G
```

Logic: only the customers the system would have lost and a faster greeting wins back are counted (r), and each of them
adds the difference between a greeted and an ungreeted close rate, not a full sale.

### Worked example (all inputs hypothetical, from the table above)

```
walk-outs per day            = 30 x 0.15                      = 4.5
recovered visitors per day   = 4.5 x 0.20                     = 0.9
extra cars per day           = 0.9 x (0.08 - 0.03)            = 0.045
extra cars per month         = 0.045 x 26                     = 1.17
extra gross profit per month = 1.17 x $1,200                  = $1,404
monthly cost                 = $300 + $700 / 36               = $319.44
net benefit per month        = $1,404 - $319.44               = $1,084.56
ROI                          = $1,084.56 / $319.44            = 3.4x
break-even                   = $319.44 / $1,200               = 0.27 extra cars a month,
                                                                about one extra car every four months
```

### Sensitivity (all hypothetical)

Three scenarios. Low and high move every input at once, so they bracket the plausible range rather than forecast it.

| Input | Low | Base | High |
|---|---|---|---|
| W walk-ins per day | 20 | 30 | 50 |
| u left ungreeted | 10% | 15% | 25% |
| r recoverable | 10% | 20% | 35% |
| c_g / c_u | 5% / 3% | 8% / 3% | 10% / 3% |
| G gross per car | $900 | $1,200 | $1,800 |
| S per month / H one-off | $450 / $1,000 | $300 / $700 | $200 / $700 |
| **Extra cars per month** | 0.10 | 1.17 | 7.96 |
| **Extra gross profit per month** | $94 | $1,404 | $14,333 |
| **Monthly cost** | $478 | $319 | $219 |
| **Net benefit per month** | -$384 | $1,085 | $14,113 |
| **ROI (net / cost)** | -0.8x | 3.4x | 64.3x |
| Break-even extra cars per month | 0.53 | 0.27 | 0.12 |

Read it honestly: the low case loses money, and the high case is an upper bound, not something to promise. The model
turns on two numbers nobody has yet.

One input at a time, others at base (net benefit per month):

| Input | Low value | Net at low | High value | Net at high |
|---|---|---|---|---|
| W walk-ins per day (u gives the same result: they multiply) | 20 | $617 | 50 | $2,021 |
| r recoverable share | 10% | $383 | 35% | $2,138 |
| c_g, with c_u at 3% | 5% | $242 | 10% | $1,646 |
| G gross per car | $900 | $734 | $1,800 | $1,787 |
| S subscription per month | $450 | $935 | $200 | $1,185 |

The result moves most with r (how many walk-outs are really recoverable) and the greeted-versus-ungreeted close-rate
gap. Cost barely matters. So the pilot should measure those two, not argue about the price.

### What the product measures, and what it cannot

- Measured by the product in a pilot: walk-ins, the share who left unattended (u, with the detector's own error), time
  to greet, waits while staff were free.
- Needs Royal Motors data: c_g and c_u (CRM or a sales CSV by salesperson and date), G, D.
- Needs an experiment: r. Compare walk-out share and sales before and after coaching, or between two comparable
  showrooms or shifts.

### Not counted (so the model errs low)

- Shorter waits for customers who do stay, and better engagement time.
- Coaching value: knowing which salesperson is quick and which one is overloaded.
- More than one camera or showroom: the model is per showroom, per month.

### Costs not counted (so the model errs high)

- The manager's time to review scorecards and run coaching conversations.
- Staff training, change management, camera cabling and placement work.
- Legal review and any notice required by local rules.

### Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Staff acceptance | Salespeople may read this as surveillance and game it, or resent it | Staff are informed before use. Position it as coaching, not ranking. Scores never trigger action on their own (ZRU-1115 decision-support framing). Share each person's own card first |
| Privacy and legal | Cameras on staff and customers are sensitive | No face recognition; faces blurred in every output; on-premise; deleted after 24 h. Have counsel review local personal-data rules before a pilot |
| Camera placement | Distance and detection quality depend on the angle. People cut off at the frame edge may not be detected and so not blurred; a low camera hides distances | Choose an angle showing full bodies and the entrance; keep people large in frame; use the optional manual floor calibration |
| Accuracy on real data | Staff suggestion and distances were never tested on a showroom | Pilot with a manager reviewing roles (one click each); report measured error back honestly |
| Speed | CPU-only: below real time on the clips we timed, about 3.2 fps live in crowded scenes | Offline analysis of recorded shifts is fine; for live, use a GPU or a faster CPU, or fewer people per camera |
| False alerts | A "walk-out" may be a browser who never wanted help | Alerts are prompts to look at a contact sheet, not verdicts |
| The ROI inputs are guesses | A sceptical partner will see this at once | Say so on the slide; hand them the formula; replace every input from their records |
| Licence follow-up | RTMPose-s training data and the bundled FFmpeg (GPL build) need review before commercial distribution | See notes 1 and 2 under the README licence table |

## 5. Questions the jury may ask

- **How do you know who is staff?** We do not know; we suggest. Four visible features (people in a similar outfit,
  distinct contacts, time on screen, who walked up to whom) give a score and plain reasons, and a person confirms or
  flips each role with one click. It has not been evaluated on real showroom data.
- **How accurate is it?** Say what is measured: the MEVA numbers in EVAL.md, the speeds in PROGRESS.md. Say what is not:
  roles and greet times on showroom footage. Do not round up.
- **Why not a GPU?** The showroom already has a PC; a CPU-only laptop keeps the entry cost near zero. Speed on crowded
  scenes is a stated limitation.
- **What about privacy and the law?** No face recognition, faces blurred everywhere, on-premise, 24-hour deletion,
  staff informed, decision support under the 2026 AI law (ZRU-1115). Counsel should review before a pilot.
- **Is the data legal?** Yes: MEVA (CC BY 4.0) for evaluation only, never shipped; no ultralytics or boxmot; no re-ID
  weights trained on Market-1501, DukeMTMC or MSMT17. Open item: the training-data terms behind RTMPose-s and the GPL
  FFmpeg build (README licence table notes).
- **Did you use partner footage?** No. None was available or used.
- **Who wrote it?** Written during the hackathon with AI assistance (Claude Code); see the pre-final declaration in the
  README.
- **What would you do with two more weeks?** Film staged scenes, label them with a person, measure staff-suggestion and
  greet-time error, tune on a dev split only, and put the real ROI inputs in.
