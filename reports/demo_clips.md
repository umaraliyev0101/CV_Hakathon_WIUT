# Demo clips: which synthetic clips to upload during the demo

Date: 2026-10-11. Stack: `deploy/` docker compose at http://127.0.0.1:8090 (nginx, basic auth), Coach at `/coach/`, Pulse
at `/pulse/`. Both apps use the default detectors (Coach `yolox_nano@2+rtmpose_s+dinov2_small`, Pulse `yolox_s+rtmpose_s`),
the `short` preset and no fine-tuned model.

## Method

- Each of the 10 clips in `deploy/demo/` was uploaded over HTTP to both apps by a scripted driver (the same flow as the
  web page, without a browser). The drivers are in the session scratchpad and are not committed.
- **Staff came from GT, not from the role suggestions.** For each detected person, its tracked boxes (copied read-only
  from the job's perception cache) were matched by IoU against the GT boxes (`gt.json` frames). The person then took the
  role of the GT person it matched most often. Staff were posted through `POST /roles` (Coach) or
  `PUT /roles` overrides (Pulse). Pulse kept its auto-suggested car zones.
- GT: `data/synthetic/share/<clip>/gt.json`, and `data/synthetic/clips/<clip>/gt.json` for s005 and s006. s005_crowded
  does have a GT file in `clips/`. The demo mp4s of s005 and s006 were re-encoded, but they have the same frame count and
  resolution, so the GT boxes still line up (mapping purity was 0.9 to 1.0).
- Coach events were matched to `events_scripted` by type, the mapped GT ids of customer and staff, and |dt| <= 6 s.
  Wait alerts were compared with the `short_clip` rule events. Coach's short preset raises a wait alert after 10 s,
  while the GT rule uses about 30 s. So a Coach wait alert on the right un-greeted customer counts as correct even when
  it comes earlier.
- Pulse has no event list. Its store and per-car numbers were compared with GT counts (customers, greeted, unserved,
  parties). GT car dwell is the time a GT customer's floor position spends inside the GT car floor polygons. The auto
  zones are not the GT zones, so dwell can only be compared roughly.
- The visual check used 8-frame montages of each annotated video, the Coach contact sheets and timeline, and the Pulse
  keyframe with auto zones drawn next to the GT zones.
- Coach was run three times, including across two container restarts, and gave the same events every time. Pulse was
  re-run after it moved to GPU (see the timing note).

### Timing note

During this session the coordinator rebuilt the stack, and Pulse moved from CPU to GPU (`CUDAExecutionProvider`).

- **On CPU, Pulse took 260 to 350 s just to perceive a 45 to 55 s clip** (about 6x slower than real time). That would be
  unusable live.
- On GPU it takes 3 to 17 s. All Pulse times below are the GPU times.
- Pulse runs one job at a time (`ThreadPoolExecutor(max_workers=1)`). If several people upload at once, each job waits
  behind the others: one run waited 516 s for its render while someone else's job ran. **Have only one person upload
  at a time during the demo.**

## Per-clip results

Abbreviations:

- "Coach ready": seconds from upload until it asks who is staff (`needs_confirm`). "Coach video": seconds from posting
  roles until the annotated video is ready.
- "Pulse ready": seconds from upload until the zones step (`ready_for_zones`). "Pulse video": seconds from scoring until
  the annotated video is ready.
- Coach events are scored against GT `events_scripted`. Pulse numbers are shown as detected vs GT.

### s000_greet_fast_cam1_day (45 s; 1 staff, 1 customer; GT greet and engagement at 11.9 s)

| | Coach | Pulse |
|---|---|---|
| time | ready 2.0 s, video 1.6 s | ready 4.0 s, video 2.2 s |
| correct | greet 11.6 s, engagement 11.2 s (to 30.2 s) | 1 assisted customer, time to assist 2.0 s |
| missed | none | none |
| false | none | visitors 2 vs 1: the customer splits into two ids at 9 to 11 s (the tracks overlap, so they cannot be merged) |
| dwell | n/a | Car 2 (red SUV) 1.6 s vs GT 28.9 s. The customer stands beside the car, outside the zone rectangle |
| visual | Clean: 2 boxes, staff in blue, customer in orange with a green tick, a link line during the talk, faces blurred, readable labels. 2 contact sheets | Boxes and faces fine. The zone labels ("Car 2 now 0, total 0") are tiny at 960 px |

### s000_greet_fast_cam2_glare (45 s; same scene from camera 2)

| | Coach | Pulse |
|---|---|---|
| time | ready 2.5 s, video 1.6 s | ready 4.0 s, video 2.3 s |
| correct | greet 11.2 s, engagement 8.2 s (3.7 s early) | 1 assisted, time to assist 5.4 s |
| missed / false | none | visitors 2 vs 1 (the customer is re-identified at 38.6 s) |
| dwell | n/a | Only 1 auto zone, placed on the floor in front of the red SUV. The people stand behind the car, so dwell is 0 s everywhere (GT 28.9 s) |
| visual | Fine, but the people are small and far away | Weak: no dwell, a single zone |

### s001_greet_slow_cam1_evening_dim (78 s; 2 staff, 1 customer; GT greet at 53.1 s; rule wait alert at 32.6 s)

| | Coach | Pulse |
|---|---|---|
| time | ready 10.1 s, video 3.1 s | ready 6.0 s, video 2.5 s |
| correct | wait alert for the customer at 16.8 s | visitors 1, staff 2, 1 assisted, **time to assist 45 s (GT about 50 s)**, Car 1 dwell 49.0 s (GT car_2 47.9 s), insight "Long unattended waits: 45 s" |
| missed | **greet and engagement at 53.1 s are missed**: the staff member walks up but is far from the camera, in dim light | none |
| false | a phantom customer C4 (a box at the top edge of the frame, not a GT person) gets a "long wait" at 37.8 s and keeps a visible wait counter to the end | "Dead zone at Car 2" (the zone is on the floor in the foreground; nobody goes there) |
| visual | Dark. The phantom C4 is visible and the customer is never ticked green | Good. A believable story |

### s002_walkout_cam1_glare (52 s; 1 staff, 1 customer who leaves unserved at 49.9 s)

| | Coach | Pulse |
|---|---|---|
| time | ready 2.5 s, video 2.1 s | ready 4.0 s, video 2.4 s |
| correct | **walkout at 49.4 s** (GT 49.9 s), wait alert at 13.6 s with the banner "Long wait: C2, free: Staff1", contact sheet "C2 left without a greet, free staff: Staff1" | 1 unserved episode, 0 assisted, 1 entry and 1 exit, insight "Unserved hotspot at Car 3: 1 of 1 interested customers got no contact", red box on the customer, Car 3 dwell 16.2 s (GT car_4 14.8 s) |
| missed / false | none | visitors 2 vs 1: a **static phantom person** (a small box at the top of the frame, near the plant or wall) counts as a visitor for the whole clip |
| visual | Bright glare scene with very clear storytelling | Good, but the phantom box shows in every frame |

### s003_party_cam1_day (55 s; 1 staff, a party of 2 customers; GT party at 2.6 s, greets at 11.4 s)

| | Coach | Pulse |
|---|---|---|
| time | ready 5.5 s, video 2.6 s | ready 4.0 s, video 2.5 s |
| correct | party (7.8 s, 5 s late), 2 greets (11.0 s), 2 engagements (10.6 s and 11.0 s): all 5 GT events | groups 1 (correct), Car 2 and Car 1 visited in the right order |
| missed | none | 1 of 2 customers counted as assisted (GT 2) |
| false | customer 3 changes id at about 37 s (C3 becomes C4), and the "new" person gets a false wait alert at 48 s plus an extra enter | visitors 3 vs 2. **"Unserved hotspot at Car 1: 2 of 2 interested customers got no contact" is wrong**: both were served at Car 2 earlier. Pulse counts unserved per car. Warning "Only 98% of the clip analysed (camera moving)" on a static camera |
| visual | Good for the first 35 s. After that the id switch shows a "wait" label on a customer who was already served | Fine. Red boxes at the end support the wrong insight |

### s003_party_cam2_evening_dim (55 s; same scene from camera 2)

| | Coach | Pulse |
|---|---|---|
| time | ready 3.5 s, video 1.6 s | ready 4.0 s, video 2.5 s |
| correct | party (6.4 s), 2 greets (10.6 s), 2 engagements (10.4 s): all 5 GT events | groups 1, 2 assisted |
| missed / false | false wait alert at 50 s (an identity fragment) | visitors 4 vs 2 (7 ids, 2 fragments hidden), the same wrong "Unserved hotspot" insight, and one customer *suggested* as staff |
| visual | Dark and small people. Works, but looks worse than cam1 | Weak |

### s004_handoff_cam1_evening_dim (62 s; 2 staff, 1 customer; GT greet at 11.3 s, handoff from staff 1 to staff 2 at 30.9 s)

| | Coach | Pulse |
|---|---|---|
| time | ready 3.0 s, video 2.6 s | ready 32 s (queued behind another job; perception itself took 4 s), video 6.3 s |
| correct | **all 4 GT events**: greet 10.6 s, engagement 9.8 s, handoff 29.0 s, engagement with staff 2 at 29.0 s. Every id 100% pure. The timeline image shows the handoff clearly | visitors 1, staff 2, 1 assisted, time to assist 6.4 s |
| missed / false | none | **"Only 68% of the clip analysed (camera moving or cut)"** on a static camera (probably the dark, noisy video), and every car shows 0 s dwell (GT car_1 55.8 s), even though the customer stands inside the drawn Car 2 zone |
| visual | Dark but clean: link lines, "S1 Staff1", "S2 Staff2" and "C3" with a tick. No glitches | Car zone counters stay at 0 while the customer stands in the zone, which looks broken |

### s004_handoff_cam2_glare (62 s; same scene from camera 2)

| | Coach | Pulse |
|---|---|---|
| time | ready 5.0 s, video 2.6 s | ready 6.0 s, video 2.4 s |
| correct | all 4 GT events (greet 10.4 s, handoff 28.6 s) | 1 visitor, 2 staff, 1 assisted |
| missed / false | none in events, but staff 1 splits into two ids (S1 and S4 "Staff3"), so a judge would have to mark 3 staff | median visit and time to assist empty, 0 dwell, 1 zone, warning "Only 94% analysed" |
| visual | The talk happens small, at the back of the room. Cam1 shows it much better | Weak |

### s005_crowded_cam1_glare (101 s; 2 staff, 5 customers including a party of 2; 3 greets; 1 walkout at 98.6 s)

| | Coach | Pulse |
|---|---|---|
| time | ready 11.6 s, video 3.5 s | ready 12.0 s, video 9.4 s |
| correct | 3 of 3 greets, 3 of 3 engagements, walkout at 97.2 s (GT 98.6 s), 1 wait alert | 2 assisted |
| missed | party at 8.6 s | party (groups 0), 1 assisted customer |
| false | **customers 11 vs 5, walkouts 6 vs 1**: customer 6 fragments into several ids, and each one "walks out" (at 12.6, 13.6, 58.4, 80, 84 s). 3 extra wait alerts. A real customer was *suggested* as staff | visitors 8 vs 5, and staff fragments (the staff count shows 4). Car 1 dwell 43.6 s (GT total car dwell is over 400 s across 4 cars) |
| visual | Busy. Labels overlap, and a false "Walk-out: C9" banner appears while C9 is still on screen | Busy, numbers not credible |

### s006_dealership_long_cam1_day (252 s; 2 staff, 2 customers; GT greet at 12.2 s with engagement until 162 s; customer 4 never served and walks out at 249 s)

| | Coach | Pulse |
|---|---|---|
| time | ready 16.6 s, video 9.1 s | ready 22.1 s, video 9.1 s |
| correct | greet 11.6 s, engagement 11.6 s, walkout at 245.8 s | **visitors 2, staff 2, entries 2, exits 1, assisted 1, time to assist 5.4 s: all exact.** Unserved hotspot at Car 2 for customer 4 (correct) |
| missed | none | none |
| false | **customers 17 vs 2, walkouts 15 vs 1**: customer 4 stands near the right edge, at the white Porsche, and keeps dropping in and out. Every return counts as a new visit and a "Walk-out: C4" banner flashes about 14 times. 4 wait alerts | "Unserved hotspot at Car 1" for customer 3 at 172 s, after their engagement ended (debatable). Car 4 (the red 911 where the long talk happens) shows 0 s dwell, because the people stand just below the zone rectangle (GT about 150 s) |
| visual | First 2 min look great. After that the repeated walk-out banners look buggy | Clean, with plausible store numbers. The tiny zone labels and the 0 s dwell at the main car are the weak spots |

## Cross-cutting problems found

1. **Role suggestions are mostly wrong or "unsure", in both apps.** In Coach, 8 of 10 clips suggest no correct staff at
   all, and s004_cam1 and s005 *suggest a customer as staff*. Pulse suggests "customer" or "unsure" for almost everyone.
   **In the demo, the presenter must click the staff manually** (staff wear red suits in every clip, which makes this
   easy). Coach's default (unsure counts as customer) means no greets appear until someone does. Plan this into the
   demo script.
2. **Pulse car zones are tight car-box rectangles, so dwell is badly under-counted.** People stand *beside* a car, and
   the foot point used for zones falls outside the rectangle. Measured dwell is 0 to 2 s where GT has 30 to 150 s
   (s000, s004, s006 Car 4). Cam2 clips get only 1 zone. Draw or enlarge the zones by hand in the zone editor before
   scoring if per-car dwell is part of the pitch.
3. **Pulse "camera moving" filter fires on static dark or glare clips.** s004_cam1 keeps 68% and s004_cam2 keeps 94%,
   and all zone dwell drops to 0.
4. **Identity fragmentation on long or crowded clips** inflates Coach visit and walk-out counts (s005: 11 customers and
   6 walk-outs instead of 5 and 1; s006: 17 and 15 instead of 2 and 1) and Pulse visitor counts (s005: 8 vs 5).
5. **Phantom static detections** at the top edge of the frame: Coach s001 (C4 with a long wait), Pulse s002 (a visitor
   for the whole clip).
6. **Pulse "unserved hotspot" is per car.** Customers already served at another car are reported as "got no contact"
   (s003 both cameras, s006). On the party clips this reads as a false claim.
7. **Pulse overlay text is unreadably small** in the 960 px annotated video (zone counters, ids). Coach overlays are
   readable.
8. Faces were blurred on every person in every frame checked, in both apps (keyframes, thumbnails, contact sheets,
   videos). No unblurred faces were seen.

## Recommended demo set, in this order

1. **s004_handoff_cam1_evening_dim (Coach)**: all 4 GT events with every id correct, and the handoff is the most
   impressive event. Timeline and contact sheets are clean, about 6 s end to end. Mark S1 and S2 as staff. Do not use
   it for Pulse (68% analysed, 0 dwell).
2. **s002_walkout_cam1_glare (both apps)**: Coach shows the wait alert, then "C2 left without a greet, free staff:
   Staff1". Pulse shows 1 unserved customer, the "Unserved hotspot at Car 3" insight and a red box. One drawback: Pulse
   counts a phantom visitor (2 instead of 1).
3. **s001_greet_slow_cam1_evening_dim (Pulse)**: Pulse's best match to GT: 1 visitor, time to assist 45 s (GT about
   50 s), 49 s dwell at the right car, and the "Long unattended waits" insight. **Do not show it in Coach**: Coach misses
   the greet and shows a phantom waiting customer.
4. **s000_greet_fast_cam1_day (Coach)**: the brightest and clearest clip, a simple greet in 11.6 s at about 4 s of
   processing. A good first upload to explain the flow. In Pulse it is fine but shows 2 visitors and almost no dwell.
5. Optional: **s006_dealership_long_cam1_day (Pulse only)**: the most realistic 4-minute scene, and Pulse store numbers
   are exactly right (2 visitors, 1 assisted, 1 unserved, 1 exit) in about 30 s. **Avoid it in Coach** (15 false
   walk-outs).

A short "best of both" set, if time allows only 3 uploads: s004_handoff_cam1 in Coach, s002_walkout_cam1 in both, and
s001_greet_slow_cam1 in Pulse.

## Clips to avoid

- **s005_crowded_cam1_glare (both apps)**: inflated customer and walk-out counts, a false walk-out banner, the party is
  missed, and the screen is cluttered.
- **s006_dealership_long_cam1_day in Coach**: the walk-out banner repeats about 14 times for one customer.
- **s001_greet_slow in Coach**: the greet is missed and a phantom customer waits.
- **s004_handoff_cam1 in Pulse**: "68% analysed" and zero dwell next to a customer standing in the zone.
- **Cam2 variants (s000_cam2_glare, s003_cam2_evening_dim, s004_cam2_glare)**: same events as cam1 but smaller, darker
  people. Coach still gets the events, but staff split into extra ids (s004_cam2). Pulse gets 1 zone and no dwell.
  Use cam1 instead.
- **s003_party (either camera) in Pulse**: the wrong "2 of 2 interested customers got no contact" insight. In Coach it
  is acceptable (all 5 GT events, party included), but a customer id switch at about 37 s produces a false wait alert.
  If you show it, stop the video before 0:37.
