# Synthetic showroom: visual review of the share pack

Reviewed 2026-10-11, 05:40–06:15 (+05), by opening the images with the image-reading tool. Frame
numbers are **output frames at 10 fps** (t = frame / 10).

**What I looked at for each clip:**
- **Contact sheet** (`data/synthetic/share/<clip>/contact_sheet.png`): 12 evenly spaced frames, plus GT boxes coloured by role, zones and event markers.
- **Zoomed review sheet** (`python -m sim.qa.review_sheets <clip>`): a full-resolution crop around every labelled person in those 12 frames, with the GT box drawn.
- **Two motion sheets** covering all 8 clips:
  - for each person, 6 consecutive frames from the middle of their longest walk;
  - the first 6 frames of their longest in-place turn.

  These are where "sliding" and "pivot shuffle" show up.
- **Spot checks:** I opened full frames where something looked odd (s003 cam2 f352/f358/f364), and checked mask and solo pixel counts where a gate flagged a gap (s004 cam1 f34–f46).

Numeric support used in the table, all from `gt.json` or the gates:
- box heights (min / median px);
- frames whose box touches the image border;
- `visible_frac`;
- mean luma.

## Results: all 12 items pass in all 8 clips

| # | Item | s000 greet_fast cam1 day **(hero)** | s000 greet_fast cam2 glare | s001 greet_slow cam1 evening | s002 walkout cam1 glare | s003 party cam1 day | s003 party cam2 evening | s004 handoff cam1 evening | s004 handoff cam2 glare |
|---|---|---|---|---|---|---|---|---|---|
| 1 | T-pose or frozen | pass: staff walking f102–107 (legs alternate), talk gestures f122–f286 | pass: f102–107 walk, f163–f245 talk arm motion | pass: idle variation f0–f212, staff #2 walk f518–523 | pass: customer walk f273–278, look-around f95–f428 | pass: f96–101 staff walk, party walk f355–368 | pass: f96–101, f82–87 | pass: f96–101, f279–284, f61–66 | pass: f279–284 staff #2 walk, f58–63 customer walk |
| 2 | Floating or sunk feet | pass: soles on floor with contact shadows f0, f122, f449; qa_scene lowest vertex −0.032…+0.004 m | pass: f41, f163, f449 | pass: f71, f283, f567 | pass: f48, f95, f475 (long glare shadows meet the soles) | pass: f100, f200, f399 | pass: f50, f399 | pass: f56, f281, f450 | pass: f0, f56, f506 |
| 3 | Moonwalking or sliding | pass: the stance foot stays put across f102–107 and f361–366; the turn f314–319 steps its feet | pass: f102–107 walk; turn f90–94 steps | pass: walk f518–523 and f275–280; turns f0–4, f86–90 step | pass: walk f273–278; turn f437–441 steps | pass: walk f96–101, f355–368; turns f335–340, f83–87 step | pass: same motion as cam1, seen from cam2 (f96–101, f83–87) | pass: walk f96–101, f279–284; turn f324–328 (staff #1 leaving after the handoff) steps | pass: turn f511–516 and f309–313 step |
| 4 | Walking through cars, walls or people | pass: route around car_1 f82–f122; closest approach between two people 1.25 m (qa_scene) | pass: f41–f122 | pass: f71–f283 customer passes the desk side | pass: f285 walks between car_1 and car_2 without touching | pass: f349 party walks past car_1 | pass: f352–f364 pass beside the desk, 2.4 m from staff (full frames checked) | pass: f338 staff #1 leaves around customer and staff #2 | pass: f281 staff #2 walks in front of the green car |
| 5 | Staff not facing customer during greet | pass: f122–f286 staff faces the customer (yaw error ≤ 20° is tested on every frame) | pass: f163–f245 | pass: f567, f637 staff #2 faces the customer, arm gesture toward them | n/a by design (no greet); staff faces the desk | pass: f150–f299 staff faces the party | pass: f150–f299 | pass: f169–f281 staff #1, then f394–f450 staff #2 (handoff) | pass: f113–f281, f338–f450 |
| 6 | Missing or pink textures | pass: no magenta pixels (qa_render 0.0000%); skin, suit and car paint textured | pass | pass | pass | pass | pass | pass | pass |
| 7 | Too dark or blown out | pass: mean luma 129.9–132.5, 0.000% clipped | pass: 104.5–109.9, 0.064% clipped | pass: evening 55.4–58.5, people readable f354–f779 | pass: glare 144.2–146.1, 0.105% clipped (limit 20%); bright floor patches, people readable | pass: day 111.9–116.8, 0.000% | pass: evening 41.4–42.7, dim but readable (f399, f549) | pass: evening 58.1–60.8 | pass: glare 118.2–128.7, 0.036% clipped |
| 8 | People < 40 px or cut off most of the clip | pass: median box height 304 / 248 px; border-touching in 0 / 7 of 450 frames | pass: median 109 / 128 px; min 56 px; 4 border frames | pass: medians 245 / 218 / 176 px; customer min 67 px | pass: medians 323 / 223 px; customer 42 px only while entering | pass: medians 354 / 348 / 304 px | pass: medians 119 / 162 / 169 px; under 40 px only on entry frames | pass: medians 367 / 193 / 303 px; staff #1 on the left border in 11 of 620 frames | pass: medians 172 / 341 / 227 px |
| 9 | Misaligned boxes | pass: boxes hug the silhouettes in every crop (f0–f449) | pass: also tight on the customer partly occluded behind the car, f367 (v=0.50) | pass | pass | pass: the two overlapping party boxes are each tight, f100–f299 | pass | pass | pass |
| 10 | Wrong role colours | pass: staff red, customer blue in all 12 sheet frames | pass | pass: both staff red (#1 desk, #2), customer blue | pass | pass: staff red, both party members blue | pass | pass: staff #1 and #2 red, customer blue | pass |
| 11 | Labelled reflections | pass: labels come only from person-mesh instance masks, so reflections in paint or floor are never labelled; no box sits on a reflection in any sheet frame | pass: glare floor reflections unlabelled | pass | pass | pass | pass | pass | pass |
| 12 | Wrong scale | pass: staff 1.74 m and customer 1.63 m read as adults next to a 4.3 m car | pass | pass | pass | pass | pass | pass | pass |

Notes from the review (no item failed):
- **People near the image edges lean.** Staff #2 in s004 cam2 f0–f225 and f563–f619, and the s001 customer at f142, appear tilted. This is wide-angle perspective near the frame border. Wall edges and plant pots at the same image position lean the same way, and the pose itself is an upright idle. It is a camera effect, not an animation bug.
- **The hard-negative customers' hair** can read as a smooth cap from behind in glare at small size (s000 cam2 f361–366). Hair cards are alpha-blended; the item 6 check (no pink or missing texture) still passes.
- **Some boxes are a few pixels taller at the head than they look.** Workbench masks count hair and eyelash cards fully, regardless of alpha. This is documented in `gt.json` notes.
- **The gate flag at s004 cam1 f37–f43 was not a label dropout.** The customer walks out past the right image border: in-image silhouette 5–246 px, under the 400 px drop rule. Mask and solo pixel counts were checked frame by frame.
