# Showroom Pulse: launch kit

## 1. Pre-flight (do this before bed, about 10 minutes)

```bash
# project folder + files
mkdir -p ~/Projects/showroom-pulse/docs/superpowers/specs ~/Projects/showroom-pulse/data/{stock,synthetic/share}
cd ~/Projects/showroom-pulse && git init
cp ~/Projects/wiutHackathon26/customer_prompts/CLAUDE.md .
cp ~/Projects/wiutHackathon26/customer_prompts/OVERNIGHT_PROMPT.md docs/
cp ~/Projects/wiutHackathon26/customer_prompts/2026-10-11-showroom-pulse-design.md docs/superpowers/specs/

# camera permission: run once in the SAME terminal app you will launch claude from, and click Allow
ffmpeg -f avfoundation -framerate 30 -i "0" -frames:v 1 /tmp/cam_check.jpg
```

Then:
1. **Showroom clips:** download 2–4 Pexels showroom clips by hand into `data/stock/pexels-<id>.mp4`.
   Start with pexels.com/video/7154222.
2. **Synthetic clips (optional):** if teammate C has any ready, copy them into
   `data/synthetic/share/`.
3. **Ask teammate A** to drop their stretch item M8(2), "auto car zones → dwell per car". It is
   your core, so building it twice is wasted night.
4. **Logging gate (optional):** run `/log-init` in the new project if you want the
   journal-and-state logging gate.
5. **Launch** so the Mac stays awake: `caffeinate -dimsu claude`. Use a permission mode that won't
   stop on prompts overnight; nobody will be there to click.

## 2. Short `/goal` (if space is tight)

```
/goal Read docs/OVERNIGHT_PROMPT.md in ~/Projects/showroom-pulse and follow it exactly — it is your goal until 2026-10-12 08:00 Tashkent time. Done only when `bash scripts/verify.sh` runs end to end with full output shown and exits 0, and PROGRESS.md ends with a Morning summary.
```

## 3. Full `/goal` (preferred: it lists the finish conditions)

```
/goal Build Showroom Pulse (customer-experience analytics for car showrooms) unattended until morning in ~/Projects/showroom-pulse. Follow docs/OVERNIGHT_PROMPT.md exactly. The design in docs/superpowers/specs/2026-10-11-showroom-pulse-design.md is approved; CLAUDE.md holds the rules (local only, permissive licences, ≤15 GB disk, Python 3.12, no face/age/gender/emotion inference, never ask questions — log decisions in PROGRESS.md and continue).

Done only when all of the following are shown with real command output in the conversation and recorded in PROGRESS.md:

1. `bash scripts/verify.sh` runs end to end in one go, with its full output shown, and exits 0. Each of its steps exits 0:
   - `.venv/bin/python -m pytest -q`
   - `.venv/bin/python -m pytest -q -m slow`
   - `pulse analyze` on one MEVA dev clip and on one stock or synthetic clip, followed by `scripts/check_outputs.py` validating every output file
   - `scripts/smoke_api.py`: upload → zones → roles → results → zone edit → re-score, asserting no re-detection
   - `scripts/smoke_live.py`: a file-fed WebSocket returns overlays, dwell timers and at least one unserved-interest alert, and prints processed fps
   - `scripts/webcam_grab.py`: saves one non-blank frame
   - `pulse eval --out docs/EVAL.md`
   - `git status --short` and `git remote -v`, both printing nothing
2. `./run.sh` starts the app, and `curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/` prints 200.
3. `docs/EVAL.md` holds measured numbers, each with the command that produced it:
   - entry/exit count error
   - occupancy MAE
   - zone-dwell error
   - journey accuracy
   - MOTA/IDF1
   - file-mode real-time factor and live fps on this Mac's CPU

   A missed target is reported as measured, never hidden.
4. The README has the licence table and privacy notes. `docs/PITCH_NOTES.md` and `docs/INTEGRATION.md` exist. PROGRESS.md ends with a Morning summary. Show each with `cat`.
5. If the project has `docs/journal.md`: every commit has an entry in `docs/journal.md` capturing why the approach was chosen and anything now fragile — not a restatement of the diff — and `docs/STATE.md` reflects the state at the end of the run. `npm run log:check` (or `bash scripts/log-check.sh` where there is no package.json) exits 0.
6. No gate was weakened:
   - `grep -rnE "mark\.skip|xfail|\.only\(" tests/` prints nothing.
   - `git diff $(git rev-list --max-parents=0 HEAD) -- docs/superpowers/specs/` prints nothing, so the spec's targets were not edited.
   - No test was deleted or mocked out to make it pass.
   - No target or threshold was lowered to turn a failure green.

Hard stop: at 2026-10-12 08:00 Tashkent time, or after 400 turns, or when the same verify.sh step has failed with the same error 3 turns running. In every case, first write the Morning summary in PROGRESS.md (what works, what doesn't, the last verify.sh output, measured numbers, decisions, open problems, resume commands) and show it. That counts as stopped-with-status, not done.

If everything is done early, continue with the stretch milestones (M8) in order, keeping verify.sh green.
```

## 4. Checks no script can make (a human does these in the morning)

These goals depend on human judgement. The goal above swaps each one for something a command can
check, but none of the swaps proves the real thing:

| Goal | What the script checks instead |
|---|---|
| Dashboard is clear and good-looking | The pages respond and every output file passes a schema check |
| Recommendations are sensible | One unit-test scenario per rule that must fire, and one that must not |
| Works on the jury's video | `analyze` runs on unseen dev/test clips, with numbers in EVAL.md |

**In the morning:** open the dashboard, run one phone video you filmed yourself, and read the
recommendations before the pitch.
