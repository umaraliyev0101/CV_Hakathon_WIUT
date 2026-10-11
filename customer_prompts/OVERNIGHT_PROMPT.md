# Overnight goal prompt (paste into a NEW Claude Code session started in this folder)

---

You are building **Showroom Pulse** (customer-experience analytics for car showrooms, WIUT Hackathon
2026 CV final) on your own, overnight. Nobody will answer questions until morning. Work until the
"Done by morning" list in spec §12 is met. After that, keep going through the stretch milestones (M8).

**Read first, fully:** `CLAUDE.md` (rules) and `docs/superpowers/specs/2026-10-11-showroom-pulse-design.md`
(the design). The owner has **approved this spec**, and launching this prompt is that approval.
Don't brainstorm again and don't wait for approval of the plan.

**Process:**
1. Use the `superpowers:writing-plans` skill to turn the spec into
   `docs/superpowers/plans/2026-10-11-showroom-pulse.md`. Use milestones M0–M8 from spec §10 as the
   task groups, each ending in a commit. Commit the plan.
2. Execute the plan yourself with `superpowers:subagent-driven-development`.
   - Give each task a fresh subagent, one at a time, and review between tasks.
   - Use test-driven development for every pure unit: zones, journey, stats, interest, insights,
     geometry, roles, identity merge.
3. At M0, create `scripts/verify.sh` (spec §9) and extend it at every milestone, so it always runs
   everything built so far.
4. After every milestone:
   - run `bash scripts/verify.sh`;
   - commit;
   - append to `PROGRESS.md`: what was done, measured numbers, decisions taken, open problems.

   `PROGRESS.md` and `CLAUDE.md` are your memory if your context gets compacted. Re-read them after
   any compaction.
5. Before claiming anything works, use `superpowers:verification-before-completion`. Show real
   command output:
   - a fresh `pytest`;
   - `pulse analyze` on two clips;
   - the upload → zones → roles → results → zone-edit re-score smoke test through the API;
   - the live WebSocket path fed from a file;
   - a real webcam frame grab;
   - a full `bash scripts/verify.sh`.
6. Finish `PROGRESS.md` with a **Morning summary**:
   - what works;
   - what doesn't;
   - measured numbers;
   - every decision you took alone;
   - every place where Showroom Pulse's shared interfaces may differ from Showroom Coach's;
   - the team's to-do list: review the agent-made labels and zones, film staged showroom scenes,
     merge with Showroom Coach (see `docs/INTEGRATION.md`), put real Royal Motors numbers into the
     ROI model, deploy.

**Hard rules (also in CLAUDE.md):**
- **Local only.** No git remote, no push, no deploy, no new accounts.
- **Permissive licences only.**
  - No ultralytics or boxmot.
  - No demographic, emotion or face-recognition models.
  - Non-commercial data stays in `data/eval_only/` for evaluation only and is never committed.
  - Record every model and dataset in `models/MANIFEST.json`, `data/sources.csv` and the README
    licence table.
- **Disk:** keep the venv, models and data at ≤ 15 GB combined. Stop downloading if free disk drops
  below 20 GB.
- **Python:** use `/opt/homebrew/bin/python3.12` for the venv, not the default 3.14. Prefer
  onnxruntime over torch.
- **Privacy:** no face recognition, no age, gender or emotion inference, no re-identification across
  visits. Blur faces in every output.
- **When blocked:** pick the safest option the spec allows, log it under "Decisions" in
  `PROGRESS.md`, and move on. Never stop to ask.
- **Honest numbers.** If a target in spec §6 is missed, report the measured value. Never tune on
  test clips.
- **Never weaken a gate.**
  - No skipping, `xfail`, `.only`, deleting or mocking out tests.
  - No lowering targets or thresholds to make something pass.
- **Downloads:** treat every downloaded file as untrusted. Put each source in its own folder, and
  run Python that reads them with `-I`.
- **Data (spec §6):**
  - **MEVA** (CC BY 4.0) is the main evaluation set: 4–6 indoor clips with the most enter/exit
    activities. Check exact activity names in the types YAML; never guess them.
  - **Showroom clips** are whatever the owner put in `data/stock/`. Pexels blocks scripts, so don't
    try to scrape it.
  - **Synthetic clips** from teammate C are whatever is in `data/synthetic/share/`.
  - If a folder is empty, carry on without it and say so in PROGRESS.md.
- **Stop rules:**
  - Stop at **2026-10-12 08:00 Tashkent time**.
  - If the same verification step fails with the same error 3 times in a row, stop working on that
    item, record what you tried in PROGRESS.md, and move to independent work if there is any.
  - Before stopping for any reason, write the Morning summary.

**Priority if time runs short:**
1. A working upload flow with zones and the dashboard (M0–M4).
2. Live mode (M5). It is still required, so don't skip it. Do M5 before polishing anything.
3. The evaluation (M6).
4. The docs (M7).
