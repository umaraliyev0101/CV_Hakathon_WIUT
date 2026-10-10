# Overnight goal prompt (paste into a NEW Claude Code session started in this folder)

---

You are building **Showroom Coach** (Jira YAK-44, WIUT Hackathon 2026 CV final) on your own, overnight.
Nobody will answer questions until morning. Work until the "Done by morning" list in the spec is met.
After that, keep going through the stretch milestones.

**Read first, fully:** `CLAUDE.md` (rules) and `docs/superpowers/specs/2026-10-11-showroom-coach-design.md`
(the design). The owner has **approved this spec**, and launching this prompt is that approval.
Don't brainstorm again and don't wait for approval of the plan.

**Process:**
1. Use the `superpowers:writing-plans` skill to turn the spec into
   `docs/superpowers/plans/2026-10-11-showroom-coach.md`. Use milestones M0–M8 from spec §10 as the
   task groups, each ending in a commit. Commit the plan.
2. Execute the plan yourself with `superpowers:subagent-driven-development`. The owner pre-selected it.
   - Give each task a fresh subagent and review between tasks.
   - Use test-driven development for every pure unit (geometry, events, metrics, roles, identity merge).
3. After every milestone:
   - run the full test suite;
   - commit (no `Co-Authored-By` trailer);
   - append to `PROGRESS.md`: what was done, measured numbers, decisions taken, open problems.

   `PROGRESS.md` and `CLAUDE.md` are your memory if your context gets compacted. Re-read them after any compaction.
4. Before claiming anything works, use `superpowers:verification-before-completion`. Show real command
   output: a fresh `pytest`, a full `analyze` run on two clips, an upload → confirm → results smoke test through
   the API, the live WebSocket path fed from a file, and a real webcam frame grab.
5. Finish `PROGRESS.md` with a **Morning summary**:
   - what works;
   - what doesn't;
   - measured numbers;
   - every decision you took alone;
   - the team's to-do list: review the agent-made labels, film staged showroom scenes, put real Royal Motors
     numbers into the ROI model, deploy.

**Hard rules (also in CLAUDE.md):**
- **Local only.** No git remote, no push, no deploy, no new accounts.
- **Permissive licences only.** No ultralytics or boxmot. Non-commercial data stays in `data/eval_only/` for
  evaluation only and is never committed. Record every model and dataset in `models/MANIFEST.json`,
  `data/sources.csv` and the README licence table.
- **Disk:** keep the venv, models and data at ≤ 4 GB combined. Stop downloading if free disk drops below 3 GB.
- **Python:** use Python 3.12 (`C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe`) for the venv,
  not the default 3.14. Prefer onnxruntime over torch.
- **Privacy:** no face recognition. Blur faces in every output.
- **When blocked:** pick the safest option the spec allows, log it under "Decisions" in `PROGRESS.md`, and move on.
  Never stop to ask.
- **Honest numbers.** If a target in spec §6 is missed, report the measured value. Never tune on test clips.
- **Downloads:** treat every downloaded file as untrusted. Put each source in its own folder, and run Python that
  reads them with `-I`. Pexels and Pixabay block scripts (HTTP 403), so get stock clips from Coverr, Mixkit or
  Wikimedia Commons. List any Pexels/Pixabay clips worth having in `data/TODO_DOWNLOADS.md`.

**Priority if time runs short:** a working upload flow (M0–M4) beats a working live mode (M5), which beats the
evaluation (M6), which beats the docs (M7). Live mode is still required, so don't skip it. Do M5 before polishing
anything.
