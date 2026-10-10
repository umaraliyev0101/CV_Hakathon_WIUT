# Project brain - Crash Twin

Planning and design material for **Crash Twin**: phone photos/video of a minor collision turned into a measured 3D accident twin and a traffic-rule verdict. WIUT Hackathon 2026, Computer Vision track (final). Team UsedByAI.

| File | What it is |
|---|---|
| `crash_twin.md` / `crash_twin.pdf` | Full project document: problem, solution, modules M1-M8, models and licences, datasets for training and testing, compute and cost, pitch plan, sources |
| `diagrams/crash_twin_bpmn.drawio` | 4-page draw.io file: business flow, technical pipeline, data/training/testing, architecture and tech stack. Open at app.diagrams.net |
| `diagrams/crash_twin_diagrams.pdf` | Notes explaining each diagram page, tech stack table, open questions |
| `diagrams/crash_twin_diagrams_preview.pdf` | Quick PDF preview of the 4 pages (approximate render; the .drawio is the source of truth) |
| `diagrams/src/` | Python that generates the diagrams (bpmn-architect skill) |

## Regenerate a diagram page

```
cd diagrams/src
python3 p1_business.py            # writes p1.drawio
python3 bpmn_validate.py p1_business
```

Same for `p2_pipeline`, `p3_data`, `p4_arch`.

Compute, timing and cost figures in these documents are estimates until measured. Re-check every licence and rule clause number before submission.
