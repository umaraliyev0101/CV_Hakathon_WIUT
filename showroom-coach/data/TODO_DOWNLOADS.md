# Downloads the owner should do by hand

Pexels blocks scripted downloads, so these showroom clips must be fetched in a browser.
They are free for commercial use under the Pexels License. Never commit them (`*.mp4` is gitignored).

| What | URL | Save as |
|---|---|---|
| Car dealer and customer (Antoni Shkraba) | https://www.pexels.com/video/7154222/ | `data/stock/pexels-7154222.mp4` |
| Any other real staff–customer scene (car showroom, retail counter) | search pexels.com for "car dealership", "car salesman", "showroom" | `data/stock/pexels-<id>.mp4` |

After adding clips:
1. Add one row per clip to `data/sources.csv` (`Pexels,<file>,https://www.pexels.com/video/<id>/,Pexels License (free commercial use),<author>,demo + agent labels,no`).
2. Run `showroom-coach analyze data/stock/pexels-<id>.mp4 --out runs/stock/<id> --preset short`.
3. To evaluate a showroom clip (staff-suggestion accuracy, greet-time error), it needs ground-truth labels first: who is
   staff, and when each customer was greeted or walked out. No labels and no scoring command exist yet; see "No showroom
   footage" in `docs/EVAL.md`.

MEVA evaluation clips are fetched by script (see `data/meva/clips.csv`, CC BY 4.0, Kitware/IARPA).
