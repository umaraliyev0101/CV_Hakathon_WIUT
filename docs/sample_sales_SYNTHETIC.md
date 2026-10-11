# sample_sales_SYNTHETIC.csv

`docs/sample_sales_SYNTHETIC.csv` is **synthetic**: the numbers are invented to demonstrate the
sales join, not real sales of any showroom. Never quote them as results.

- Header is exactly `car_model,period,sold` (a CSV cannot carry a comment row and stay parseable, so
  the "synthetic" label lives in the file name and in this note).
- Car names match the zones in `labels/zones/wm-bangalore-07.json` ("Car: white hatchback",
  "Car: white SUV (left, cropped)"). Matching is case-insensitive after dropping a leading `Car:`.
- `sold` is summed over periods per car; the report shows `sold` and `attention_to_sales`
  (stops per sale; blank when sold is 0 or the car has no rows).

Try it: `pulse analyze data/stock/wm-bangalore-07.mp4 --out runs/demo --zones labels/zones/wm-bangalore-07.json --sales docs/sample_sales_SYNTHETIC.csv`
(the web app takes the same file via `POST /api/jobs/{id}/sales`).
