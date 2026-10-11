# Deploy: Showroom Suite on this machine, public through ngrok

One Docker Compose stack:

| service | what | reachable at |
|---|---|---|
| `web` | nginx: suite page, reverse proxy, demo videos, optional password | `http://127.0.0.1:8090/` (local only) |
| `coach` | Showroom Coach (salespeople), `showroom-coach/Dockerfile` | `/coach/` |
| `pulse` | Showroom Pulse (customers), `showroom-pulse/Dockerfile` | `/pulse/` |
| `ngrok` | public HTTPS tunnel to `web` (profile `public`) | printed by `up.sh --public` |

Everything is on one origin, so one tunnel is enough. That origin is HTTPS, so browsers allow the webcam on both live
pages, and there are no mixed-content or cross-origin problems between the suite page and the two panels.
Both apps also still run on their own at `/` (their frontends now resolve every URL against the page, not the host root).

## Run

```bash
cp deploy/.env.example deploy/.env      # optional; set DEMO_USER / DEMO_PASS before going public
deploy/up.sh                            # build + start locally: http://127.0.0.1:8090/
deploy/up.sh --public                   # same, plus ngrok; prints the public https URL
```

- **Stop:** `docker compose -f deploy/docker-compose.yml --profile public down`.
  Add `-v` to also delete the uploaded jobs, caches and Pulse's downloaded models.
- **Logs:** `docker compose -f deploy/docker-compose.yml logs -f coach pulse web ngrok`
- **ngrok token:** `NGROK_AUTHTOKEN` in `deploy/.env`, else `NGROK_TOKEN` in the repo-root `.env` (gitignored), else `~/.config/ngrok/ngrok.yml`. If you set `NGROK_DOMAIN`
  (your static ngrok domain), the URL stays the same across restarts.

**Neither app has accounts.** Once public, anyone with the URL can upload videos (up to 1 GB each) and use CPU. Set
`DEMO_USER` and `DEMO_PASS` in `deploy/.env`. nginx then asks for that password on every page. `up.sh` warns
when you go public without one.

## Demo videos

`deploy/demo/` (gitignored) is served at `/demo/` and linked from the suite page's top bar ("Demo videos"). It holds
our rendered synthetic showroom clips (`data/synthetic/share/*/*.mp4`, plus the long and crowded scenes encoded from
their frames). Visitors download a clip and upload it to either panel. `reports/demo_clips.md` says which clips show
each app best.

## Our fine-tuned detector

`models/` is mounted read-only at `/opt/ft` in both app containers. When the fine-tuned RF-DETR Nano ONNX is ready:

1. Copy it to `models/`, e.g. `models/showroom_detector_ft.onnx`.
2. In `deploy/.env`:
   ```
   SHOWROOM_RFDETR_FT=/opt/ft/showroom_detector_ft.onnx
   SHOWROOM_DETECTOR=rfdetr_ft@2        # Coach upload mode (optional: SHOWROOM_LIVE_DETECTOR for live)
   PULSE_RFDETR_FT=/opt/ft/showroom_detector_ft.onnx   # Pulse: people only; cars keep the COCO detector
   ```
3. Run `deploy/up.sh` again. Results cached under the old weights are not reused, because the weights' hash is part of
   each app's cache key.

Switch the default only if the fine-tuned model beats each app's benchmarked default on held-out clips.

## Notes

- **Compute:** CPU inference (onnxruntime), `APP_CPUS` cores per app (default 16). GPU 0 belongs to other
  services on this machine.
- **Ports:** only `127.0.0.1:8090` is published. Change it with `WEB_PORT`.
- **Licence:** the Coach image contains imageio-ffmpeg's GPL FFmpeg build (see `showroom-coach/Dockerfile`). Running
  the image on our own machine is not distribution. Don't push the image to a public registry.
