"""qa_video: ffprobe codec/pix_fmt/fps/duration; OpenCV decodes first and last frames."""
import json
import subprocess
import sys
from fractions import Fraction

from sim.qa.common import clip_arg, load_qa, report
from sim import labels


def run(clip_dir, qa):
    q = qa["video"]
    gt = labels.read_gt(clip_dir / "gt.json")
    mp4 = clip_dir / f"{gt['clip_id']}.mp4"
    fails, info = [], []
    if not mp4.exists():
        return [f"missing {mp4}"], info
    pr = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,width,height:format=duration",
         "-of", "json", str(mp4)], text=True))
    st, fmt = pr["streams"][0], pr["format"]
    fps = float(Fraction(st["r_frame_rate"]))
    dur = float(fmt["duration"])
    nfr = int(st["nb_read_frames"])
    info.append(f"{mp4.name}: codec={st['codec_name']} pix_fmt={st['pix_fmt']} fps={st['r_frame_rate']} "
                f"avg={st['avg_frame_rate']} frames={nfr} duration={dur:.3f}s size={st['width']}x{st['height']}")
    if st["codec_name"] != q["codec"]:
        fails.append(f"codec {st['codec_name']} != {q['codec']}")
    if st["pix_fmt"] != q["pix_fmt"]:
        fails.append(f"pix_fmt {st['pix_fmt']} != {q['pix_fmt']}")
    if abs(fps - q["fps"]) > 1e-6 or abs(float(Fraction(st["avg_frame_rate"])) - q["fps"]) > 1e-6:
        fails.append(f"fps {st['r_frame_rate']} / avg {st['avg_frame_rate']} != {q['fps']}")
    tol = q["duration_tol_frames"] / q["fps"]
    if abs(dur - gt["duration_s"]) > tol + 1e-9:
        fails.append(f"duration {dur:.3f}s vs {gt['duration_s']}s (> {q['duration_tol_frames']} frame)")
    if abs(nfr - len(gt["frames"])) > q["duration_tol_frames"]:
        fails.append(f"frame count {nfr} vs gt {len(gt['frames'])}")
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    ok0, f0 = cap.read()
    n_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_cv - 1))
    ok1, f1 = cap.read()
    cap.release()
    info.append(f"opencv: frame_count={n_cv} first_ok={ok0} last_ok={ok1}")
    if not ok0 or f0 is None:
        fails.append("OpenCV cannot decode the first frame")
    if not ok1 or f1 is None:
        fails.append("OpenCV cannot decode the last frame")
    return fails, info


def main(argv=sys.argv):
    fails, info = run(clip_arg(argv), load_qa())
    return report("qa_video", fails, info)


if __name__ == "__main__":
    sys.exit(main())
