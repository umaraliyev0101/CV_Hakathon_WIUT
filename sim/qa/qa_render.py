"""qa_render: frame count, exposure, magenta, mask colours, box bounds, in-view coverage."""
import sys

import numpy as np

from sim.qa.common import REPO, clip_arg, load_qa, report
from sim import labels


def run(clip_dir, qa):
    q = qa["render"]
    fails, info = [], []
    gt = labels.read_gt(clip_dir / "gt.json")
    W, H = gt["width"], gt["height"]
    rgb = sorted((clip_dir / "rgb").glob("*.png"))
    masks = sorted((clip_dir / "mask").glob("*.png"))
    expected = q["expected_frames"]
    if q.get("expected_frames_per_clip"):
        from sim import schema
        cfg = schema.load_scene(REPO / "sim" / gt["generator"]["config"])
        expected = len(schema.output_scene_frames(cfg))
        info.append(f"expected frames {expected} (from {gt['generator']['config']})")
    for name, files in (("rgb", rgb), ("mask", masks), ("gt frames", gt["frames"])):
        if len(files) != expected:
            fails.append(f"{name} count {len(files)} != {expected}")
    clip_max = q["clipped_frac_max_glare"] if gt.get("lighting") == "glare" else q["clipped_frac_max"]
    lumas, clips, mags = [], [], []
    mr = q["magenta_rule"]
    for k, f in enumerate(rgb):
        img = labels.read_png_rgb(f).astype(np.float32)
        if img.shape[:2] != (H, W):
            fails.append(f"rgb {k}: shape {img.shape[:2]} != {(H, W)}")
            continue
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        m, c = float(luma.mean()), float((luma >= q["clip_luma_level"]).mean())
        mag = float(((img[..., 0] >= mr["r_min"]) & (img[..., 2] >= mr["b_min"]) & (img[..., 1] <= mr["g_max"])).mean())
        lumas.append(m), clips.append(c), mags.append(mag)
        if m < q["mean_luma_min"]:
            fails.append(f"rgb {k}: mean luma {m:.1f} < {q['mean_luma_min']}")
        if c > clip_max:
            fails.append(f"rgb {k}: clipped {c:.3%} > {clip_max:.0%}")
        if mag > q["magenta_frac_max"]:
            fails.append(f"rgb {k}: magenta {mag:.3%} > {q['magenta_frac_max']:.1%}")
    if lumas:
        info.append(f"mean luma min/mean/max {min(lumas):.1f}/{np.mean(lumas):.1f}/{max(lumas):.1f}; "
                    f"max clipped {max(clips):.3%}; max magenta {max(mags):.4%}")
    colours = {p["id"]: tuple(p["mask_rgb"]) for p in gt["persons"]}
    unknown_total = 0
    for k, f in enumerate(masks):
        u = labels.unknown_colour_px(labels.read_png_rgb(f), colours)
        unknown_total += u
        if u:
            fails.append(f"mask {k}: {u} px of unknown colour")
    info.append(f"unknown mask px total {unknown_total} over {len(masks)} masks")
    for fr in gt["frames"]:
        for p in fr["people"]:
            x1, y1, x2, y2 = p["box_xyxy"]
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                fails.append(f"t={fr['t']}: person {p['id']} box {p['box_xyxy']} outside image")
    # track continuity: fail on a gap > max_track_gap_frames during which the person was well
    # visible (visible_frac = mask px / in-image solo px > track_gap_visible_frac, with an in-image
    # silhouette of at least min_mask_px), i.e. a label dropout. Gaps while the person is out of the
    # image or occluded are legitimate.
    min_px = qa["labels"]["min_mask_px"]
    for pers in gt["persons"]:
        col = tuple(pers["mask_rgb"])
        seen = [any(p["id"] == pers["id"] for p in fr["people"]) for fr in gt["frames"]]
        run = 0
        for k, s_ in enumerate(seen):
            vis_gap = False
            if not s_:
                solo = labels.read_png_rgb(clip_dir / "solo" / f"p{pers['id']}" / f"{k:04d}.png")
                full = int(np.all(solo == np.asarray(col, np.uint8), axis=-1).sum())
                if full >= min_px:
                    mk = labels.read_png_rgb(clip_dir / "mask" / f"{k:04d}.png")
                    vis_gap = labels.visible_frac(mk, solo, col) > q["track_gap_visible_frac"]
            run = run + 1 if vis_gap else 0
            if run == q["max_track_gap_frames"] + 1:
                fails.append(f"person {pers['id']}: unlabelled for > {q['max_track_gap_frames']} frames "
                             f"(ending frame {k}) while visible_frac > {q['track_gap_visible_frac']}")
    n = len(gt["frames"]) or 1
    for pers in gt["persons"]:
        seen = sum(any(p["id"] == pers["id"] for p in fr["people"]) for fr in gt["frames"])
        info.append(f"person {pers['id']} ({pers['role']}) labelled in {seen}/{n} frames ({seen / n:.1%})")
        if seen / n < q["min_in_view_frac"]:
            fails.append(f"person {pers['id']} in view {seen / n:.1%} < {q['min_in_view_frac']:.0%}")
    return fails, info


def main(argv=sys.argv):
    clip = clip_arg(argv)
    fails, info = run(clip, load_qa())
    return report("qa_render", fails, info)


if __name__ == "__main__":
    sys.exit(main())
