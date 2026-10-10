#!/usr/bin/env python3
"""Build slides/*.json from content/*.txt — one page per idea, identical anatomy.

Block format (blank line between blocks):
  @ Name | Tier | Customer | score | refs
  what: one sentence, what it is            -> subtitle
  how:  Bold | sub || Bold | sub || ...     -> 4-step pipeline (last step = the decision)
  why:  why it matters                      -> card 1
  prac: data, where it runs, cost           -> card 2
  pays: who pays, price                     -> card 3 (top)
  demo: what the judge does                 -> card 3 (bottom)
  moat: why hard to copy                    -> note

Text is wrapped against fixed character budgets; anything that does not fit stops the
build with a FIT error instead of being truncated.
"""
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = "Report: Automotive CV hackathon ideas (10 Oct 2026)"
TIERS = {
    "A": ("Tier A", "Fundable stories",
          "Clear payer, repeat revenue, a moat that compounds and a market beyond Uzbekistan. Pitch one of these."),
    "B": ("Tier B", "Solid B2B niches",
          "Real payers and buildable in a sprint, but smaller markets. Strong modules inside a Tier A platform."),
    "C": ("Tier C", "Great demo, hard investor case",
          "City hall is the only payer, the ticket is too small, or it is a feature of a bigger product."),
}

STEP_X, STEP_W = [10, 291, 572, 853], 236
CARD_X, CARD_W = [10, 375, 740], 350
BUDGET = {"what": (124, 1), "sub": (31, 2), "bold": (21, 1), "card": (41, 6),
          "pays": (41, 2), "demo": (41, 2), "note": (170, 1)}


ERRORS = []

# Wrap by measured pixel width (Helvetica Neue, the deck font) when PIL is available;
# fall back to the character budgets above otherwise.
PX = {"card": 300, "pays": 300, "demo": 300, "sub": 212, "bold": 212}
try:
    from PIL import ImageFont

    def _face(size, weight):
        for i in range(24):
            f = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=i)
            name = " ".join(f.getname())
            if name.endswith(weight) and "Condensed" not in name:
                return f
        raise OSError(weight)

    _m15, _b15, _m13 = _face(15, "Medium"), _face(15, "Bold"), _face(13, "Medium")
    MEASURE = {"card": _m15, "pays": _m15, "demo": _m15, "sub": _m13, "bold": _b15}
except Exception:  # no PIL or no font: character budgets only
    MEASURE = {}


def fit(text, key, where):
    width, max_lines = BUDGET[key]
    if key in ("what", "note"):
        if len(text) > width:
            ERRORS.append(f"FIT {where}.{key}: {len(text)} chars > {width}: {text}")
        return [text]
    font = MEASURE.get(key)
    fits = (lambda l: font.getlength(l) <= PX[key]) if font else (lambda l: len(l) <= width)
    lines, cur = [], ""
    for w in text.split():
        if not cur:
            cur = w
        elif fits(cur + " " + w):
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines or not all(fits(l) for l in lines):
        ERRORS.append(f"FIT {where}.{key}: {len(lines)} lines (max {max_lines} x {width}): {text}")
    return lines


def t(cls, x, y, s, mid=False):
    anchor = " text-anchor='middle'" if mid else ""
    return f"<text class='{cls}' x='{x:g}' y='{y:g}'{anchor}>{html.escape(s, quote=False)}</text>"


def svg(idea, where):
    out = [t("dg-t-a", 10, 16, "HOW IT WORKS")]
    steps = idea["how"]
    if len(steps) != 4:
        sys.exit(f"FIT {where}.how: needs 4 steps, has {len(steps)}")
    for i, (bold, sub) in enumerate(steps):
        x, cx = STEP_X[i], STEP_X[i] + STEP_W / 2
        out.append(f"<rect class='{'dg-box-a' if i == 3 else 'dg-box'}' x='{x}' y='28' width='{STEP_W}' height='90'/>")
        fit(bold, "bold", f"{where}.step{i+1}")
        subs = fit(sub, "sub", f"{where}.step{i+1}")
        y0 = 60 if len(subs) == 2 else 66
        out.append(t("dg-t-b", cx, y0, bold, mid=True))
        out += [t("dg-t-s", cx, y0 + 22 + 17 * j, s, mid=True) for j, s in enumerate(subs)]
        if i < 3:
            out.append(f"<path class='dg-arrow' d='M{x + STEP_W + 5} 73 L{STEP_X[i + 1] - 6} 73'/>")
    cards = [("WHY IT MATTERS", fit(idea["why"], "card", f"{where}.why")),
             ("HOW PRACTICAL", fit(idea["prac"], "card", f"{where}.prac"))]
    for i, (head, lines) in enumerate(cards):
        x = CARD_X[i]
        out.append(f"<rect class='dg-box-m' x='{x}' y='148' width='{CARD_W}' height='187'/>")
        out.append(t("dg-t-a", x + 20, 176, head))
        out += [t("dg-t", x + 20, 204 + 21 * j, s) for j, s in enumerate(lines)]
    x = CARD_X[2]
    out.append(f"<rect class='dg-box-m' x='{x}' y='148' width='{CARD_W}' height='187'/>")
    out.append(t("dg-t-a", x + 20, 176, "WHO PAYS"))
    out += [t("dg-t", x + 20, 204 + 21 * j, s) for j, s in enumerate(fit(idea["pays"], "pays", f"{where}.pays"))]
    out.append(f"<line class='dg-rule' x1='{x + 20}' y1='247' x2='{x + CARD_W - 20}' y2='247'/>")
    out.append(t("dg-t-a", x + 20, 272, "JUDGE DEMO"))
    out += [t("dg-t", x + 20, 297 + 21 * j, s) for j, s in enumerate(fit(idea["demo"], "demo", f"{where}.demo"))]
    return "".join(out)


def parse(path):
    ideas, cur = [], None
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            name, tier, cust, score, refs = [p.strip() for p in line[1:].split("|")]
            cur = {"name": name, "tier": tier, "cust": cust, "score": score, "refs": refs}
            ideas.append(cur)
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "how":
            cur["how"] = [tuple(p.strip() for p in step.split("|")) for step in val.split("||")]
        else:
            cur[key] = val
    return ideas


def parse_plain(folder):
    """content/plain/*.txt: '@ Name' then tag / story / uses (|| separated) / buyers."""
    out, cur = {}, None
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".txt"):
            continue
        for raw in open(os.path.join(folder, f), encoding="utf-8"):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                cur = out.setdefault(line[1:].strip(), {})
                continue
            key, _, val = line.partition(":")
            cur[key.strip()] = [u.strip() for u in val.split("||")] if key.strip() == "uses" else val.strip()
    return out


def main():
    plain = parse_plain(os.path.join(ROOT, "content", "plain"))
    files = sorted(f for f in os.listdir(os.path.join(ROOT, "content")) if f.endswith(".txt"))
    ideas = [i for f in files for i in parse(os.path.join(ROOT, "content", f))]
    total = len(ideas)
    names = [i["name"] for i in ideas]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        sys.exit(f"duplicate ideas: {dupes}")
    missing, extra = set(names) - set(plain), set(plain) - set(names)
    if (missing and not os.environ.get("ALLOW_MISSING")) or extra:
        ERRORS.append(f"plain-language pages: missing {sorted(missing)}, unknown {sorted(extra)}")
    by_tier = {}
    for rank, idea in enumerate(ideas, 1):
        idea["rank"] = rank
        by_tier.setdefault(idea["tier"], []).append(idea)
    for n, tier in enumerate(["A", "B", "C"], 1):
        num, title, sub = TIERS[tier]
        group = by_tier.get(tier, [])
        slides = [{"layout": "section", "theme": "dark", "num": f"{num} · {len(group)} ideas",
                   "title": title, "subtitle": sub}]
        for idea in group:
            where = idea["name"]
            for k in ("what", "why", "prac", "pays", "demo", "moat", "how"):
                if k not in idea:
                    sys.exit(f"{where}: missing {k}")
            eyebrow = f"{num} · {idea['rank']:02d} / {total} · {idea['cust']}"
            if len(eyebrow) > 46:
                sys.exit(f"FIT {where}.eyebrow: {eyebrow}")
            src = f"{SRC}" + (f", refs {idea['refs']}" if idea["refs"] not in ("", "-") else "")
            pl = plain.get(where)
            if pl:
                for k in ("tag", "story", "why", "uses", "buyers"):
                    if k not in pl:
                        ERRORS.append(f"{where}: plain page missing {k}")
                words = len(" ".join([pl.get("tag", ""), pl.get("story", ""), pl.get("why", ""), *pl.get("uses", []), pl.get("buyers", "")]).split())
                if words > 150:
                    ERRORS.append(f"FIT {where}.plain: {words} words > 150")
                for k, cap in (("story", 360), ("why", 230), ("buyers", 130)):
                    if len(pl.get(k, "")) > cap:
                        ERRORS.append(f"FIT {where}.{k}: {len(pl[k])} chars > {cap}")
                if len(pl.get("uses", [])) not in (3, 4) or any(len(u) > 84 for u in pl.get("uses", [])):
                    ERRORS.append(f"FIT {where}.uses: need 3-4 items of <=84 chars: {pl.get('uses')}")
                if len(pl.get("tag", "")) > 90:
                    ERRORS.append(f"FIT {where}.tag: {len(pl['tag'])} chars > 90 (keep the tagline on one line)")
                slides.append({"layout": "explain", "eyebrow": eyebrow, "title": idea["name"],
                               "subtitle": pl.get("tag", ""), "story": pl.get("story", ""), "why": pl.get("why", ""),
                               "uses": pl.get("uses", []), "buyers": pl.get("buyers", ""), "source": src})
            note = f"**Moat** — {idea['moat']}  ·  Report score {idea['score']}/100"
            fit(note, "note", where)
            fit(idea["what"], "what", where)
            slides.append({"layout": "diagram", "eyebrow": f"{num} · {idea['rank']:02d} / {total} · How it works", "title": idea["name"],
                           "subtitle": idea["what"], "viewBox": "0 0 1100 340",
                           "svg": svg(idea, where), "note": note, "source": src})
        if ERRORS:
            continue
        out = os.path.join(ROOT, "slides", f"{n:02d}-tier-{tier.lower()}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(slides, fh, ensure_ascii=False, indent=1)
        print(f"{out}: {len(slides)} slides")
    if ERRORS:
        print("\n".join(ERRORS))
        sys.exit(f"{len(ERRORS)} fit errors")
    print(f"{total} ideas")


if __name__ == "__main__":
    main()
