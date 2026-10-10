"""
bpmn_validate.py — Gate 1 (programmatic) of the two-gate quality check.

Usage:  python bpmn_validate.py <layout_module>      # module must expose: nodes, lanes, edges
Exits non-zero if any FAIL. WARN items are advisory. Gate 2 (your eyes on the render)
still runs after this — Gate 1 catches the measurable defects, your eye catches the rest.

Checks (see RUBRIC.md):
  R1  aspect <=2.6                R2  every node inside a lane box (no pool overflow)
  R3  lanes sized to content      R4  grid
  R5  no node overlap             R6  uniform sizing
  R7  column/track alignment      R8  DECISION-gateway out-edges labelled (parallel forks exempt)
  R9  edge bends <=3 (advisory)   R10 lane header text fits lane height
  R11 node label text fits its box (advisory, conservative estimate)
  X1  no edge segment through a non-endpoint node body (both elbow orientations;
      a crossing in only ONE orientation is a WARN "X1?" — pin a waypoint)
  X2  same for a node's external label box (gateway/event names)
  X3  no explicit waypoint INSIDE any node box (incl. the edge's own src/dst —
      a waypoint at the target's centre makes the arrow overshoot into the box)
  X4  anchored edges approach from the anchor's side (an entry on the bottom
      approached from above = a line straight through the box)
  X5  segment passing <8px from a box reads as touching (advisory)
  X6  two edges running <10px apart for >40px render as an ugly double line
      (advisory; EXACTLY collinear shared buses are fine and render as one)
NOTE: edge-LABEL crossings (a label struck by another edge) are hard to verify
programmatically; rely on Gate 2 for those, and use Edge.lx to move a label to a clear segment.
"""
import math
import sys, importlib
from collections import defaultdict
from bpmn_emit import SIZES, GRID, PITCH_X, TRACK_H

LABEL_W, LABEL_H, LABEL_GAP = 84, 16, 6   # approx external-label box for gateways/events


def _anchor(box, frac):
    x, y, w, h = box
    return (x + frac[0] * w, y + frac[1] * h)

def _seg_hits(p, q, box, pad=2):
    bx, by, bw, bh = box; bx -= pad; by -= pad; bw += 2*pad; bh += 2*pad
    x1, y1 = p; x2, y2 = q
    if x1 == x2 and bx < x1 < bx+bw:
        lo, hi = sorted((y1, y2));  return not (hi <= by or lo >= by+bh)
    if y1 == y2 and by < y1 < by+bh:
        lo, hi = sorted((x1, x2));  return not (hi <= bx or lo >= bx+bw)
    return False

def _raw(e, boxes):
    sb, tb = boxes[e.src], boxes[e.dst]
    start = _anchor(sb, e.exit) if e.exit else (sb[0]+sb[2]/2, sb[1]+sb[3]/2)
    end   = _anchor(tb, e.entry) if e.entry else (tb[0]+tb[2]/2, tb[1]+tb[3]/2)
    return [start] + list(e.points) + [end]

def _polyline(raw, hfirst=True):
    """Expand diagonal hops into orthogonal elbows. draw.io picks its own elbow
    orientation, so callers must consider BOTH hfirst=True and hfirst=False."""
    poly = [raw[0]]
    for nx in raw[1:]:
        if poly[-1][0] != nx[0] and poly[-1][1] != nx[1]:
            poly.append((nx[0], poly[-1][1]) if hfirst else (poly[-1][0], nx[1]))
        poly.append(nx)
    return poly

def _label_box(n):
    """approx external-label rectangle for a gateway/event by its lpos side."""
    x, y, w, h = n.box()
    side = n.lpos or ("top" if n.kind == "gateway" else "bottom")
    cx = x + w/2
    if side == "top":    return (cx-LABEL_W/2, y-LABEL_H-LABEL_GAP, LABEL_W, LABEL_H)
    if side == "bottom": return (cx-LABEL_W/2, y+h+LABEL_GAP, LABEL_W, LABEL_H)
    if side == "left":   return (x-LABEL_W-LABEL_GAP, y+h/2-LABEL_H/2, LABEL_W, LABEL_H)
    return (x+w+LABEL_GAP, y+h/2-LABEL_H/2, LABEL_W, LABEL_H)

def _side(frac):
    fx, fy = frac
    if fy == 0 and 0 < fx < 1: return "top"
    if fy == 1 and 0 < fx < 1: return "bottom"
    if fx == 0 and 0 < fy < 1: return "left"
    if fx == 1 and 0 < fy < 1: return "right"
    return None

def _hits(poly, e, boxes, label_boxes):
    nhits, lhits = set(), set()
    for i in range(len(poly)-1):
        for nid, box in boxes.items():
            if nid in (e.src, e.dst): continue
            if _seg_hits(poly[i], poly[i+1], box): nhits.add(nid)
        for nid, lb in label_boxes:
            if nid in (e.src, e.dst): continue
            if _seg_hits(poly[i], poly[i+1], lb): lhits.add(nid)
    return nhits, lhits


def check(mod_name):
    m = importlib.import_module(mod_name)
    nodes, lanes, edges = m.nodes, m.lanes, m.edges
    boxes = {n.id: n.box() for n in nodes}
    fails, warns = [], []

    xs = [b[0] for b in boxes.values()]; xe = [b[0]+b[2] for b in boxes.values()]
    ys = [b[1] for b in boxes.values()]; ye = [b[1]+b[3] for b in boxes.values()]
    W, H = max(xe)-min(xs), max(ye)-min(ys)
    ar = W / H
    if ar > 2.6: fails.append(f"R1 aspect {ar:.2f} > 2.6 (W={W} H={H}) — fold the flow")
    elif ar > 2.3: warns.append(f"R1 aspect {ar:.2f} approaching limit")

    # R2: every node fully inside SOME lane rectangle. A node past the pool edge
    # (the classic doc/link-event overflow at the last column) renders outside the
    # swimlane frame — instantly reads as broken.
    if lanes:
        for n in nodes:
            b = boxes[n.id]
            if not any(b[0] >= ln.x - 2 and b[0]+b[2] <= ln.x+ln.w + 2 and
                       b[1] >= ln.y - 2 and b[1]+b[3] <= ln.y+ln.h + 2 for ln in lanes):
                fails.append(f"R2 node {n.id} is outside every lane box — it spills past "
                             f"the pool edge; move it inside a lane or widen the lanes")

    off = [n.id for n in nodes if n.cx % GRID or n.cy % GRID]
    if off: fails.append(f"R4 off-grid node centers: {off}")

    ids = list(boxes)
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = boxes[ids[i]], boxes[ids[j]]
            ox = min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])
            oy = min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])
            if ox > 0 and oy > 0: fails.append(f"R5 overlap {ids[i]}~{ids[j]}")

    for kind in set(n.kind for n in nodes):
        sz = set(SIZES[kind] for n in nodes if n.kind == kind)
        if len(sz) > 1: fails.append(f"R6 mixed sizes for {kind}")

    colmap = defaultdict(set)
    for n in nodes: colmap[round(n.cx/PITCH_X)].add(n.cx)
    for c, v in colmap.items():
        if len(v) > 1 and (max(v)-min(v)) < 40:
            fails.append(f"R7 near-but-unaligned column ~{c}: {sorted(v)}")

    for ln in lanes:
        # R10: the vertical lane header (fontSize 12 ≈ 6.5px/char + padding) must fit the
        # lane height, else it overflows into the neighbouring lane — invisible to all
        # other checks but the first thing a human notices on the render.
        need = len(ln.name) * 6.5 + 10
        if need > ln.h:
            fails.append(f"R10 lane header '{ln.name}' too long for lane height "
                         f"{ln.h} (needs ~{int(need)}px) — shorten the name or add tracks")

        inner = [b for b in boxes.values()
                 if b[0] >= ln.x and b[0]+b[2] <= ln.x+ln.w and b[1] >= ln.y-2 and b[1]+b[3] <= ln.y+ln.h+2]
        if not inner:
            warns.append(f"R3 lane '{ln.name}' (y={ln.y}) has no nodes"); continue
        top_gap = min(b[1] for b in inner) - (ln.y + 28)
        bot_gap = (ln.y + ln.h) - max(b[1]+b[3] for b in inner)
        if top_gap > 0.6*TRACK_H or bot_gap > 0.6*TRACK_H:
            warns.append(f"R3 lane '{ln.name}' slack top={int(top_gap)} bot={int(bot_gap)}")

    # R11: wrapped label height vs box height, deliberately conservative
    # (0.5×fontSize per char) so it only flags labels that clearly spill out.
    for n in nodes:
        if n.kind not in ("task", "doc") or not n.label: continue
        w, h = boxes[n.id][2], boxes[n.id][3]
        fs = 11 if n.kind == "task" else 9
        inner_w = w - (16 if n.kind == "task" else 12)
        max_lines = int((h - 8) // (fs * 1.25))
        need = sum(max(1, math.ceil(len(line) * fs * 0.5 / inner_w))
                   for line in n.label.split("\n"))
        if need > max_lines:
            warns.append(f"R11 label of {n.id} needs ≈{need} wrapped lines but only "
                         f"{max_lines} fit its {w}x{h} box — text will overflow; "
                         f"shorten the label or use a task instead of a doc")

    # R8: only DECISION gateways need labelled out-edges; a parallel (AND) fork has
    # no conditions, and a converging/merge gateway is self-evident.
    gw_decision = {n.id for n in nodes
                   if n.kind == "gateway" and n.ntype not in ("parallel", "+")}
    for e in edges:
        if e.src in gw_decision and not e.dotted and not e.label:
            outs = [x for x in edges if x.src == e.src and not x.dotted]
            if len(outs) > 1:
                warns.append(f"R8 gateway {e.src} out-edge to {e.dst} unlabelled")

    label_boxes = [(n.id, _label_box(n)) for n in nodes
                   if n.kind in ("gateway", "event_start", "event_end",
                                 "event_link_throw", "event_link_catch") and n.label]
    all_hpolys = []
    for e in edges:
        raw = _raw(e, boxes)
        pH = _polyline(raw, True)
        pV = _polyline(raw, False)
        all_hpolys.append((e, pH))

        bends = sum(1 for i in range(1, len(pH)-1)
                    if (pH[i][0]-pH[i-1][0])*(pH[i+1][0]-pH[i][0]) < 0
                    or (pH[i][1]-pH[i-1][1])*(pH[i+1][1]-pH[i][1]) < 0)
        if bends > 3: warns.append(f"R9 edge {e.src}->{e.dst} has {bends} bends")

        # X1/X2 on BOTH elbow orientations: a hit in both is a certain crossing;
        # a hit in one is routing draw.io may or may not pick — pin a waypoint.
        nH, lH = _hits(pH, e, boxes, label_boxes)
        nV, lV = (nH, lH) if pV == pH else _hits(pV, e, boxes, label_boxes)
        for nid in nH & nV:
            fails.append(f"X1 edge {e.src}->{e.dst} crosses node {nid}")
        for nid in nH ^ nV:
            warns.append(f"X1? edge {e.src}->{e.dst} may cross node {nid} depending on "
                         f"elbow orientation — pin a waypoint to force the clear route")
        for nid in lH & lV:
            fails.append(f"X2 edge {e.src}->{e.dst} crosses LABEL of {nid}")
        for nid in lH ^ lV:
            warns.append(f"X2? edge {e.src}->{e.dst} may cross LABEL of {nid} depending on "
                         f"elbow orientation — pin a waypoint")

        # X3: an explicit waypoint inside ANY box (own endpoints included) makes the
        # rendered line dive through that box — the "arrow ends inside the task" defect.
        for px, py in e.points:
            for nid, (bx, by, bw, bh) in boxes.items():
                if bx+2 < px < bx+bw-2 and by+2 < py < by+bh-2:
                    fails.append(f"X3 edge {e.src}->{e.dst} waypoint ({px},{py}) is INSIDE "
                                 f"node {nid} — move it to a clear corridor / the perimeter")

        # X4: an anchored edge must approach its anchor from that side's outside.
        # Entering a bottom anchor from above draws the line through the whole box.
        if e.entry:
            side = _side(e.entry)
            ax, ay = _anchor(boxes[e.dst], e.entry)
            px, py = raw[-2]
            bad = ((side == "bottom" and py < ay - 2) or (side == "top" and py > ay + 2) or
                   (side == "left" and px > ax + 2) or (side == "right" and px < ax - 2))
            if bad:
                fails.append(f"X4 edge {e.src}->{e.dst} enters the {side} anchor of {e.dst} "
                             f"from the wrong side — the line passes through the box; "
                             f"approach from the {side} or move the anchor")
        if e.exit:
            side = _side(e.exit)
            ax, ay = _anchor(boxes[e.src], e.exit)
            px, py = raw[1]
            bad = ((side == "bottom" and py < ay - 2) or (side == "top" and py > ay + 2) or
                   (side == "left" and px > ax + 2) or (side == "right" and px < ax - 2))
            if bad:
                fails.append(f"X4 edge {e.src}->{e.dst} leaves the {side} anchor of {e.src} "
                             f"toward the wrong side — the line passes back through the box")

        # X5: grazing — close enough to read as touching, but not a crossing.
        for i in range(len(pH)-1):
            if pH[i] == pH[i+1]: continue
            for nid, box in boxes.items():
                if nid in (e.src, e.dst): continue
                if _seg_hits(pH[i], pH[i+1], box, pad=8) and \
                   not _seg_hits(pH[i], pH[i+1], box, pad=2):
                    warns.append(f"X5 edge {e.src}->{e.dst} passes <8px from node {nid} — "
                                 f"reads as touching; give it clearance")

    # X6: near-parallel corridors. Exactly-collinear segments (shared fork/join buses)
    # merge into one drawn line and are fine; 1–10px apart renders as a double line.
    def _segs(poly, horiz):
        out = []
        for i in range(len(poly)-1):
            (x1, y1), (x2, y2) = poly[i], poly[i+1]
            if horiz and y1 == y2 and x1 != x2: out.append((y1, min(x1, x2), max(x1, x2)))
            if not horiz and x1 == x2 and y1 != y2: out.append((x1, min(y1, y2), max(y1, y2)))
        return out
    for horiz in (True, False):
        segs = [(e, s) for e, p in all_hpolys for s in _segs(p, horiz)]
        for i in range(len(segs)):
            for j in range(i+1, len(segs)):
                (e1, (c1, lo1, hi1)), (e2, (c2, lo2, hi2)) = segs[i], segs[j]
                if e1 is e2: continue
                d = abs(c1 - c2)
                if 0.5 < d < 10 and min(hi1, hi2) - max(lo1, lo2) > 40:
                    warns.append(f"X6 edges {e1.src}->{e1.dst} and {e2.src}->{e2.dst} run "
                                 f"{d:.0f}px apart for a long stretch — double-line look; "
                                 f"share the corridor exactly or separate by ≥20px")

    fails = sorted(set(fails)); warns = sorted(set(warns))
    print(f"\n=== {mod_name}: bbox {W}x{H} aspect {ar:.2f}, {len(nodes)} nodes, {len(lanes)} lanes ===")
    print(f"FAILS ({len(fails)}):");  [print("  ✗", f) for f in fails]
    print(f"WARNINGS ({len(warns)}):");  [print("  ·", w) for w in warns]
    ok = not fails
    print("GATE-1:", "PASS ✅ (now do Gate-2: render and LOOK)" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else "layout") else 1)
