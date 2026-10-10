"""
bpmn_emit.py — geometry-driven draw.io BPMN emitter.

The composer (you) decides every position; this only draws it. There is NO layout
engine here on purpose — deliberate composition is what makes a diagram look
human-made rather than auto-generated. Styles are draw.io's own BPMN library
(verified via the drawio MCP search_shapes): events/gateways carry explicit
perimeter points and a white-masked label whose side is configurable.

Grid primitives (use these in your layout module):
    GRID=20, PITCH_X=180, TRACK_H=80
    col(i)                  -> center x of column i
    cy(lane_top, track)     -> center y of a node in `track` of a lane at `lane_top`
    lane_height(n_tracks)   -> pixel height of a lane with n tracks
    stack_lanes([...])      -> assign y to a top-to-bottom list of (name, n_tracks, tint)

Public types:
    Lane(name, x, y, w, h, tint)
    Node(id, kind, label, cx, cy, ntype="", lpos="")   # cx,cy = CENTER
    Edge(src, dst, label="", points=[], exit=None, entry=None,
         accent=False, dotted=False, lx=0.0)
    render(title, subtitle, standards, lanes, nodes, edges, out_path)

kind: task | gateway | event_start | event_end | event_link_throw | event_link_catch | doc
ntype: gateway-> exclusive|parallel|inclusive ; task-> user|service|send|manual|receive|script ;
       event-> timer|message|signal|error
lpos: label side for gateway/event labels -> top|bottom|left|right (defaults: gateway top, event bottom)
"""
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

GRID = 20
PITCH_X = 180
TRACK_H = 80
LANE_PAD = 20
COL0 = 140
HEADER_TOP = 100   # content starts below the title/standards band

SIZES = {
    "task": (150, 64), "gateway": (50, 50),
    "event_start": (44, 44), "event_end": (44, 44),
    "event_link_throw": (44, 44), "event_link_catch": (44, 44),
    "doc": (80, 54),
}
# explicit perimeter connection points (from draw.io's own BPMN shapes)
_ELLIPSE_PTS = "points=[[0.145,0.145,0],[0.5,0,0],[0.855,0.145,0],[1,0.5,0],[0.855,0.855,0],[0.5,1,0],[0.145,0.855,0],[0,0.5,0]];"
_RHOMBUS_PTS = "points=[[0.25,0.25,0],[0.5,0,0],[0.75,0.25,0],[1,0.5,0],[0.75,0.75,0],[0.5,1,0],[0.25,0.75,0],[0,0.5,0]];"
_GW = {"exclusive": "exclusive", "x": "exclusive", "×": "exclusive",
       "parallel": "parallel", "+": "parallel", "inclusive": "inclusive", "○": "inclusive"}


# ---------- grid helpers ----------
def col(i, pitch=PITCH_X, x0=COL0):
    return x0 + i * pitch

def cy(lane_top, track):
    return lane_top + LANE_PAD + track * TRACK_H + (TRACK_H // 2)

def lane_height(n_tracks):
    return n_tracks * TRACK_H + 2 * LANE_PAD

def stack_lanes(specs, x=40, w=1860, top=HEADER_TOP, gap_after=None):
    """specs: list of (name, n_tracks, tint). gap_after: dict {index: extra_gap} to separate pools.
    Returns (lanes, tops) where tops[i] is the lane's top y (for cy())."""
    lanes, tops, y = [], [], top
    for i, (name, n, tint) in enumerate(specs):
        h = lane_height(n)
        lanes.append(Lane(name, x, y, w, h, tint))
        tops.append(y)
        y += h
        if gap_after and i in gap_after:
            y += gap_after[i]
    return lanes, tops


@dataclass
class Lane:
    name: str; x: int; y: int; w: int; h: int; tint: str = "#f5f5f5"


@dataclass
class Node:
    id: str; kind: str; label: str; cx: int; cy: int; ntype: str = ""; lpos: str = ""
    def box(self):
        w, h = SIZES[self.kind]
        return (self.cx - w // 2, self.cy - h // 2, w, h)


@dataclass
class Edge:
    src: str; dst: str; label: str = ""
    points: list = field(default_factory=list)
    exit: tuple = None; entry: tuple = None
    accent: bool = False; dotted: bool = False; lx: float = 0.0
    loff: tuple = None   # (dx, dy) label offset override, e.g. (16, 0) beside a short vertical hop


def _label_dir(lpos, default):
    p = lpos or default
    return {
        "top":    "verticalLabelPosition=top;verticalAlign=bottom;labelPosition=center;align=center;",
        "bottom": "verticalLabelPosition=bottom;verticalAlign=top;labelPosition=center;align=center;",
        "left":   "verticalLabelPosition=middle;verticalAlign=middle;labelPosition=left;align=right;",
        "right":  "verticalLabelPosition=middle;verticalAlign=middle;labelPosition=right;align=left;",
    }[p]


def _node_style(n: Node) -> str:
    if n.kind == "task":
        return ("rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#ffffff;"
                "strokeColor=#23344d;fontColor=#11181f;fontSize=11;align=center;"
                "verticalAlign=middle;spacing=6;")
    if n.kind == "gateway":
        g = _GW.get(n.ntype, "exclusive")
        return (_RHOMBUS_PTS + "shape=mxgraph.bpmn.gateway2;html=1;perimeter=rhombusPerimeter;"
                f"outlineConnect=0;outline=none;background=outline;gwType={g};"
                + _label_dir(n.lpos, "top") +
                "fillColor=#ffffff;strokeColor=#23344d;fontColor=#11181f;fontSize=10;"
                "labelBackgroundColor=#ffffff;")
    if n.kind in ("event_start", "event_end"):
        outline = "standard" if n.kind == "event_start" else "end"
        sym = n.ntype if n.ntype in ("timer", "message", "signal", "error") else "general"
        sw = "3" if n.kind == "event_end" else "1.5"
        return (_ELLIPSE_PTS + "shape=mxgraph.bpmn.event;html=1;perimeter=ellipsePerimeter;"
                f"outlineConnect=0;aspect=fixed;outline={outline};symbol={sym};"
                + _label_dir(n.lpos, "bottom") +
                f"fillColor=#ffffff;strokeColor=#23344d;strokeWidth={sw};fontColor=#11181f;"
                "fontSize=10;labelBackgroundColor=#ffffff;")
    if n.kind in ("event_link_throw", "event_link_catch"):
        outline = "throwing" if n.kind == "event_link_throw" else "catching"
        return (_ELLIPSE_PTS + "shape=mxgraph.bpmn.event;html=1;perimeter=ellipsePerimeter;"
                f"outlineConnect=0;aspect=fixed;outline={outline};symbol=link;"
                + _label_dir(n.lpos, "bottom") +
                "fillColor=#ffffff;strokeColor=#23344d;strokeWidth=1.5;fontColor=#11181f;"
                "fontSize=10;labelBackgroundColor=#ffffff;")
    if n.kind == "doc":
        return ("shape=note;size=14;whiteSpace=wrap;html=1;fillColor=#fbfbf7;"
                "strokeColor=#9a9a8a;fontColor=#3a3a30;fontSize=9;align=center;verticalAlign=middle;")
    return "rounded=0;html=1;"


def _edge_style(e: Edge) -> str:
    if e.dotted:
        return ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;dashed=1;dashPattern=2 4;"
                "endArrow=none;startArrow=none;strokeColor=#9a9a8a;jettySize=auto;")
    col_ = "#c0392b" if e.accent else "#3a4a5a"
    s = ("edgeStyle=orthogonalEdgeStyle;rounded=1;arcSize=8;jumpStyle=arc;jumpSize=10;"
         "orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;"
         f"strokeColor={col_};fontColor=#33414f;fontSize=10;labelBackgroundColor=#ffffff;")
    if e.exit:
        s += f"exitX={e.exit[0]};exitY={e.exit[1]};exitDx=0;exitDy=0;"
    if e.entry:
        s += f"entryX={e.entry[0]};entryY={e.entry[1]};entryDx=0;entryDy=0;"
    return s


def _cell(cid, value, style, x, y, w, h, parent="1"):
    geo = f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
    return (f'<mxCell id="{cid}" value="{escape(value or "")}" style="{style}" '
            f'parent="{parent}" vertex="1">{geo}</mxCell>')


def render(title, subtitle, standards, lanes, nodes, edges, out_path, caption=None):
    node_ids = {n.id for n in nodes}
    if len(node_ids) != len(nodes):
        dupes = sorted({n.id for n in nodes if [m.id for m in nodes].count(n.id) > 1})
        raise ValueError(f"duplicate node ids: {dupes}")
    for e in edges:
        missing = [t for t in (e.src, e.dst) if t not in node_ids]
        if missing:
            raise ValueError(f"edge {e.src}->{e.dst} references unknown node(s): {missing}")

    # header band + page track the actual content extent — a fixed 1600 band under- or
    # overshoots the pool and a fixed page cuts wide diagrams at the page break in print
    ext = [(ln.x, ln.y, ln.w, ln.h) for ln in lanes] + [n.box() for n in nodes]
    x0 = min(b[0] for b in ext); x1 = max(b[0] + b[2] for b in ext)
    y1 = max(b[1] + b[3] for b in ext)
    hdr_w = x1 - x0

    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    cells.append(_cell("hdr_t", title,
        "text;html=1;align=center;fontSize=16;fontStyle=1;fontColor=#11181f;", x0, 8, hdr_w, 24))
    if subtitle:
        cells.append(_cell("hdr_s", subtitle,
            "text;html=1;align=center;fontSize=11;fontStyle=2;fontColor=#52606d;", x0, 34, hdr_w, 18))
    cap = caption if caption is not None else (("Standards: " + standards) if standards else None)
    if cap:
        cells.append(_cell("hdr_std", cap,
            "rounded=0;html=1;fillColor=#eef1f5;strokeColor=#cdd5df;fontSize=10;fontColor=#33414f;"
            "align=center;verticalAlign=middle;", x0, 58, hdr_w, 22))
    used = {"0", "1", "hdr_t", "hdr_s", "hdr_std"} | node_ids
    for i, ln in enumerate(lanes):
        lid = f"lane{i}"
        while lid in used: lid += "_"
        used.add(lid)
        cells.append(_cell(lid, ln.name,
            f"swimlane;horizontal=0;startSize=28;fillColor={ln.tint};strokeColor=#9aa7b4;"
            "swimlaneFillColor=#ffffff;fontColor=#33414f;fontStyle=1;fontSize=12;html=1;"
            "verticalAlign=middle;", ln.x, ln.y, ln.w, ln.h))
    for n in nodes:
        x, y, w, h = n.box()
        cells.append(_cell(n.id, n.label, _node_style(n), x, y, w, h))
    for k, e in enumerate(edges):
        pts = ""
        if e.points:
            pts = '<Array as="points">' + "".join(
                f'<mxPoint x="{px}" y="{py}"/>' for px, py in e.points) + '</Array>'
        lo = e.loff if e.loff else (0, -11)
        off = f'<mxPoint x="{lo[0]}" y="{lo[1]}" as="offset"/>' if e.label else ""
        xattr = f' x="{e.lx}"' if e.lx else ""
        geo = f'<mxGeometry{xattr} relative="1" as="geometry">{pts}{off}</mxGeometry>'
        eid = f"e{k}"
        while eid in used: eid += "_"        # a node named e<number> must not collide
        used.add(eid)
        cells.append(f'<mxCell id="{eid}" value="{escape(e.label)}" style="{_edge_style(e)}" '
                     f'parent="1" source="{e.src}" target="{e.dst}" edge="1">{geo}</mxCell>')
    page_w = max(1600, int(x1) + 40)
    page_h = max(1100, int(y1) + 40)
    xml = ('<mxfile><diagram name="' + escape(title) + '">'
           f'<mxGraphModel dx="1200" dy="800" grid="1" gridSize="{GRID}" guides="1" '
           'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           f'pageWidth="{page_w}" pageHeight="{page_h}" math="0" shadow="0"><root>'
           + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path
