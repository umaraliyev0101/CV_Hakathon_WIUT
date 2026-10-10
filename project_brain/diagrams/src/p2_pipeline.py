"""P2 - Technical runtime pipeline: one case, from phone video to rule verdicts.
Folded: pool 1 = capture + 3D + parallel vision branches; pool 2 = physics, rules, report."""
import common
import bpmn_emit as E
from bpmn_emit import Node, Edge, col, cy, stack_lanes

PH, API, GPU, RUL = "#eaf1f7", "#f5f2ea", "#eef4ec", "#f3eef7"
lanes, top = stack_lanes([
    ("Phone web app", 1, PH), ("API backend", 1, API), ("GPU worker", 4, GPU),          # Pool A
    ("GPU worker", 1, GPU), ("Rules + report", 2, RUL), ("API backend", 1, API), ("Phone web app", 1, PH),  # Pool B
], gap_after={2: 60})
A_P, A_A, A_G, B_G, B_R, B_A, B_P = top

nodes = [
    # Pool A
    Node("s", "event_start", "Driver taps\nStart capture", col(0), cy(A_P, 0)),
    Node("p1", "task", "2.1 Record video,\nupload + GPS", col(1), cy(A_P, 0)),
    Node("a1", "task", "2.2 Create case,\nqueue GPU job", col(2), cy(A_A, 0)),
    Node("q1", "task", "2.3 Sample frames,\nquality check, blur\nplates + faces", col(3), cy(A_G, 1)),
    Node("g_fr", "gateway", "Enough sharp\nframes?", col(4), cy(A_G, 1), "exclusive", lpos="bottom"),
    Node("a2", "task", "2.x Return retake\nhint to phone", col(4), cy(A_A, 0)),
    Node("x_rt", "event_end", "Retake\nrequested", col(4), cy(A_P, 0), lpos="right"),
    Node("v1", "task", "2.4 Poses + points\n(VGGT; COLMAP\nfallback)", col(5), cy(A_G, 1)),
    Node("fork", "gateway", "", col(6), cy(A_G, 1), "parallel"),
    Node("b0", "task", "2.5 Cars: detect,\nsegment, 3D box,\nmodel, metric scale", col(7), cy(A_G, 0)),
    Node("b1", "task", "2.6 Damage masks\nlifted onto 3D cars", col(7), cy(A_G, 1)),
    Node("b2", "task", "2.7 Road: lanes,\nsigns, markings\n+ OpenStreetMap", col(7), cy(A_G, 2)),
    Node("gs1", "task", "2.8 Train Gaussian\nsplat (gsplat,\nbackground)", col(7), cy(A_G, 3)),
    Node("join", "gateway", "", col(8), cy(A_G, 1), "parallel"),
    Node("gs2", "task", "2.9 Publish 3D\nreplay to storage", col(8), cy(A_G, 3)),
    Node("throwA", "event_link_throw", "A", col(9), cy(A_G, 1)),
    Node("x_3d", "event_end", "3D replay\nready", col(9), cy(A_G, 3)),
    # Pool B
    Node("catchA", "event_link_catch", "A", col(0), cy(B_G, 0)),
    Node("ph", "task", "2.10 Physics: impact\ndirection, speed\nrange (Monte Carlo)", col(1), cy(B_G, 0)),
    Node("r1", "task", "2.11 Evaluate rule\npredicates on the\nscene graph", col(2), cy(B_R, 0)),
    Node("d_rules", "doc", "Traffic rules\n(Res. 172)", col(2), cy(B_R, 1)),
    Node("g_conf", "gateway", "Confidence\n>= 0.6?", col(3), cy(B_R, 0), "exclusive", lpos="top"),
    Node("v_yes", "task", "2.12 Verdict: likely\nviolated / not", col(4), cy(B_R, 0)),
    Node("v_no", "task", "2.13 Verdict: cannot\ntell + missing\nevidence", col(4), cy(B_R, 1)),
    Node("m", "gateway", "", col(5), cy(B_R, 0), "exclusive"),
    Node("llm", "task", "2.14 LLM cites clause\ntext, writes UZ / RU\nreport", col(6), cy(B_R, 0)),
    Node("save", "task", "2.15 Save case,\nrender PDF", col(7), cy(B_A, 0)),
    Node("d_pdf", "doc", "Case report\n(PDF)", col(8), cy(B_A, 0)),
    Node("show", "task", "2.16 Show case page:\nverdicts + 3D link", col(8), cy(B_P, 0)),
    Node("ee", "event_end", "Report\ndelivered", col(9), cy(B_P, 0)),
]

edges = [
    Edge("s", "p1"),
    Edge("p1", "a1", exit=(1, 0.5), entry=(0.5, 0), points=[(col(2), cy(A_P, 0))]),
    Edge("a1", "q1", exit=(1, 0.5), entry=(0.5, 0), points=[(col(3), cy(A_A, 0))]),
    Edge("q1", "g_fr"),
    Edge("g_fr", "v1", "Yes"),
    Edge("g_fr", "a2", "No", accent=True, exit=(0.5, 0), entry=(0.5, 1), loff=(8, 0)),
    Edge("a2", "x_rt", accent=True, exit=(0.5, 0), entry=(0.5, 1)),
    Edge("v1", "fork"),
    Edge("fork", "b0", exit=(0.5, 0), entry=(0, 0.5), points=[(col(6), cy(A_G, 0))]),
    Edge("fork", "b1"),
    Edge("fork", "b2", exit=(0.5, 1), entry=(0, 0.5), points=[(col(6), cy(A_G, 2))]),
    Edge("fork", "gs1", exit=(0.5, 1), entry=(0, 0.5), points=[(col(6), cy(A_G, 3))]),
    Edge("b0", "join", exit=(1, 0.5), entry=(0.5, 0), points=[(col(8), cy(A_G, 0))]),
    Edge("b1", "join"),
    Edge("b2", "join", exit=(1, 0.5), entry=(0.5, 1), points=[(col(8), cy(A_G, 2))]),
    Edge("join", "throwA"),
    Edge("gs1", "gs2"),
    Edge("gs2", "x_3d"),
    # Pool B
    Edge("catchA", "ph"),
    Edge("ph", "r1", exit=(1, 0.5), entry=(0.5, 0), points=[(col(2), cy(B_G, 0))]),
    Edge("r1", "d_rules", dotted=True, exit=(0.5, 1), entry=(0.5, 0)),
    Edge("r1", "g_conf"),
    Edge("g_conf", "v_yes", "Yes"),
    Edge("g_conf", "v_no", "No", exit=(0.5, 1), entry=(0, 0.5), points=[(col(3), cy(B_R, 1))]),
    Edge("v_yes", "m"),
    Edge("v_no", "m", exit=(1, 0.5), entry=(0.5, 1), points=[(col(5), cy(B_R, 1))]),
    Edge("m", "llm"),
    Edge("llm", "save", exit=(1, 0.5), entry=(0.5, 0), points=[(col(7), cy(B_R, 0))]),
    Edge("save", "d_pdf", dotted=True),
    Edge("save", "show", exit=(0.5, 1), entry=(0, 0.5), points=[(col(7), cy(B_P, 0))]),
    Edge("show", "ee"),
]

TITLE = "P2 - Technical pipeline: one case, from phone video to rule verdicts"
SUB = "Instant report about 1-3 min; full 3D replay 5-15 min in the background (estimates, to be measured)"
STD = "Models: VGGT-1B-Commercial, COLMAP/GLOMAP, RT-DETR + SAM 2, gsplat  |  Map: OpenStreetMap (ODbL)  |  The LLM writes text only, never the verdict"

if __name__ == "__main__":
    E.render(title=TITLE, subtitle=SUB, standards=STD, lanes=lanes, nodes=nodes, edges=edges,
             out_path="p2.drawio")
    print("wrote p2.drawio")
