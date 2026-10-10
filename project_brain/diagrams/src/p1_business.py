"""P1 - Business flow: minor collision to settled claim (to-be, with Crash Twin).
Folded into two pools joined by link event A: at the scene / claim settlement."""
import common
from common import DRV, CT, POL, INS, SHOP
import bpmn_emit as E
from bpmn_emit import Node, Edge, col, cy, stack_lanes

lanes, top = stack_lanes([
    ("Driver(s)", 1, DRV), ("Crash Twin", 2, CT), ("Traffic police", 1, POL),          # Pool A
    ("Driver", 1, DRV), ("Crash Twin", 1, CT), ("Body shop", 1, SHOP), ("Insurer", 2, INS),  # Pool B
], gap_after={2: 60})
A_D, A_C, A_P, B_D, B_C, B_S, B_I = top

nodes = [
    # Pool A - at the scene
    Node("s", "event_start", "Minor collision\n(no injuries)", col(0), cy(A_D, 0)),
    Node("t11", "task", "1.1 Open Crash Twin link,\nfilm 60 s walk-around\n+ damage close-ups", col(1), cy(A_D, 0)),
    Node("t12", "task", "1.2 Check capture,\nblur plates and faces", col(2), cy(A_C, 0)),
    Node("g_ok", "gateway", "Capture OK?", col(3), cy(A_C, 0), "exclusive", lpos="bottom"),
    Node("t13", "task", "1.3 Build 3D twin,\nmeasure, apply rules", col(4), cy(A_C, 0)),
    Node("t14", "task", "1.4 Send draft report\n+ 3D replay link", col(5), cy(A_C, 0)),
    Node("d_rep", "doc", "Draft case\nreport + 3D", col(5), cy(A_C, 1)),
    Node("g_agree", "gateway", "Drivers agree\non fault?", col(6), cy(A_D, 0), "exclusive", lpos="top"),
    Node("t21", "task", "2.1 Inspector reviews\ntwin, decides fault", col(6), cy(A_P, 0)),
    Node("m_a", "gateway", "", col(7), cy(A_D, 0), "exclusive"),
    Node("t15", "task", "1.5 Move cars,\nclear the lane", col(8), cy(A_D, 0)),
    Node("throwA", "event_link_throw", "A", col(9), cy(A_D, 0)),
    # Pool B - claim settlement
    Node("catchA", "event_link_catch", "A", col(0), cy(B_D, 0)),
    Node("t31", "task", "3.1 File claim with\ncase link", col(1), cy(B_D, 0)),
    Node("t32", "task", "3.2 Register claim,\nopen the case", col(2), cy(B_I, 0)),
    Node("t33", "task", "3.3 Consistency and\nfraud check (score)", col(3), cy(B_C, 0)),
    Node("g_match", "gateway", "Damage\nmatches\nscene?", col(4), cy(B_I, 0), "exclusive", lpos="left"),
    Node("t34", "task", "3.4 Approve repair\nestimate", col(5), cy(B_I, 0)),
    Node("t38", "task", "3.5 Field adjuster\ninspection", col(4), cy(B_I, 1)),
    Node("g_valid", "gateway", "Claim valid?", col(5), cy(B_I, 1), "exclusive", lpos="bottom"),
    Node("x_ref", "event_end", "Claim refused", col(6), cy(B_I, 1)),
    Node("t35", "task", "3.6 Repair intake from\n3D damage map", col(6), cy(B_S, 0)),
    Node("t36", "task", "3.7 Repair car,\ninvoice insurer", col(7), cy(B_S, 0)),
    Node("t37", "task", "3.8 Pay body shop,\nclose claim", col(8), cy(B_I, 0)),
    Node("ee", "event_end", "Claim settled", col(9), cy(B_I, 0)),
]

edges = [
    # Pool A
    Edge("s", "t11"),
    Edge("t11", "t12", exit=(0.5, 1), entry=(0, 0.5), points=[(col(1), cy(A_C, 0))]),
    Edge("t12", "g_ok"),
    Edge("g_ok", "t13", "Yes"),
    Edge("g_ok", "t11", "No - retake", accent=True, exit=(0.5, 0), entry=(1, 0.5),
         points=[(col(3), cy(A_D, 0))], lx=0.3),
    Edge("t13", "t14"),
    Edge("t14", "d_rep", dotted=True, exit=(0.5, 1), entry=(0.5, 0)),
    Edge("t14", "g_agree", exit=(0.5, 0), entry=(0, 0.5), points=[(col(5), cy(A_D, 0))]),
    Edge("g_agree", "m_a", "Yes"),
    Edge("g_agree", "t21", "No", exit=(0.5, 1), entry=(0.5, 0), lx=-0.5),
    Edge("t21", "m_a", exit=(1, 0.5), entry=(0.5, 1), points=[(col(7), cy(A_P, 0))]),
    Edge("m_a", "t15"),
    Edge("t15", "throwA"),
    # Pool B
    Edge("catchA", "t31"),
    Edge("t31", "t32", exit=(1, 0.5), entry=(0.5, 0), points=[(col(2), cy(B_D, 0))]),
    Edge("t32", "t33", exit=(1, 0.5), entry=(0.5, 1), points=[(col(3), cy(B_I, 0))]),
    Edge("t33", "g_match", exit=(1, 0.5), entry=(0.5, 0), points=[(col(4), cy(B_C, 0))]),
    Edge("g_match", "t34", "Yes"),
    Edge("g_match", "t38", "No", accent=True, exit=(0.5, 1), entry=(0.5, 0), loff=(8, 0)),
    Edge("t38", "g_valid"),
    Edge("g_valid", "t34", "Yes", exit=(0.5, 0), entry=(0.5, 1), loff=(8, 0)),
    Edge("g_valid", "x_ref", "No", accent=True),
    Edge("t34", "t35", exit=(1, 0.5), entry=(0.5, 1), points=[(col(6), cy(B_I, 0))]),
    Edge("t35", "t36"),
    Edge("t36", "t37", exit=(1, 0.5), entry=(0.5, 0), points=[(col(8), cy(B_S, 0))]),
    Edge("t37", "ee"),
]

TITLE = "P1 - Business flow: minor collision to settled claim (to-be, with Crash Twin)"
SUB = "Pool 1: at the scene (minutes, lane cleared)  |  Pool 2: insurance claim (link A continues the case)"
STD = "Road Traffic Rules - CM Resolution No. 172 (12.04.2022)  |  Fault decision stays with police / court; Crash Twin drafts the evidence"

if __name__ == "__main__":
    E.render(title=TITLE, subtitle=SUB, standards=STD, lanes=lanes, nodes=nodes, edges=edges,
             out_path="p1.drawio")
    print("wrote p1.drawio")
