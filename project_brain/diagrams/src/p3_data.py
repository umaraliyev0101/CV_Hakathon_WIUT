"""P3 - Data, training and testing flow (hackathon build).
Three data sources in parallel, test sets frozen BEFORE any training, then a train/evaluate loop."""
import common
import bpmn_emit as E
from bpmn_emit import Node, Edge, col, cy, stack_lanes

PUB, OUR, SYN, EVA, TRN = "#eaf1f7", "#eef4ec", "#f5f2ea", "#f3eef7", "#fdf3e7"
lanes, top = stack_lanes([
    ("Public datasets", 1, PUB), ("Team captures", 1, OUR), ("Synthetic", 1, SYN),
    ("Evaluation", 2, EVA), ("Training", 2, TRN),
])
L_P, L_O, L_S, L_E, L_T = top

nodes = [
    Node("s", "event_start", "Build starts", col(0), cy(L_O, 0)),
    Node("fork", "gateway", "", col(1), cy(L_O, 0), "parallel"),
    Node("p1", "task", "3.1 Download VehiDE,\nCarDD, Nexar clips,\nCISS cases", col(2), cy(L_P, 0)),
    Node("p2", "task", "3.2 Check licences,\nlist every source\nin the README", col(3), cy(L_P, 0)),
    Node("o1", "task", "3.3 Film parked cars;\nstage toy-car crashes\non printed mat", col(2), cy(L_O, 0)),
    Node("o2", "task", "3.4 Label masks\n(CVAT), tape-measure\nground truth", col(3), cy(L_O, 0)),
    Node("y1", "task", "3.5 Import Tashkent\nOSM junctions\ninto CARLA", col(2), cy(L_S, 0)),
    Node("y2", "task", "3.6 Script crashes\nwith known speeds\n+ clauses", col(3), cy(L_S, 0)),
    Node("join", "gateway", "", col(4), cy(L_O, 0), "parallel"),
    Node("fz", "task", "3.7 Freeze test sets\nT1-T7 (never\ntrained on)", col(5), cy(L_E, 0)),
    Node("d_t", "doc", "Test sets\nT1-T7", col(4), cy(L_E, 0)),
    Node("tr", "task", "3.8 Fine-tune damage\nseg + car-model\nclassifier", col(6), cy(L_T, 0)),
    Node("d_w", "doc", "Weights +\ntraining log", col(6), cy(L_T, 1)),
    Node("run", "task", "3.9 Run full pipeline\non every test set", col(7), cy(L_E, 0)),
    Node("g_ok", "gateway", "Targets met?", col(8), cy(L_E, 0), "exclusive", lpos="top"),
    Node("err", "task", "3.10 Error analysis,\nfix weakest module", col(8), cy(L_E, 1)),
    Node("ee", "event_end", "Metrics + failure\ncases published", col(9), cy(L_E, 0)),
]

edges = [
    Edge("s", "fork"),
    Edge("fork", "p1", exit=(0.5, 0), entry=(0, 0.5), points=[(col(1), cy(L_P, 0))]),
    Edge("fork", "o1"),
    Edge("fork", "y1", exit=(0.5, 1), entry=(0, 0.5), points=[(col(1), cy(L_S, 0))]),
    Edge("p1", "p2"), Edge("o1", "o2"), Edge("y1", "y2"),
    Edge("p2", "join", exit=(1, 0.5), entry=(0.5, 0), points=[(col(4), cy(L_P, 0))]),
    Edge("o2", "join"),
    Edge("y2", "join", exit=(1, 0.5), entry=(0.5, 1), points=[(col(4), cy(L_S, 0))]),
    Edge("join", "fz", exit=(1, 0.5), entry=(0.5, 0), points=[(col(5), cy(L_O, 0))]),
    Edge("fz", "d_t", dotted=True),
    Edge("fz", "tr", exit=(0.5, 1), entry=(0, 0.5), points=[(col(5), cy(L_T, 0))]),
    Edge("tr", "run", exit=(0.5, 0), entry=(0, 0.5), points=[(col(6), cy(L_E, 0))]),
    Edge("tr", "d_w", dotted=True, exit=(0.5, 1), entry=(0.5, 0)),
    Edge("run", "g_ok"),
    Edge("g_ok", "ee", "Yes"),
    Edge("g_ok", "err", "No", accent=True, exit=(0.5, 1), entry=(0.5, 0), loff=(8, 0)),
    Edge("err", "tr", accent=True, exit=(0.5, 1), entry=(1, 0.5), points=[(col(8), cy(L_T, 0))]),
]

TITLE = "P3 - Data, training and testing flow"
SUB = "Test sets are fixed before training starts, so no reported number is measured on data the model has seen"
STD = "Public: VehiDE, CarDD (research), Nexar (Nexar ODL), NHTSA CISS (US public)  |  Synthetic: CARLA (MIT) on OSM (ODbL)  |  Own: parked cars + toy crashes, plates/faces blurred"

if __name__ == "__main__":
    E.render(title=TITLE, subtitle=SUB, standards=STD, lanes=lanes, nodes=nodes, edges=edges,
             out_path="p3.drawio")
    print("wrote p3.drawio")
