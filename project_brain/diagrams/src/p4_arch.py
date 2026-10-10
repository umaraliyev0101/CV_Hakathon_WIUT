"""P4 - System architecture and tech stack (hackathon demo deployment).
Lanes are deployment tiers; boxes are components; arrows are calls / data flow."""
import common
import bpmn_emit as E
from bpmn_emit import Node, Edge, col, cy, stack_lanes

USR, WEB, API, GPU, EXT = "#eaf1f7", "#eef4ec", "#f5f2ea", "#f3eef7", "#fdf3e7"
lanes, top = stack_lanes([
    ("Users", 1, USR), ("Web front-end", 1, WEB), ("Backend server", 2, API),
    ("GPU worker", 2, GPU), ("Data + external", 1, EXT),
])
L_U, L_W, L_B, L_G, L_X = top

nodes = [
    Node("u_drv", "task", "Driver's phone\n(camera + GPS,\nany browser)", col(0), cy(L_U, 0)),
    Node("u_desk", "task", "Insurer / police /\njury browser", col(9), cy(L_U, 0)),
    Node("f_cap", "task", "Capture page\n(Next.js PWA,\nMediaRecorder)", col(1), cy(L_W, 0)),
    Node("f_case", "task", "Case page +\n3D splat viewer\n(three.js)", col(9), cy(L_W, 0)),
    Node("b_api", "task", "FastAPI: cases,\nupload, status", col(2), cy(L_B, 0)),
    Node("b_q", "task", "Redis + RQ\njob queue", col(3), cy(L_B, 0)),
    Node("b_db", "doc", "PostgreSQL\n(cases)", col(3), cy(L_B, 1)),
    Node("b_rep", "task", "Report service:\ncase JSON + PDF", col(8), cy(L_B, 0)),
    Node("g_qc", "task", "M1 Frame sampling,\nQC, plate + face\nblur", col(4), cy(L_G, 0)),
    Node("g_rec", "task", "M2 VGGT-1B-\nCommercial\n(COLMAP fallback)", col(5), cy(L_G, 0)),
    Node("g_cv", "task", "M3-M5 Cars, damage,\nroad (RT-DETR,\nSAM 2, fine-tunes)", col(6), cy(L_G, 0)),
    Node("g_rule", "task", "M7-M8 Physics +\nrules engine\n(Res. 172)", col(7), cy(L_G, 0)),
    Node("g_gs", "task", "M6 gsplat\n3D replay\n(background)", col(5), cy(L_G, 1)),
    Node("x_obj", "task", "Object storage\n(S3 / MinIO): video,\nframes, splats", col(2), cy(L_X, 0)),
    Node("x_osm", "task", "OpenStreetMap\n(Overpass API)", col(6), cy(L_X, 0)),
    Node("x_llm", "task", "LLM API: clause\ntext, UZ / RU\nwording only", col(8), cy(L_X, 0)),
]

edges = [
    Edge("u_drv", "f_cap", "opens link", exit=(1, 0.5), entry=(0.5, 0), points=[(col(1), cy(L_U, 0))]),
    Edge("f_cap", "b_api", "HTTPS upload", exit=(1, 0.5), entry=(0.5, 0), points=[(col(2), cy(L_W, 0))]),
    Edge("b_api", "b_q"),
    Edge("b_api", "b_db", dotted=True, exit=(0.5, 1), entry=(0, 0.5), points=[(col(2), cy(L_B, 1))]),
    Edge("b_api", "x_obj", "store video", dotted=True, exit=(0.5, 1), entry=(0.5, 0), loff=(8, 0)),
    Edge("b_q", "g_qc", "pull job", exit=(1, 0.5), entry=(0.5, 0), points=[(col(4), cy(L_B, 0))]),
    Edge("g_qc", "g_rec"),
    Edge("g_rec", "g_cv"),
    Edge("g_cv", "g_rule"),
    Edge("g_rec", "g_gs", exit=(0.5, 1), entry=(0.5, 0)),
    Edge("g_gs", "x_obj", "splat file", dotted=True, exit=(0.5, 1), entry=(1, 0.5), points=[(col(5), cy(L_X, 0))]),
    Edge("g_cv", "x_osm", "road graph", dotted=True, exit=(0.5, 1), entry=(0.5, 0), loff=(8, 0)),
    Edge("g_rule", "b_rep", "scene + verdicts", exit=(0.5, 0), entry=(0, 0.5), points=[(col(7), cy(L_B, 0))]),
    Edge("b_rep", "x_llm", "wording", dotted=True, exit=(0.5, 1), entry=(0.5, 0), loff=(8, 0)),
    Edge("b_rep", "f_case", "case JSON", exit=(1, 0.5), entry=(0.5, 1), points=[(col(9), cy(L_B, 0))]),
    Edge("f_case", "u_desk", exit=(0.5, 0), entry=(0.5, 1)),
]

TITLE = "P4 - System architecture and tech stack (demo deployment)"
SUB = "One rented GPU server (RTX 4090 / L4 / A100 class, 24 GB+) runs backend and worker; front-end is static on Vercel; deployed from the public repo"
STD = "Python 3, PyTorch, FastAPI, Redis/RQ, PostgreSQL, S3/MinIO  |  Next.js, three.js splat viewer  |  VGGT-1B-Commercial, COLMAP/GLOMAP (BSD), RT-DETR (Apache-2.0), SAM 2 (Apache-2.0), gsplat (Apache-2.0)"

if __name__ == "__main__":
    E.render(title=TITLE, subtitle=SUB, standards=STD, lanes=lanes, nodes=nodes, edges=edges,
             out_path="p4.drawio")
    print("wrote p4.drawio")
