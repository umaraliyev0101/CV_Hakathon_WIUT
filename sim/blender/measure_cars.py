"""Import every car model with its fixes and write sim/configs/assets_cars.json (bbox dims incl. mirrors).

blender -b --factory-startup --python sim/blender/measure_cars.py

length = long horizontal axis, width = other horizontal axis, height = wheels-on-floor to roof, all in the
model's native metres (after the unit fix). build_scene rescales each car uniformly to the config length.
"""
import json
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.blender import build_scene  # noqa: E402

META = {
    "ferrari_458": dict(name="Ferrari 458 Italia spider", author="vicent091036 (Sketchfab), via three.js examples",
                        licence="CC-BY-4.0",
                        fixes="importer materials flat (metallic 0.2/rough 0.8): paint, glass, chrome, tyres re-tuned"),
    "khronos_carconcept": dict(name="Khronos Car Concept", author="Eric Chadwick / Darmstadt Graphics Group (Khronos "
                               "glTF-Sample-Assets)", licence="CC-BY-4.0",
                               fixes="lift +0.159 m (generic: min z -> 0)"),
    "lamborghini_urus": dict(name="Lamborghini Urus", author="Steven Grey (Sketchfab), via pmndrs/examples",
                             licence="CC-BY-NC-4.0", fixes="centimetres: scale 0.01; centred on origin: lift to z=0"),
    "porsche_930_turbo": dict(name="Porsche 911 (930) Turbo 1975", author="probably Lionsharp Studios (Sketchfab), "
                              "via pmndrs/examples", licence="unknown (believed CC-BY-4.0); demo only",
                              fixes="deleted 16 m ground disc + 4 traffic cones (Mesh_2 except plates) and the radio "
                                    "antenna (Mesh_0 islands above 1.45 m)"),
}


def main():
    out = {"_doc": "Car models usable by sim (measured by sim/blender/measure_cars.py after import fixes). "
                   "Dims in metres at native scale; build_scene scales uniformly so length == car.length_m.",
           "excluded": {"car.glb (BabylonJS)": "0.5 m low-poly toy", "wireframes/car.glb (pmndrs)": "actually a robot"},
           "cars": []}
    for model, spec in build_scene.CAR_MODELS.items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        root, objs, paint, (L, W, H) = build_scene.import_car_template(model)
        tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in objs if o.type == "MESH")
        m = META[model]
        out["cars"].append({
            "model": model, "name": m["name"], "file": spec["file"],
            "length_m": round(L, 3), "width_m": round(W, 3), "height_m": round(H, 3),
            "width_over_length": round(W / L, 4), "height_over_length": round(H / L, 4),
            "paint_material": paint.name if paint else None, "native_scale": spec["scale"],
            "triangles": tris, "licence": m["licence"], "author": m["author"], "fixes": m["fixes"],
        })
        print("CAR", json.dumps(out["cars"][-1]))
    path = REPO / "sim" / "configs" / "assets_cars.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
