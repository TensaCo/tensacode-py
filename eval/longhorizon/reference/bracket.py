"""Reference build for own/cad_bracket: fixes the volume and bbox the checker compares against.

    /opt/tools/bin/python bracket.py [out.stl out/report.json]
"""
import json
import sys

from build123d import *

PLATE_L, PLATE_W, PLATE_T = 120, 80, 10
BOSS_R, BOSS_H = 20, 20
BORE_R, CB_R, CB_DEPTH = 8, 13, 5

with BuildPart() as bracket:
    Box(PLATE_L, PLATE_W, PLATE_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    chamfer(bracket.edges().filter_by(Axis.Z), 3)
    with Locations((45, 25), (-45, 25), (45, -25), (-45, -25)):
        Hole(radius=4)
    with BuildPart(Plane.XY.offset(PLATE_T)) as boss:
        Cylinder(radius=BOSS_R, height=BOSS_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    add(boss.part)
    Hole(radius=BORE_R)
    with BuildPart(Plane.XY.offset(PLATE_T + BOSS_H - CB_DEPTH)) as counterbore:
        Cylinder(radius=CB_R, height=CB_DEPTH, align=(Align.CENTER, Align.CENTER, Align.MIN))
    add(counterbore.part, mode=Mode.SUBTRACT)

part = bracket.part
size = part.bounding_box().size
summary = {"volume": part.volume, "bbox": [size.X, size.Y, size.Z]}
if len(sys.argv) > 2:
    export_stl(part, sys.argv[1])
    json.dump({"volume_mm3": round(part.volume, 3), "bbox_mm": [round(size.X, 3), round(size.Y, 3), round(size.Z, 3)], "holes": 4}, open(sys.argv[2], "w"), indent=1)
print(json.dumps(summary))
