"""Original procedural warehouse fixtures with real static collision geometry.

No third-party visual assets. Geometry is created before simulation and is
therefore visible to the same cameras and contacts as the robots. Rack/carton
colors deliberately avoid calibrated orange, magenta and cyan marker colors.
"""

import xml.etree.ElementTree as ET


def add_scenery(world, lane_offsets):
    """Add a back-of-cell rack row outside x<=1.5 robot work envelopes.

    All fixtures are fixed world geoms. Physical cartons are placed on shelves;
    these are inventory scenery, not claimed as manipulated task parcels.
    """
    lanes = list(lane_offsets)
    if not lanes:
        return
    colors = {
        "steel": ".12 .16 .20 1",
        "shelf": ".34 .37 .39 1",
        "carton": ".38 .32 .24 1",
        "tape": ".48 .45 .38 1",
        "label": ".82 .83 .80 1",
        "ink": ".09 .10 .11 1",
        "wood": ".29 .25 .20 1",
    }

    def box(name, position, halfsize, material, collision=True):
        return ET.SubElement(
            world,
            "geom",
            name="warehouse_" + name,
            type="box",
            pos=" ".join(map(str, position)),
            size=" ".join(map(str, halfsize)),
            rgba=colors[material],
            contype="1" if collision else "0",
            conaffinity="1" if collision else "0",
            friction=".7 .01 .001",
        )

    for lane_index, y in enumerate(lanes):
        x = 2.65
        prefix = f"rack{lane_index}"
        for side in (-1, 1):
            for depth in (-1, 1):
                box(
                    f"{prefix}_post{side}_{depth}",
                    (x + depth * 0.34, y + side * 0.70, 1.35),
                    (0.035, 0.035, 1.35),
                    "steel",
                )
        for level, z in enumerate((0.15, 1.0, 1.85, 2.68)):
            box(f"{prefix}_shelf{level}", (x, y, z), (0.39, 0.74, 0.035), "shelf")
            if level == 3:
                continue
            for item in range(3):
                cy = y + (item - 1) * 0.44
                h = 0.21 + 0.025 * ((item + level + lane_index) % 3)
                box(
                    f"{prefix}_carton{level}_{item}",
                    (x, cy, z + 0.035 + h),
                    (0.28, 0.19, h),
                    "carton",
                )
                # Thin label/tape are presentation faces on physical carton surfaces.
                face = x - 0.281
                box(
                    f"{prefix}_label{level}_{item}",
                    (face, cy, z + 0.08 + h),
                    (0.001, 0.105, 0.055),
                    "label",
                    False,
                )
                box(
                    f"{prefix}_tape{level}_{item}",
                    (face, cy, z + 0.035 + 2 * h - 0.025),
                    (0.001, 0.19, 0.02),
                    "tape",
                    False,
                )
                for bar in range(9):
                    box(
                        f"{prefix}_barcode{level}_{item}_{bar}",
                        (face - 0.0015, cy - 0.075 + bar * 0.018, z + 0.08 + h),
                        (0.001, 0.003 if bar % 3 else 0.006, 0.035),
                        "ink",
                        False,
                    )
    # A low, empty pallet beyond the back rack: clearly outside the work lanes.
    edge = max(lanes) + 1.35
    for i in range(5):
        box(
            f"pallet_slat{i}",
            (2.6, edge - 0.4 + i * 0.2, 0.13),
            (0.48, 0.075, 0.025),
            "wood",
        )
    for i in (-1, 1):
        box(
            f"pallet_runner{i}",
            (2.6 + i * 0.35, edge, 0.065),
            (0.055, 0.49, 0.04),
            "wood",
        )
