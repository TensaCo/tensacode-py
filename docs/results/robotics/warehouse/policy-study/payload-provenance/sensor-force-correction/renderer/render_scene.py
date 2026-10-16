"""Presentation settings for a single, shared physical warehouse scene.

All geometry comes from the recorded simulation. Only lighting/materials change.
No downloaded assets or synthetic robot motion are introduced here.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


def prepare_scene(source: Path, output: Path, assets: Path | None = None):
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for field in ("meshdir", "texturedir"):
            value = compiler.get(field)
            if value:
                directory = Path(value)
                if field == "meshdir" and assets:
                    directory = assets / "src/assets/robots/unitree_g1/xmls/assets"
                elif not directory.is_absolute():
                    directory = source.parent / directory
                compiler.set(field, str(directory.resolve()))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for name, attributes in {
        "global": {"offwidth": "1920", "offheight": "1080"},
        "quality": {"shadowsize": "4096", "offsamples": "4"},
        "headlight": {
            "ambient": ".25 .27 .30",
            "diffuse": ".4 .4 .4",
            "specular": ".2 .2 .2",
        },
        "map": {"znear": ".01", "zfar": "80"},
    }.items():
        element = visual.find(name)
        if element is None:
            element = ET.SubElement(visual, name)
        element.attrib.update(attributes)
    world = root.find("worldbody")
    for light in list(world.findall("light")):
        world.remove(light)
    ET.SubElement(
        world,
        "light",
        name="presentation_key",
        pos="2 -3 8",
        dir="-.2 .3 -1",
        directional="true",
        diffuse=".8 .77 .70",
        specular=".3 .3 .3",
        castshadow="true",
    )
    ET.SubElement(
        world,
        "light",
        name="presentation_fill",
        pos="-3 4 6",
        dir=".3 -.2 -1",
        directional="true",
        diffuse=".35 .40 .5",
        castshadow="false",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output)
    return output
