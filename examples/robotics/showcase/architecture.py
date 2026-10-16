"""Paper-style diagrams of the implemented controller boundaries."""

from html import escape


def architecture(kind, ident):
    marker = f"arrow-{ident}"
    pieces = [
        f'<svg viewBox="0 0 1300 370" role="img" aria-labelledby="title-{ident}" xmlns="http://www.w3.org/2000/svg"><title id="title-{ident}">Controller architecture and observation flow</title><defs><marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#647771"/></marker></defs>'
    ]

    def box(x, y, title, sub, color="sensor"):
        fills = {
            "sensor": "#f1f5f3",
            "algorithm": "#e3f0e9",
            "learned": "#e8edf8",
            "action": "#faf0dd",
        }
        pieces.append(
            f'<g><rect x="{x}" y="{y}" width="180" height="68" rx="3" fill="{fills[color]}" stroke="#8fa49b"/><text x="{x + 90}" y="{y + 28}" text-anchor="middle" font-size="15" font-weight="600">{escape(title)}</text><text x="{x + 90}" y="{y + 49}" text-anchor="middle" font-size="11" fill="#556760">{escape(sub)}</text></g>'
        )

    def arrow(path, dashed=False):
        pieces.append(
            f'<path d="{path}" fill="none" stroke="#647771" stroke-width="1.6" marker-end="url(#{marker})"'
            + (' stroke-dasharray="5 4"' if dashed else "")
            + "/>"
        )

    def note(x, y, text):
        pieces.append(
            f'<text x="{x}" y="{y}" font-size="12" fill="#556760">{escape(text)}</text>'
        )

    if kind not in {"sensor", "transfer", "gr00t", "legacy"}:
        raise ValueError(f"Unknown architecture: {kind}")
    if kind == "legacy":
        box(20, 100, "Simulator state", "Earlier privileged-state demo")
        box(235, 100, "Authored controller", "Navigation + manipulation", "algorithm")
        box(450, 100, "Frozen RL walker", "Supplied Unitree policy", "learned")
        box(665, 100, "IK + joint servo", "Arm and locomotion targets", "algorithm")
        box(880, 100, "G1 robot", "Physics simulation", "action")
        for x in [200, 415, 630, 845]:
            arrow(f"M{x} 134 H{x + 35}")
        note(
            20,
            240,
            "Historical reference; live simulator state is supplied to this earlier controller.",
        )
    elif kind == "gr00t":
        box(20, 100, "Ego RGB", "Rendered camera frames")
        box(235, 100, "Vision encoder", "Pretrained visual features", "learned")
        box(450, 100, "VLA backbone", "GR00T N1.6 · frozen", "learned")
        box(665, 100, "Action head", "Flow-matching action chunks", "learned")
        box(880, 100, "Whole-body control", "Upstream RL controller", "learned")
        box(1095, 100, "G1 + hands", "Physics simulation", "action")
        for x in [200, 415, 630, 845, 1060]:
            arrow(f"M{x} 134 H{x + 35}")
        box(450, 240, "Language + joints", "Instruction + 43 joint angles")
        arrow("M540 240 V168")
        box(880, 240, "Simulator state", "Base attitude / rates + joints")
        arrow("M970 240 V168")
        note(
            20,
            340,
            "Independent reference task. Official WBC consumes simulator state; this is not the TensorCode observation boundary.",
        )
    else:
        transfer = kind == "transfer"
        box(20, 100, "RGB-D camera", "Calibrated workcell view")
        box(235, 100, "Visual detection", "Color mask + backprojection")
        box(450, 100, "Object belief", "Gated Gaussian posterior", "algorithm")
        box(
            665,
            100,
            "ALG.2 attention" if transfer else "Authored trajectory",
            "Phase → demo commands" if transfer else "Reach → grip → lift",
            "algorithm",
        )
        box(880, 100, "Arm IK", "Estimated robot kinematics", "algorithm")
        for x in [200, 415, 630, 845]:
            arrow(f"M{x} 134 H{x + 35}")
        box(20, 240, "IMU / joints / contact", "Proprioceptive measurements")
        box(235, 240, "Robot state estimate", "Stance odometry + fiducial", "algorithm")
        box(665, 240, "Frozen RL walker", "Unitree locomotion policy", "learned")
        box(880, 240, "Joint servo", "Arm + legs + gripper", "action")
        box(1095, 240, "G1 robot", "Physical contact simulation", "action")
        arrow("M200 274 H235")
        arrow("M325 168 V240")
        arrow("M415 274 H665")
        note(490, 264, "Proprioception")
        arrow("M540 168 V223 H755 V240")
        note(560, 217, "Navigation goal")
        arrow("M325 240 V205 H930 V168")
        arrow("M970 168 V240")
        arrow("M845 274 H880")
        arrow("M1060 274 H1095")
        arrow("M1185 308 V333 H110 V308", True)
        note(
            500,
            351,
            "Sensor feedback · live simulator object/root positions do not feed actions",
        )
        if transfer:
            box(665, 5, "One demonstration", "Command trajectory · frozen", "algorithm")
            arrow("M755 73 V100")
            note(
                20,
                30,
                "Comparison: fixed replay keeps the original anchor; retargeting uses the observed pickup anchor.",
            )
        else:
            note(
                20,
                30,
                "Known cube shape/color, camera calibration and pelvis marker are supplied engineering priors.",
            )
    pieces.append("</svg>")
    return "".join(pieces)
