# Compositional scene grounding from actual warehouse RGB-D

The initial and terminal observations of the **failed 160-second v5 trial** identify two parcels and two receiving stations from actual rendered RGB-D. A calibrated workcell region becomes the parent of the observed colored objects. The restricted language parser grounds:

> Move the orange parcel in workcell_0 to the magenta station in workcell_0 then move the green parcel in workcell_0 to the blue station in workcell_0.

The resulting two-task DAG carries observation references and a dependency from the second task to the first. This demonstrates symbolic reference grounding, **not successful execution or reachability**. The physical first task failed and the second remained blocked. Fresh terminal images show both parcels near their pickup regions.

At both initial and terminal frames, the system answers observed parcel count, workcell contents, and the positions of the orange and green parcels. It abstains from “what is on the blue station?” because this observation adapter does not establish physical support/contact. Simulator poses and controller outcomes never enter these answers. `physical-outcome-evaluation-only.json` records the physical failure separately.

The earlier yellow-station smoke capture is retained. Saturated orange pixels also appear yellowish, creating competing shape/color hypotheses. The system refuses to alias ambiguous yellow receiver candidates and abstains from grounding the whole instruction. The blue-station version resolves that catalog ambiguity. Blue segmentation excludes the cyan robot fiducial. These are authored color, metric shape and calibration priors, not learned semantic recognition.

Files:

- `single-twojobs-v2/composition.json`: initial/terminal scene graphs, individual view hypotheses, uncertainty, authored parse trees, grounded tasks, QA and source hashes.
- `observed-scene.png`: actual initial/terminal camera views with projected **fused measured positions**.
- `smoke-v1/`, `smoke-v2/`: short three-second observation checks, not completed robot tasks.
- `results.json`: observation results and separately recorded physical outcomes.
- `resources/`: guarded capture and inference logs.
- `source.zip`, `files.sha256.json`: reproducible source and artifact bindings.

The capture renderer loads recorded physical configurations solely to produce pixels. The observation stage reads only RGB, axial depth, camera calibration, capture timestamps and declared workcell/color-role priors. The workcell hierarchy is a calibrated spatial partition, not a visually discovered building layout. No “on” relation or placement success is invented from the language instruction.
