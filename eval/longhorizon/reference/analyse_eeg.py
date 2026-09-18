"""Reference analysis for own/eeg_motor_imagery: sets the accuracies the checker compares against."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/data")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/app/out")
mne.set_log_level("ERROR")


def subject_epochs(files: list[Path]) -> mne.Epochs:
    raws = []
    for f in sorted(files):
        raw = mne.io.read_raw_edf(f, preload=True)
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    events, event_id = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    return mne.Epochs(raw, events, event_id, tmin=-1.0, tmax=4.0, proj=True, baseline=None, preload=True,
                      picks=mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"))


results = {"per_subject": {}}
for subject in (1, 2, 3):
    files = sorted(DATA.glob(f"S{subject:03d}R*.edf"))
    epochs = subject_epochs(files)
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    clf = Pipeline([("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)), ("lda", LinearDiscriminantAnalysis())])
    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=1)
    results["per_subject"][str(subject)] = {"cv_accuracy": round(float(np.mean(scores)), 4), "n_epochs": int(len(y))}
    if subject == 1:
        OUT.mkdir(parents=True, exist_ok=True)
        fig = epochs.compute_psd(fmax=40).plot(show=False)
        fig.savefig(OUT / "psd.png", dpi=110)
        plt.close(fig)

results["mean_cv_accuracy"] = round(float(np.mean([v["cv_accuracy"] for v in results["per_subject"].values()])), 4)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=1))
lines = [f"- subject {s}: {v['cv_accuracy']:.2f} ({v['n_epochs']} epochs)" for s, v in results["per_subject"].items()]
(OUT / "report.md").write_text("# Motor imagery: hands vs feet\n\nPipeline: EEGBCI runs 6/10/14, 7-30 Hz band-pass, epochs -1..4 s,"
                               " CSP (4 components) + LDA, 5-fold cross-validation per subject.\n\n" + "\n".join(lines)
                               + f"\n\nMean cross-validated accuracy {results['mean_cv_accuracy']:.2f}, which is above the 0.5 chance level for this two-class problem.\n")
print(json.dumps(results, indent=1))
