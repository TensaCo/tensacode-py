"""Sensor-only multi-view fusion and conservative persistent box association.

Coordinates without visible-plane/full-extent evidence do not update the filter.
Missing observations grow uncertainty and eventually force abstention. Quality
scores are heuristic, not calibrated posterior probabilities. This tracker does
not establish semantic identity when identical objects cross or fully occlude.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingConfig:
    measurement_floor_m: float = 0.002
    systematic_floor_m: float = 0.0015
    max_position_std_m: float = 0.008
    max_evidence_age_s: float = 0.5
    acceleration_std_m_s2: float = 0.3
    association_gate_mahalanobis: float = 25.0
    association_max_distance_m: float = 0.4
    identity_margin_m: float = 0.02
    max_track_age_s: float = 10.0


@dataclass
class BoxTrack:
    id: str
    kind: str
    color: str
    dimensions_xyz: tuple
    state: np.ndarray
    covariance: np.ndarray
    timestamp: float
    axis_observed_at: np.ndarray
    observations: int = 1
    sources: list[dict] = field(default_factory=list)

    def snapshot(self, now, config):
        sigma = np.sqrt(np.diag(self.covariance)[:3] + config.systematic_floor_m**2)
        ages = now - self.axis_observed_at
        localized = bool(
            np.all(sigma <= config.max_position_std_m)
            and np.all(ages <= config.max_evidence_age_s)
        )
        quality = float(np.exp(-np.max(sigma) / 0.02) * np.exp(-np.max(ages) / 2))
        if not localized:
            quality = min(quality, 0.49)
        return {
            "id": self.id,
            "kind": self.kind,
            "color": self.color,
            "center_xyz": self.state[:3].tolist(),
            "velocity_xyz": self.state[3:].tolist(),
            "dimensions_xyz": list(self.dimensions_xyz),
            "position_std_xyz_m": sigma.tolist(),
            "axis_evidence_age_s": [float(x) if np.isfinite(x) else None for x in ages],
            "localized": localized,
            "confidence_score": quality,
            "confidence_semantics": "heuristic_fused_geometry_quality_not_calibrated_probability",
            "observations": self.observations,
            "timestamp": float(now),
            "sources": list(self.sources),
        }


class SceneTracker:
    def __init__(self, config=None):
        self.config = config or TrackingConfig()
        self.tracks = []
        self.events = []
        self._next_id = 0
        self._timestamp = -np.inf

    def _predict(self, track, now):
        dt = now - track.timestamp
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        acceleration = (
            self.config.acceleration_std_m_s2 if track.kind != "receiver" else 0.005
        )
        drive = np.vstack((np.eye(3) * dt**2 / 2, np.eye(3) * dt))
        track.state = transition @ track.state
        track.covariance = (
            transition @ track.covariance @ transition.T
            + drive @ drive.T * acceleration**2
        )
        track.timestamp = now

    def _measurement(self, hypothesis):
        axes = np.array(hypothesis.constrained_axes, dtype=int)
        # Shape residual grows the observation noise; ambiguous axes are omitted.
        sigma = np.maximum(
            np.asarray(hypothesis.position_ambiguity_m), self.config.measurement_floor_m
        )
        sigma = np.hypot(sigma, hypothesis.residual_m)
        return axes, np.diag(sigma[axes] ** 2)

    @staticmethod
    def _source(hypothesis):
        return {
            "observation_id": hypothesis.observation_id,
            "timestamp": hypothesis.timestamp,
            "frame_sha256": hypothesis.provenance.get("frame_sha256"),
            "constrained_axes": list(hypothesis.constrained_axes),
            "residual_m": hypothesis.residual_m,
        }

    def _start(self, hypothesis, now):
        axes, noise = self._measurement(hypothesis)
        variance = np.full(3, 0.05**2)
        variance[axes] = np.diag(noise)
        observed = np.full(3, -np.inf)
        observed[axes] = now
        track = BoxTrack(
            f"{hypothesis.kind}_track_{self._next_id}",
            hypothesis.kind,
            hypothesis.color,
            hypothesis.dimensions_xyz,
            np.r_[hypothesis.center_xyz, np.zeros(3)],
            np.diag(np.r_[variance, np.full(3, 0.05**2)]),
            now,
            observed,
            sources=[self._source(hypothesis)],
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def update(self, hypotheses, timestamp):
        now = float(timestamp)
        if not np.isfinite(now) or now < self._timestamp:
            raise ValueError("Finite, monotonic capture timestamps required")
        if any(abs(float(h.timestamp) - now) > 1e-6 for h in hypotheses):
            raise ValueError("Hypothesis timestamps must match capture timestamp")
        self._timestamp = now
        for track in self.tracks:
            self._predict(track, now)
        self.tracks = [
            t
            for t in self.tracks
            if now - np.max(t.axis_observed_at) <= self.config.max_track_age_s
        ]
        cost = np.full((len(self.tracks), len(hypotheses)), 1e9)
        distances = cost.copy()
        for ti, track in enumerate(self.tracks):
            for hi, h in enumerate(hypotheses):
                if (track.kind, track.color) != (h.kind, h.color):
                    continue
                if not np.allclose(track.dimensions_xyz, h.dimensions_xyz):
                    continue
                axes, noise = self._measurement(h)
                if not len(axes):
                    continue
                innovation = np.asarray(h.center_xyz)[axes] - track.state[axes]
                distances[ti, hi] = np.linalg.norm(innovation)
                covariance = track.covariance[np.ix_(axes, axes)] + noise
                score = innovation @ np.linalg.solve(covariance, innovation)
                if (
                    score < self.config.association_gate_mahalanobis
                    and distances[ti, hi] < self.config.association_max_distance_m
                ):
                    cost[ti, hi] = score
        matched = set()
        if cost.size:
            track_indices, observation_indices = linear_sum_assignment(cost)
            for ti, hi in zip(track_indices, observation_indices):
                if cost[ti, hi] >= 1e9:
                    continue
                # A single closest solution is not enough when identical boxes
                # have indistinguishable nearby alternatives in either direction.
                column = np.sort(distances[:, hi][cost[:, hi] < 1e9])
                row = np.sort(distances[ti][cost[ti] < 1e9])
                if any(
                    len(values) > 1
                    and values[1] - values[0] < self.config.identity_margin_m
                    for values in (row, column)
                ):
                    self.events.append(
                        {
                            "timestamp": now,
                            "event": "identity_ambiguous",
                            "observation_id": hypotheses[hi].observation_id,
                        }
                    )
                    matched.add(hi)
                    continue
                h, track = hypotheses[hi], self.tracks[ti]
                axes, noise = self._measurement(h)
                observation = np.eye(6)[axes]
                innovation = np.asarray(h.center_xyz)[axes] - observation @ track.state
                covariance = observation @ track.covariance @ observation.T + noise
                gain = np.linalg.solve(covariance, observation @ track.covariance).T
                track.state += gain @ innovation
                complement = np.eye(6) - gain @ observation
                track.covariance = (
                    complement @ track.covariance @ complement.T + gain @ noise @ gain.T
                )
                track.axis_observed_at[axes] = now
                track.observations += 1
                track.sources.append(self._source(h))
                track.sources = track.sources[-30:]
                matched.add(hi)
        for hi, h in enumerate(hypotheses):
            if hi not in matched:
                self._start(h, now)
        return [track.snapshot(now, self.config) for track in self.tracks]
